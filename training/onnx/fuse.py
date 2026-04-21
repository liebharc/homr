import re
from dataclasses import dataclass

import onnx
from onnxruntime.transformers.fusion_skiplayernorm import FusionSkipLayerNormalization
from onnxruntime.transformers.onnx_model import OnnxModel

from homr.simple_logging import eprint
from homr.transformer.configs import Config


@dataclass()
class DecoderAttentionBlock:
    layer_id: int
    cache_index: int
    q_input: str
    k_input: str
    v_input: str
    attention_output: str

    @property
    def is_self_attention(self) -> bool:
        return self.layer_id % 3 == 0


def _add_contrib_opset(model: onnx.ModelProto) -> None:
    if any(opset.domain == "com.microsoft" for opset in model.opset_import):
        return
    model.opset_import.extend([onnx.helper.make_operatorsetid("com.microsoft", 1)])


def _collect_decoder_attention_blocks(model: OnnxModel) -> list[DecoderAttentionBlock]:
    output_name_to_node = model.output_name_to_node()
    blocks: list[DecoderAttentionBlock] = []

    for cache_index in range(0, 32, 2):
        cache_output = f"cache_out{cache_index}"
        cache_node = output_name_to_node.get(cache_output)
        if cache_node is None:
            continue

        match = re.search(r"layers\.(\d+)\.1", cache_node.name)
        if match is None:
            continue

        layer_id = int(match.group(1))
        prefix = f"/model/attn_layers/layers.{layer_id}.1"

        reshape_q = output_name_to_node.get(f"{prefix}/split_q_heads/Reshape_output_0")
        reshape_k = output_name_to_node.get(f"{prefix}/split_k_heads/Reshape_output_0")
        reshape_v = output_name_to_node.get(f"{prefix}/split_v_heads/Reshape_output_0")
        merge = output_name_to_node.get(f"{prefix}/merge_heads/Reshape_output_0")

        if reshape_q is None or reshape_k is None or reshape_v is None or merge is None:
            continue

        blocks.append(
            DecoderAttentionBlock(
                layer_id=layer_id,
                cache_index=cache_index,
                q_input=reshape_q.input[0],
                k_input=reshape_k.input[0],
                v_input=reshape_v.input[0],
                attention_output=merge.output[0],
            )
        )

    return blocks


def _fuse_decoder_attention_to_mha(model: OnnxModel, num_heads: int) -> None:
    graph = model.graph()
    output_name_to_node = model.output_name_to_node()
    blocks = _collect_decoder_attention_blocks(model)

    for block in blocks:
        merge_node = output_name_to_node[block.attention_output]
        present_k = output_name_to_node[f"cache_out{block.cache_index}"]
        present_v = output_name_to_node[f"cache_out{block.cache_index + 1}"]

        mha_node = onnx.helper.make_node(
            "MultiHeadAttention",
            inputs=[
                block.q_input,
                block.k_input,
                block.v_input,
                "",
                "",
                "",
                f"cache_in{block.cache_index}",
                f"cache_in{block.cache_index + 1}",
            ],
            outputs=[
                block.attention_output,
                f"cache_out{block.cache_index}",
                f"cache_out{block.cache_index + 1}",
            ],
            name=f"layers_{block.layer_id}_1_MultiHeadAttention",
            domain="com.microsoft",
        )
        mha_node.attribute.append(onnx.helper.make_attribute("num_heads", num_heads))
        if block.is_self_attention:
            mha_node.attribute.append(onnx.helper.make_attribute("unidirectional", 1))

        graph.node.extend([mha_node])
        graph.node.remove(merge_node)
        graph.node.remove(present_k)
        graph.node.remove(present_v)

    if blocks:
        model.topological_sort()
        model.prune_graph()


def _fuse_skip_layer_norm(model: OnnxModel) -> None:
    FusionSkipLayerNormalization(model).apply()
    model.topological_sort()
    model.prune_graph()


def fuse_decoder(
    model_path: str,
    output_path: str | None = None,
    fuse_skip_layer_norm: bool = True,
) -> None:
    config = Config()
    num_heads = config.decoder_heads

    if output_path is None:
        output_path = model_path

    model_proto = onnx.load(model_path)
    _add_contrib_opset(model_proto)
    model = OnnxModel(model_proto)

    _fuse_decoder_attention_to_mha(model, num_heads=num_heads)

    if fuse_skip_layer_norm:
        _fuse_skip_layer_norm(model)

    onnx.checker.check_model(model.model)
    onnx.save(model.model, output_path)
    eprint(f"Graph fusion completed. Saved model to: {output_path}")
