"""Add beam selection to a batched decoder graph (after fusion).

New input `beam_index` (out_batch,): the cache rows each output row continues. Every
cache input is gathered by it and cut by as many positions as the first cache is longer
than `cache_len` (all caches grow by one position per step), so a longer cache can be
reused to continue from an earlier step.
"""

import onnx
from onnx import TensorProto, helper


def add_cache_selection(path_in: str, path_out: str) -> None:
    model = onnx.load(path_in)
    graph = model.graph
    cache_inputs = [i.name for i in graph.input if i.name.startswith("cache_in")]
    # self and cross attention caches differ in length, and the input rows in number from
    # the output rows: distinct dim names, else ORT may plan one buffer for all of them
    for inp in graph.input:
        if inp.name.startswith("cache_in"):
            dims = inp.type.tensor_type.shape.dim
            dims[0].dim_param = "in_batch"
            dims[2].dim_param = (
                "cross_len" if (int(inp.name[len("cache_in") :]) // 2) % 2 else "seq_len"
            )
    del graph.value_info[:]
    graph.input.append(
        helper.make_tensor_value_info("beam_index", TensorProto.INT64, ["out_batch"])
    )
    nodes = [
        helper.make_node(
            "Constant", [], ["cs_axis2"], value=helper.make_tensor("v", TensorProto.INT64, [1], [2])
        ),
        helper.make_node(
            "Constant", [], ["cs_zero"], value=helper.make_tensor("v", TensorProto.INT64, [1], [0])
        ),
        helper.make_node("Shape", [cache_inputs[0]], ["cs_len0"], start=2, end=3),
        helper.make_node("Sub", ["cs_len0", "cache_len"], ["cs_drop"]),
    ]
    renames = {}
    for name in cache_inputs:
        nodes += [
            helper.make_node("Gather", [name, "beam_index"], [f"{name}_g"], axis=0),
            helper.make_node("Shape", [name], [f"{name}_len"], start=2, end=3),
            helper.make_node("Sub", [f"{name}_len", "cs_drop"], [f"{name}_end"]),
            helper.make_node(
                "Slice", [f"{name}_g", "cs_zero", f"{name}_end", "cs_axis2"], [f"{name}_sel"]
            ),
        ]
        renames[name] = f"{name}_sel"
    for node in graph.node:
        for k, inp in enumerate(node.input):
            if inp in renames:
                node.input[k] = renames[inp]
    for node in reversed(nodes):
        graph.node.insert(0, node)
    onnx.checker.check_model(model)
    onnx.save(model, path_out)
