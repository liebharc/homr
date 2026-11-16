import os

import torch
from torch.export import Dim

from homr.segmentation.config import segnet_path_onnx, segnet_path_torch
from homr.simple_logging import eprint
from homr.transformer.configs import Config
from training.architecture.segmentation.model import create_segnet  # type: ignore
from training.architecture.transformer.decoder import (
    ScoreTransformerWrapper,
    get_score_wrapper,
    init_cache,
)
from training.architecture.transformer.encoder import get_encoder


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model: ScoreTransformerWrapper) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        slurs: torch.Tensor,
        context: torch.Tensor,
        cache_len: torch.Tensor,
        *cache: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
    ]:
        (
            out_rhythms,
            out_pitchs,
            out_lifts,
            out_positions,
            out_articulations,
            out_slurs,
            _x,
            attention,
            *cache,
        ) = self.model(
            rhythms=rhythms,
            pitchs=pitchs,
            lifts=lifts,
            articulations=articulations,
            slurs=slurs,
            context=context,
            cache_len=cache_len,
            mask=None,
            cache=cache,
            return_center_of_attention=True,
        )
        return (
            out_rhythms,
            out_pitchs,
            out_lifts,
            out_positions,
            out_articulations,
            out_slurs,
            attention,
            *cache,
        )


def convert_encoder(overwrite: bool) -> str | None:
    """
    Converts the encoder to onnx
    """
    config = Config()

    path_out = config.filepaths.encoder_path

    if os.path.exists(path_out) and not overwrite:
        eprint(
            f"Encoder already exists at {path_out}. Use --overwrite to overwrite the existing file."
        )
        return None

    # Get Encoder
    model = get_encoder(config)

    # Load weights
    model.load_state_dict(
        torch.load(r"encoder_weights.pt", weights_only=True, map_location=torch.device("cpu")),
        strict=True,
    )

    # Set eval mode
    model.eval()

    # Prepare input tensor
    input_tensor = torch.randn(1, 1, config.max_height, config.max_width).float()

    # Export to onnx
    torch.onnx.export(
        model,
        input_tensor,  # type: ignore
        path_out,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    return path_out


def _center_of_attention_batched(
    self: ScoreTransformerWrapper, intermediates: list[torch.Tensor]
) -> torch.Tensor:
    """ScoreTransformerWrapper.get_center_of_attention for a batch: (batch, 2)."""
    filtered = [
        intermediates[-5].post_softmax_attn[:, :, -1, :],  # type: ignore[attr-defined]
        intermediates[-3].post_softmax_attn[:, :, -1, :],  # type: ignore[attr-defined]
        intermediates[-1].post_softmax_attn[:, :, -1, :],  # type: ignore[attr-defined]
    ]
    attention = torch.mean(torch.stack(filtered), dim=0).mean(dim=1)  # (batch, keys)
    h, w = self.attention_height, self.attention_width
    image = attention[:, : h * w].reshape(-1, h, w)
    weights = torch.clamp(image, min=1e-4).pow(4.0)
    y = torch.linspace(0.5, h - 0.5, h, device=weights.device, dtype=weights.dtype)
    x = torch.linspace(0.5, w - 0.5, w, device=weights.device, dtype=weights.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    mass = weights.sum(dim=(1, 2))
    row = (weights * yy).sum(dim=(1, 2)) / mass
    col = (weights * xx).sum(dim=(1, 2)) / mass
    return torch.stack([col * self.patch_size, row * self.patch_size], dim=1)


def convert_decoder(overwrite: bool) -> str | None:
    """
    Converts the decoder to onnx, with a dynamic batch dimension: homr decodes several
    readings of a staff in one call (see homr/transformer/decoder_inference.py).
    """
    config = Config()
    ScoreTransformerWrapper.get_center_of_attention = _center_of_attention_batched  # type: ignore[method-assign]
    model = get_score_wrapper(config, attn_flash=False)
    model.eval()

    path_out = config.filepaths.decoder_path

    if os.path.exists(path_out) and not overwrite:
        eprint(
            f"Decoder already exists at {path_out}. Use --overwrite to overwrite the existing file."
        )
        return None

    model.load_state_dict(
        torch.load(r"decoder_weights.pt", weights_only=True, map_location=torch.device("cpu")),
        strict=True,
    )

    # Using a wrapper model with a custom forward() function
    wrapped_model = DecoderWrapper(model)
    wrapped_model.eval()

    # Create input data
    # Mask is not used since it caused problems with the tensor size
    batch = 2
    kv_cache, kv_input_names, kv_output_names, dynamic_axes, cache_length = init_cache(
        0, torch.device("cpu")
    )
    rhythms = torch.randint(0, config.num_rhythm_tokens, (batch, 1)).long()
    pitchs = torch.randint(0, config.num_pitch_tokens, (batch, 1)).long()
    lifts = torch.randint(0, config.num_lift_tokens, (batch, 1)).long()
    articulations = torch.randint(0, config.num_articulation_tokens, (batch, 1)).long()
    slurs = torch.randint(0, config.num_slur_tokens, (batch, 1)).long()
    cache_len = torch.tensor([cache_length]).long()
    cache = [c.expand(batch, -1, -1, -1).contiguous() for c in kv_cache]
    context = torch.randn((batch, 1280, config.encoder_dim)).float()

    for name in ["rhythms", "pitchs", "lifts", "articulations", "slurs"]:
        dynamic_axes[name] = {0: "batch"}
    dynamic_axes["context"] = {0: "batch", 1: "cache_exists"}
    for name in kv_input_names:
        dynamic_axes[name] = {0: "batch", 2: "seq_len"}
    output_names = [
        "out_rhythms",
        "out_pitchs",
        "out_lifts",
        "out_positions",
        "out_articulations",
        "out_slurs",
        "attention",
        *kv_output_names,
    ]
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    torch.onnx.export(
        wrapped_model,
        (rhythms, pitchs, lifts, articulations, slurs, context, cache_len, *cache),
        path_out,
        input_names=[
            "rhythms",
            "pitchs",
            "lifts",
            "articulations",
            "slurs",
            "context",
            "cache_len",
            *kv_input_names,
        ],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )
    return path_out


def convert_segnet(overwrite: bool) -> str | None:
    """
    Converts the segnet model to onnx.
    """
    path_out = segnet_path_onnx

    if os.path.exists(path_out) and not overwrite:
        eprint(
            f"Segnet already exists at {path_out}. Use --overwrite to overwrite the existing file."
        )
        return None

    model = create_segnet()
    model.load_state_dict(torch.load(segnet_path_torch, weights_only=True), strict=True)
    model.eval()

    # Input dimension is 1x3x320x320
    sample_inputs = torch.randn(8, 3, 320, 320)

    torch.onnx.export(
        model,
        sample_inputs,  # type: ignore
        path_out,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # dyamic axes are required for dynamic batch_size
        dynamic_shapes={"image": (Dim("batch_size"), 3, 320, 320)},
        dynamo=True,
        external_data=False,
    )
    return path_out
