# ruff: noqa: T201

import os

from training.onnx.convert import (
    convert_decoder,
    convert_encoder,
    convert_segnet,
)
from training.onnx.quantization import quantization_int8
from training.onnx.simplify import main as simplify_onnx_model
from training.onnx.split_weights import split_weights


def convert_all(transformer_path: str | None = None, segnet_path: str | None = None) -> None:
    if transformer_path is None and segnet_path is None:
        raise FileExistsError("You did not specify the path of your pytorch models")

    # Warnings might occur
    if segnet_path is not None:
        path_to_segnet = convert_segnet()
        simplify_onnx_model(path_to_segnet)
        print(path_to_segnet)

    if transformer_path is not None:
        split_weights(transformer_path)  # Make sure to the filepath of the transformer!
        path_to_encoder = convert_encoder()
        simplify_onnx_model(path_to_encoder)
        print(path_to_encoder)

        path_to_decoder = convert_decoder()
        simplify_onnx_model(path_to_decoder)

        # Only the decoder gets quantized.
        # The segnet showed 80% worse performance on x86-64.
        # Only improved size by around 15MB without any speedups
        # (maybe even slowing inference down).
        # FP16 slowed inference speed down (CPU).
        quantization_int8(path_to_decoder)
        print(path_to_decoder)

    os.remove("decoder_weights.pt")
    os.remove("encoder_weights.pt")


if __name__ == "__main__":
    # Converts pytorch models used by homr to onnx

    from homr.segmentation.config import segnet_path_torch
    from homr.transformer.configs import Config
    from training.onnx.main import convert_all

    convert_all(transformer_path=Config().filepaths.checkpoint, segnet_path=segnet_path_torch)
