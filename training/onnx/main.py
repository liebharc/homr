# ruff: noqa: T201

import os

from homr.segmentation.config import segnet_path_onnx_fp16
from homr.transformer.configs import Config
from training.onnx.convert import (
    convert_decoder,
    convert_encoder,
    convert_segnet,
)
from training.onnx.quantization import quantization_fp16, quantization_int8
from training.onnx.simplify import main as simplify_onnx_model
from training.onnx.split_weights import split_weights


def convert_all() -> None:
    # Warnings might occur
    path_to_segnet = convert_segnet()
    simplify_onnx_model(path_to_segnet)
    print(path_to_segnet)
    path_to_segnet_fp16 = quantization_fp16(path_to_segnet, segnet_path_onnx_fp16)
    print(path_to_segnet_fp16)

    config = Config()
    split_weights(config.filepaths.checkpoint)  # Make sure to the filepath of the transformer!
    path_to_encoder = convert_encoder()
    simplify_onnx_model(path_to_encoder)
    print(path_to_encoder)
    path_to_encoder_fp16 = quantization_fp16(path_to_encoder, config.filepaths.encoder_path_fp16)
    print(path_to_encoder_fp16)
    path_to_decoder = convert_decoder()
    simplify_onnx_model(path_to_decoder)
    print(path_to_decoder)

    # Only the decoder gets quantized.
    # The segnet showed 80% worse performance on x86-64.
    # Only improved size by around 15MB without any speedups
    # (maybe even slowing inference down).
    # FP16 slowed inference speed down (CPU).
    path_to_decoder_fp16 = quantization_fp16(path_to_decoder, config.filepaths.decoder_path_fp16)
    print(path_to_decoder_fp16)
    path_to_decoder_int8 = quantization_int8(path_to_decoder, config.filepaths.decoder_path)
    print(path_to_decoder_int8)
    os.remove("decoder_weights.pt")
    os.remove("encoder_weights.pt")


if __name__ == "__main__":
    # Converts pytorch models used by homr to onnx

    convert_all()
