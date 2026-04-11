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
from training.onnx.fuse import fuse_decoder

def segnet_to_onnx() -> None:
    """
    Converts and Quantizes the Segnet.
    """
    path_to_segnet = convert_segnet()
    simplify_onnx_model(path_to_segnet)
    quantization_fp16(path_to_segnet, segnet_path_onnx_fp16)


def tromr_to_onnx() -> None:
    """
    Converts and Quantizes the Transformer (results in an encoder and a decoder).
    """
    config = Config()
    split_weights(config.filepaths.checkpoint)

    # Encoder
    path_to_encoder = convert_encoder()
    simplify_onnx_model(path_to_encoder)
    quantization_fp16(path_to_encoder, config.filepaths.encoder_path_fp16)

    # Decoder
    path_to_decoder = convert_decoder()
    simplify_onnx_model(path_to_decoder)

    # Conv-heavy models should use static quantization
    quantization_fp16(path_to_decoder, config.filepaths.decoder_path_fp16)
    quantization_int8(path_to_decoder)

    fuse_decoder(path_to_decoder)
    fuse_decoder(config.filepaths.decoder_path_fp16)

    os.remove("decoder_weights.pt")
    os.remove("encoder_weights.pt")


if __name__ == "__main__":
    # Converts pytorch models used by homr to onnx
    tromr_to_onnx()
    segnet_to_onnx()
