# ruff: noqa: T201

import argparse
import os

from homr.segmentation.config import segnet_path_onnx_fp16
from homr.simple_logging import eprint
from homr.transformer.configs import Config
from training.onnx.convert import (
    convert_decoder,
    convert_encoder,
    convert_segnet,
)
from training.onnx.fuse import fuse_decoder
from training.onnx.quantization import quantization_fp16, quantization_int8
from training.onnx.simplify import main as simplify_onnx_model
from training.onnx.split_weights import split_weights


def segnet_to_onnx(overwrite: bool) -> None:
    """
    Converts and Quantizes the Segnet.
    """
    path_to_segnet = convert_segnet(overwrite)

    if path_to_segnet:
        simplify_onnx_model(path_to_segnet)
        quantization_fp16(path_to_segnet, segnet_path_onnx_fp16)


def tromr_to_onnx(overwrite: bool) -> None:
    """
    Converts and Quantizes the Transformer (results in an encoder and a decoder).
    """
    config = Config()
    split_weights(config.filepaths.checkpoint)

    # Encoder
    path_to_encoder = convert_encoder(overwrite)
    if path_to_encoder:
        simplify_onnx_model(path_to_encoder)
        quantization_fp16(path_to_encoder, config.filepaths.encoder_path_fp16)

    # Decoder
    path_to_decoder = convert_decoder(overwrite)
    if path_to_decoder:
        simplify_onnx_model(path_to_decoder)

        # Conv-heavy models should use static quantization
        try:
            quantization_fp16(path_to_decoder, config.filepaths.decoder_path_fp16)
            quantization_int8(path_to_decoder)
        except Exception as e:
            eprint(f"Quantization failed. {e}")
        try:
            fuse_decoder(path_to_decoder)
            fuse_decoder(config.filepaths.decoder_path_fp16)
        except Exception as e:
            eprint(f"Fusing failed. {e}")

    os.remove("decoder_weights.pt")
    os.remove("encoder_weights.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing models")
    parser.add_argument("--segnet", action="store_true", help="Only convert segnet")
    parser.add_argument("--tromr", action="store_true", help="Only convert tromr")

    args = parser.parse_args()

    both_false = not args.segnet and not args.tromr

    if args.segnet or both_false:
        segnet_to_onnx(args.overwrite)

    if args.tromr or both_false:
        tromr_to_onnx(args.overwrite)


if __name__ == "__main__":
    main()
