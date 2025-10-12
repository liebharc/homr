# ruff: noqa: T201

import os
import argparse

from homr.simple_logging import eprint
from training.onnx.convert import (
    convert_decoder,
    convert_encoder,
    convert_segnet,
)
from training.onnx.quantization import quantization_int8
from training.onnx.simplify import main as simplify_onnx_model
from training.onnx.split_weights import split_weights


def convert(transformer_path: str | None = None, segnet_path: str | None = None) -> None:
    if transformer_path is None and segnet_path is None:
        raise FileExistsError("You did not specify a path to the weights")

    # Warnings might occur
    if segnet_path is not None:
        path_onnx_segnet = convert_segnet(segnet_path)
        simplify_onnx_model(path_onnx_segnet)
        eprint(path_onnx_segnet)

    if transformer_path is not None:
        split_weights(transformer_path)  # Make sure to the filepath of the transformer!
        path_onnx_encoder = convert_encoder(transformer_path)
        simplify_onnx_model(path_onnx_encoder)
        eprint(path_onnx_encoder)

        path_onnx_decoder = convert_decoder(transformer_path)
        simplify_onnx_model(path_onnx_decoder)

        # Only the decoder gets quantized.
        # The encoder and segnet need quint8 which has worse inference times on x86 cpus.
        # Although fp16 slowed inference down on cpu, it could be worth it to use fp16.
        quantization_int8(path_onnx_decoder)
        eprint(path_onnx_decoder)

        os.remove("decoder_weights.pt")
        os.remove("encoder_weights.pt")

def main():
    parser = argparse.ArgumentParser(
        prog="Onnx converter",
        description="Convert pytorch models to onnx format"
    )
    parser.add_argument(
        "--path_transformer",
        type=str,
        help="Path to the transformer weights"
    )
    parser.add_argument(
        "--path_segnet",
        type=str,
        help="Path to the segnet weights"
    )
    args = parser.parse_args()

    convert(
        transformer_path=args.path_transformer,
        segnet_path=args.path_segnet
    )


if __name__ == "__main__":
    main()