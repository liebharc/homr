import os
import argparse

from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process


def quantization_int8(
    model_path: str, out_path: str = None, arm_optimized: bool = False
) -> None:
    """
    Dynamic Quantization of an onnx model to int8
    Args:
        model_path(str): Path to onnx model
        out_path(str): Path for saving the quantized model
        preprocess(bool): For better quantization results it
            is recommended to use preprocessing. Default True
    """
    if out_path is None:
        out_path = model_path
    
    quant_pre_process(
            model_path, "model_preprocessed.onnx"
        )  # Preprocess model for better quantization results

    if arm_optimized:
        # Quint8 is slower on x86; faster on arm (sometimes)
        print('arm')
        quantize_dynamic("model_preprocessed.onnx", out_path, weight_type=QuantType.QUInt8)
    else:
        quantize_dynamic(
            "model_preprocessed.onnx", out_path, weight_type=QuantType.QInt8
        )
    os.remove("model_preprocessed.onnx")

def main():
    parser = argparse.ArgumentParser(
        prog="quantization_int8",
        description="Quantize an onnx model to dynamic int8"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the .onnx model"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Path to save the quantized .onnx model"
    )
    parser.add_argument(
        "--arm_optimized",
        action="store_true",
        default=True,
        help="Optimize model for arm"
    )    
    args = parser.parse_args()

    quantization_int8(
        args.model_path, 
        args.out_path,
        args.arm_optimized
    )

if __name__ == '__main__':
    main()
