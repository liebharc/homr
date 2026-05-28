import os

import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from homr.simple_logging import eprint


def quantization_int8(model_path: str, out_path: str | None = None, preprocess: bool = True) -> str:
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

    if preprocess:
        quant_pre_process(
            model_path, "model_preprocessed.onnx"
        )  # Preprocess model for better quantization results
        quantize_dynamic(
            "model_preprocessed.onnx", out_path, weight_type=QuantType.QInt8
        )  # Quint8 is slower on x86-64
    else:
        quantize_dynamic(model_path, out_path, weight_type=QuantType.QInt8)

    os.remove("model_preprocessed.onnx")
    return out_path


def quantization_fp16(model_path: str, out_path: str | None = None) -> str:
    if out_path is None:
        out_path = model_path

    if os.path.exists(out_path):
        eprint(out_path, "is already present")
        return out_path

    model = onnx.load(model_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, out_path)
    return out_path
