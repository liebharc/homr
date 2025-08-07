import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxconverter_common import float16
from onnxruntime.quantization.shape_inference import quant_pre_process
import os 

def quantization_int8(model_path, out_path, preprocess=True):
    """
    Dynamic Quantization of an onnx model to int8
    Args:
        model_path(str): Path to onnx model
        out_path(str): Path for saving the quantized model 
        preprocess(bool): For better quantization results it is recommended to use preprocessing. Default True
    """
    if preprocess:
        quant_pre_process(model_path, 'model_preprocessed.onnx') # Preprocess model for better quantization results
        quantize_dynamic('model_preprocessed.onnx', out_path, weight_type=QuantType.QInt8) # Quint8 is slower on x86-64
    else:
        quantize_dynamic(model_path, out_path, weight_type=QuantType.QUInt8)
    
    os.remove('model_preprocessed.onnx')


def quantization_fp16(model_path, out_path):
    """
    Quantization of an onnx model to fp16. Currently not used at all because of higher inference time on CPU.
    Args:
        model_path(str): Path to onnx model
        out_path(str): Path for saving the quantized model 
    """
    model = onnx.load(model_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, out_path)
