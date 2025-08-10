from training.convert_onnx.main import convert_all
from training.convert_onnx.transformer.configs import Config
from training.convert_onnx.segmentation.config import segnet_path


convert_all(transformer_path=Config().filepaths.checkpoint, segnet_path=segnet_path)