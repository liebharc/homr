from training.convert_onnx.main import convert_all
from training.architecture.transformer.configs import Config
from training.architecture.segmentation.config import segnet_path


convert_all(transformer_path=Config().filepaths.checkpoint, segnet_path=segnet_path)
