import importlib
import json

modules = [
    "torch",
    "onnx",
    "onnxruntime",
    "segmentation_models_pytorch",
    "pytorch_lightning",
    "transformers",
    "albumentations",
    "tensorboard",
    "onnxscript",
    "x_transformers",
    "rapidocr",
    "cv2",
    "PIL",
    "musicxml",
    "requests",
    "matplotlib",
    "pandas",
]

results = {}
for mod in modules:
    try:
        importlib.import_module(mod)
        results[mod] = True
    except Exception as e:
        results[mod] = False
        results[f"{mod}_error"] = str(e)

print(json.dumps(results, indent=2))
