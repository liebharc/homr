import os

script_location = os.path.dirname(os.path.realpath(__file__))

model_name = "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f"

segnet_path_onnx = os.path.join(script_location, f"{model_name}.onnx")

segnet_path_onnx_fp16 = os.path.join(script_location, f"{model_name}_fp16.onnx")

segnet_path_torch = os.path.join(
    os.getcwd(),
    "training",
    "architecture",
    "segmentation",
    f"{model_name}.pth",
)

segnet_version = os.path.basename(segnet_path_onnx).split("_")[1]

segmentation_version = segnet_version
