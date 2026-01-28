import os

script_location = os.path.dirname(os.path.realpath(__file__))

segnet_path_onnx = os.path.join(
    script_location, "segnet_303-71bee8fac626ac28e8d17ddc33138421edc8e714.onnx"
)

segnet_path_onnx_fp16 = os.path.join(
    script_location, "segnet_303-71bee8fac626ac28e8d17ddc33138421edc8e714.onnx"
)

segnet_path_torch = os.path.join(
    os.getcwd(),
    "training",
    "architecture",
    "segmentation",
    "segnet_303-71bee8fac626ac28e8d17ddc33138421edc8e714.pth",
)

segnet_version = os.path.basename(segnet_path_onnx).split("_")[1]

segmentation_version = segnet_version
