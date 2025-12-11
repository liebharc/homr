import os

script_location = os.path.dirname(os.path.realpath(__file__))

segnet_path_onnx = os.path.join(
    script_location, "segnet_155-1240eedca553155b3c75fc9c7f643465383430a0.onnx"
)

segnet_path_onnx_fp16 = os.path.join(
    script_location, "segnet_155-1240eedca553155b3c75fc9c7f643465383430a0_fp16.onnx"
)

segnet_path_torch = os.path.join(
    os.getcwd(),
    "training",
    "architecture",
    "segmentation",
    "segnet_155-1240eedca553155b3c75fc9c7f643465383430a0.pth",
)

segnet_version = os.path.basename(segnet_path_onnx).split("_")[1]

segmentation_version = segnet_version
