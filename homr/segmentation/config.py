import os

script_location = os.path.dirname(os.path.realpath(__file__))

segnet_path_onnx = os.path.join(
    script_location, "segnet_306-da10d8ae04b8acb8862f3a77ce660ef744026fb7.onnx"
)

segnet_path_onnx_fp16 = os.path.join(
    script_location, "segnet_306-da10d8ae04b8acb8862f3a77ce660ef744026fb7_fp16.onnx"
)

segnet_path_torch = os.path.join(
    os.getcwd(),
    "training",
    "architecture",
    "segmentation",
    "segnet_306-da10d8ae04b8acb8862f3a77ce660ef744026fb7.pth",
)

segnet_version = os.path.basename(segnet_path_onnx).split("_")[1]

segmentation_version = segnet_version
