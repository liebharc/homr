import os

script_location = os.path.dirname(os.path.realpath(__file__))

segnet_path_onnx = os.path.join(
    script_location, "segnet_301-f1c2688efe7efb0aace96b06364022baf5c65e64.onnx"
)

segnet_path_onnx_fp16 = os.path.join(
    script_location, "segnet_301-f1c2688efe7efb0aace96b06364022baf5c65e64_fp16.onnx"
)

segnet_path_torch = os.path.join(
    os.getcwd(),
    "training",
    "architecture",
    "segmentation",
    "segnet_301-f1c2688efe7efb0aace96b06364022baf5c65e64.pth",
)

segnet_version = os.path.basename(segnet_path_onnx).split("_")[1]

segmentation_version = segnet_version
