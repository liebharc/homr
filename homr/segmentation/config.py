import os

script_location = os.path.dirname(os.path.realpath(__file__))

segnet_path = os.path.join(
    script_location, "segnet_155-1240eedca553155b3c75fc9c7f643465383430a0.onnx"
)

segnet_version = os.path.basename(segnet_path).split("_")[1]

segmentation_version = segnet_version
