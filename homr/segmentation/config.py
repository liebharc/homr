import os

script_location = os.path.dirname(os.path.realpath(__file__))

segnet_path_onnx = os.path.join(
    script_location, "segnet_308-806067c5351749d387d2831c8a7926c0b7ee0fb4.onnx"
)

segnet_path_onnx_fp16 = os.path.join(
    script_location, "segnet_308-806067c5351749d387d2831c8a7926c0b7ee0fb4_fp16.onnx"
)

segnet_path_torch = os.path.join(
    os.getcwd(),
    "homr",
    "segmentation",
    "segnet_308-806067c5351749d387d2831c8a7926c0b7ee0fb4.pth",
)

segnet_version = os.path.basename(segnet_path_onnx).split("_")[1]

segmentation_version = segnet_version
