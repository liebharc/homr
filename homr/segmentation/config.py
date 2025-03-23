import os

script_location = os.path.dirname(os.path.realpath(__file__))

unet_path = os.path.join(script_location, "unet_91-df68794a7f3420b749780deb1eba938911b3d0d3")
segnet_path = os.path.join(script_location, "segnet_89-f8076e6ee78bf998e291a56647477de80aa19f64")
staffs_path = os.path.join(
    script_location, "staff_123-c1e1e5a24927c12ea42e23c1c465d72fd68ffeff.onnx"
)
unet_version = os.path.basename(unet_path).split("_")[1]
segnet_version = os.path.basename(segnet_path).split("_")[1]

segmentation_version = unet_version + "_" + segnet_version
