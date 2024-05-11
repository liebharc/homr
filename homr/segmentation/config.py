import os

script_location = os.path.dirname(os.path.realpath(__file__))

unet_path = os.path.join(script_location, "unet_91-0000000000000000000000000000000000000000")
segnet_path = os.path.join(script_location, "segnet_15-feaabda30f22f912471898bf5b44338749440b0b")
unet_version = os.path.basename(unet_path).split("_")[1]
segnet_version = os.path.basename(segnet_path).split("_")[1]

segmentation_version = unet_version + "_" + segnet_version
