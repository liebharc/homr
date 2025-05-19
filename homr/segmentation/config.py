import os

script_location = os.path.dirname(os.path.realpath(__file__))

unet_path = os.path.join(script_location, "unet_143-0a20cbd35bf31b58b8354e2ec94807ff3c9dc16f")
segnet_path = os.path.join(script_location, "segnet_143-0a20cbd35bf31b58b8354e2ec94807ff3c9dc16f")
unet_version = os.path.basename(unet_path).split("_")[1]
segnet_version = os.path.basename(segnet_path).split("_")[1]

segmentation_version = unet_version + "_" + segnet_version
