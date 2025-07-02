import os

script_location = os.path.dirname(os.path.realpath(__file__))

unet_path = os.path.join(script_location, "unet_160-777e3c95d3c486b058868d6fa60d2a255c1ff629.pth")
segnet_path = os.path.join(
    script_location, "segnet_143-aa8e2f59ef1bc8ce3b9c79672b0fb559f759f782.pth"
)
unet_version = os.path.basename(unet_path).split("_")[1]
segnet_version = os.path.basename(segnet_path).split("_")[1]

segmentation_version = unet_version + "_" + segnet_version
