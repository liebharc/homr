import os

script_location = os.path.dirname(os.path.realpath(__file__))

unet_path = os.path.join(script_location, "unet_144-d6eda3cccad148085bcf62cae34ff1f805e02bab.pth")
segnet_path = os.path.join(
    script_location, "segnet_143-aa8e2f59ef1bc8ce3b9c79672b0fb559f759f782.pth"
)
unet_version = os.path.basename(unet_path).split("_")[1]
segnet_version = os.path.basename(segnet_path).split("_")[1]

segmentation_version = unet_version + "_" + segnet_version
