import glob
import os
import random

import fastai.vision.all as fai
import numpy as np
import torch
from fastai.vision.augment import RandomErasing
from PIL import Image

from homr.simple_logging import eprint
from training.segmentation.build_dataset import build_dataset

file_limit = 5000
image_patch_size = 512


def get_data_paths(dataset_path: str) -> list[list[str]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found, download the dataset first.")
    image_files = glob.glob(os.path.join(dataset_path, "*_img.png"))
    image_files_and_mask = []
    for image_file in image_files:
        image_files_and_mask.append([image_file, image_file.replace("_img.png", "_mask.png")])
    random.shuffle(image_files_and_mask)
    if file_limit > 0:
        return image_files_and_mask[0:file_limit]
    return image_files_and_mask


def get_image(path):
    return np.array(Image.open(path).convert("RGB"))


def create_mask(mask_path):
    mask_img = Image.open(mask_path).convert("L")
    mask = np.zeros_like(mask_img, np.uint8)  # Initialize mask with zeros (background)
    mask[np.isin(mask_img, [128])] = 1  # Staff region → Class 1
    mask[np.isin(mask_img, [255])] = 2  # Bracket pixels → Class 2
    mask[np.isin(mask_img, [196])] = 3  # Staff lines → Class 3
    return mask


def get_x_fn(x):
    return get_image(x[0])


def get_y_fn(x):
    return create_mask(x[1])


def export_onnx(model, filename, input_shape=(1, 3, image_patch_size, image_patch_size)):
    model.cpu().eval()

    # Create dummy input with batch size 1
    dummy_input = torch.randn(input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },  # Allow dynamic batch size
    )


def train_segnet(filename: str):
    dataset_path = build_dataset()
    dblock = fai.DataBlock(
        blocks=(fai.ImageBlock, fai.MaskBlock(codes=["background", "staff", "brackets"])),
        get_items=get_data_paths,
        get_x=get_x_fn,
        get_y=get_y_fn,
        splitter=fai.RandomSplitter(),
        batch_tfms=[
            *fai.aug_transforms(
                size=(image_patch_size, image_patch_size),
                max_rotate=10,
                max_zoom=1.2,
                max_lighting=0.2,
                max_warp=0.2,
                p_affine=0.75,
                p_lighting=0.8,
            ),
            fai.Brightness(max_lighting=0.2),
            fai.Contrast(),
            fai.Saturation(),
            fai.Hue(max_hue=0.1),
            RandomErasing(0.2),
            fai.Normalize.from_stats(*fai.imagenet_stats),
        ],
    )

    dls = dblock.dataloaders(dataset_path, bs=8, num_workers=4)

    # squeezenet1_1, resnet34, resnet18, alexnet

    class_weights = torch.tensor([1.0, 5.0, 5.0, 5.0]).cuda()
    learn = fai.unet_learner(
        dls,
        fai.alexnet,
        metrics=fai.DiceMulti,
        loss_func=torch.nn.CrossEntropyLoss(weight=class_weights),
        # loss_func=FocalLoss(),
        self_attention=True,
        act_cls=fai.Mish,
        n_out=4,
    )
    learn = learn.to_fp16()
    learn.fine_tune(1)
    export_onnx(learn.model, filename)


if __name__ == "__main__":
    filename = "segnet.onnx"
    train_segnet(filename)
    eprint(f"Model exported to {filename}")
