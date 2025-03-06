import os

import numpy as np
from fastai.callback.fp16 import *
from fastai.vision.all import *
from PIL import Image

file_limit = -1
target_size = 512


def get_cvc_data_paths(dataset_path: str) -> list[list[str]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found, download the dataset first.")

    dirs = [
        "curvature",
        "ideal",
        "interrupted",
        "kanungo",
        "rotated",
        "staffline-thickness-variation-v1",
        "staffline-thickness-variation-v2",
        "staffline-y-variation-v1",
        "staffline-y-variation-v2",
        "thickness-ratio",
        "typeset-emulation",
        "whitespeckles",
    ]

    data = []
    for dd in dirs:
        dir_path = os.path.join(dataset_path, dd)
        folders = os.listdir(dir_path)
        for folder in folders:
            data_path = os.path.join(dir_path, folder)
            imgs = os.listdir(os.path.join(data_path, "image"))
            for img in imgs:
                img_path = os.path.join(data_path, "image", img)
                staffline = os.path.join(data_path, "gt", img)
                symbol_path = os.path.join(data_path, "symbol", img)
                data.append([img_path, staffline, symbol_path])

    if file_limit > 0:
        return data[0:file_limit]
    return data


def get_image(path):
    return 255 - np.array(Image.open(path).convert("RGB"))


def create_mask(staff_path, symbol_path):
    staff_img = Image.open(staff_path).convert("1")
    symbol_img = Image.open(symbol_path).convert("1")

    staff_img = np.array(staff_img)
    symbol_img = np.array(symbol_img)

    mask = np.zeros_like(staff_img, np.uint8)  # Initialize mask with zeros (background)
    mask[staff_img > 0] = 1  # Staff pixels → Class 1
    mask[symbol_img > 0] = 2  # Symbol pixels → Class 2

    if not np.any(mask == 1):
        raise ValueError("Error: The generated mask contains no staff (class 1).")

    if not np.any(mask == 2):
        raise ValueError("Error: The generated mask contains no symbol (class 2).")

    return mask


def get_x_fn(x):
    return get_image(x[0])


def get_y_fn(x):
    return create_mask(x[1], x[2])


def export_onnx(model, filename="model.onnx", input_shape=(1, 3, target_size, target_size)):
    model.cpu().eval()
    dummy_input = torch.randn(input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Model exported to {filename}")


dblock = DataBlock(
    blocks=(ImageBlock, MaskBlock(codes=["background", "staff", "symbol"])),
    get_items=get_cvc_data_paths,
    get_x=get_x_fn,
    get_y=get_y_fn,
    splitter=RandomSplitter(),
    item_tfms=[Resize(target_size)],
    batch_tfms=[
        *aug_transforms(
            size=(target_size, target_size),
            max_rotate=10,
            max_zoom=1.2,
            max_lighting=0.2,
            max_warp=0.2,
            p_affine=0.75,
            p_lighting=0.8,
        ),
        Brightness(max_lighting=0.2),
        Contrast(),
        Saturation(),
        Hue(max_hue=0.1),
        Normalize.from_stats(*imagenet_stats),
    ],
)

dls = dblock.dataloaders("datasets/CvcMuscima-Distortions", bs=4, num_workers=4)

# save_examples(dls, num_examples=4)


# squeezenet1_1, resnet34, resnet18

learn = unet_learner(
    dls, squeezenet1_1, metrics=DiceMulti, self_attention=True, act_cls=Mish, n_out=3
)
learn = learn.to_fp16()
learn.fine_tune(5)
export_onnx(learn.model, filename="segnet.onnx")
