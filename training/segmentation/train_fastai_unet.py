import os

import numpy as np
from fastai.callback.fp16 import *
from fastai.vision.all import *
from PIL import Image

from training.segmentation.build_label import build_label
from training.segmentation.constant_min import CHANNEL_NUM

file_limit = -1
target_size = 512


def get_deep_score_data_paths(dataset_path: str) -> list[list[str]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found, download the dataset first.")

    imgs = os.listdir(os.path.join(dataset_path, "images"))
    paths = []
    for img in imgs:
        image_path = os.path.join(dataset_path, "images", img)
        seg_path = os.path.join(dataset_path, "segmentation", img.replace(".png", "_seg.png"))
        paths.append([image_path, seg_path])

    if file_limit > 0:
        return paths[0:file_limit]
    return paths


def get_image(path):
    return np.array(Image.open(path).convert("RGB"))


def create_mask(segmentation_path):
    strengthen_channels = {
        1: (5, 5),
    }
    mask = build_label(segmentation_path, strenghten_channels=strengthen_channels).astype(np.uint8)

    # Reverse the last dimension (channel order)
    reversed_mask = mask[..., ::-1]

    # Find the first nonzero channel index in reversed order
    reduced_mask = np.argmax(reversed_mask > 0, axis=-1)

    # Convert from reversed index to original index
    reduced_mask = mask.shape[-1] - 1 - reduced_mask

    reduced_mask = reduced_mask.astype(np.uint8)

    if not np.any(reduced_mask == 2) and not np.any(reduced_mask == 3):
        print("WARN: The generated mask only contains symbols for class 1: " + segmentation_path)

    return reduced_mask


def get_x_fn(x):
    return get_image(x[0])


def get_y_fn(x):
    return create_mask(x[1])


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
    get_items=get_deep_score_data_paths,
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

dls = dblock.dataloaders("datasets/ds2_dense", bs=4, num_workers=4)

# save_examples(dls, num_examples=4)


# squeezenet1_1, resnet34, resnet18

learn = unet_learner(
    dls, squeezenet1_1, metrics=DiceMulti, self_attention=True, act_cls=Mish, n_out=CHANNEL_NUM
)
learn = learn.to_fp16()
learn.fine_tune(5)
export_onnx(learn.model, filename="unet.onnx")
