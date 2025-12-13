import glob
import os
import random
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from homr.resize import calc_target_image_size, resize_image
from homr.simple_logging import eprint
from homr.staff_detection import make_lines_stronger
from homr.type_definitions import NDArray
from training.architecture.segmentation.model import (  # type: ignore
    create_segnet,
    create_unet,
)
from training.run_id import get_run_id
from training.segmentation.build_label import (
    HALF_WHOLE_NOTE,
    fill_hole,
    reconstruct_lines_between_staffs,
)
from training.segmentation.dense_dataset_definitions import (
    CHANNEL_NUM,
    CLASS_CHANNEL_MAP,
)
from training.segmentation.dense_dataset_definitions import (
    DENSE_DATASET_DEFINITIONS as DEF,
)

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A  # noqa


def process_image(path: str, patch_size: int) -> list[tuple[str, int, int]]:
    image = cv2.imread(path)
    if image is None:
        return []  # Skip unreadable files
    height, width, channels = image.shape
    width, height = calc_target_image_size(width, height)

    # Generate patch coordinates
    return [(path, y, x) for y in range(0, height, patch_size) for x in range(0, width, patch_size)]


class SegmentationBaseDataset(BaseDataset[tuple[NDArray, NDArray]]):
    def __init__(self, images: list[str], augmentation: Any | None = None) -> None:
        self.patch_size = 320
        self.ids = []
        self.last_path = ""
        self.last_image: NDArray | None = None
        self.last_mask: NDArray | None = None
        eprint("Preparing files...")
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(partial(process_image, patch_size=self.patch_size), images))
        # Flatten the list of lists
        self.ids = [item for sublist in results for item in sublist]
        eprint(f"Prepared {len(self.ids)} patches.")
        self.augmentation: Any | None = augmentation

    def __getitem__(self, i: int) -> tuple[NDArray, NDArray]:
        # Read the image
        path, y, x = self.ids[i]
        if path == self.last_path and self.last_image is not None and self.last_mask is not None:
            image = self.last_image
            mask = self.last_mask
        else:
            loaded = cv2.imread(path)
            if loaded is None:
                raise ValueError("Failed to read " + path)
            image = loaded
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = self._build_label(image, path)
            image = resize_image(image)
            image = self._prepare_image(image)
            mask = cv2.resize(
                mask, (int(image.shape[1]), int(image.shape[0])), interpolation=cv2.INTER_NEAREST
            )
            self.last_path = path
            self.last_image = image
            self.last_mask = mask
        image = self._get_patch(image, x, y, 255)
        mask = self._get_patch(mask, x, y, 0)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        # image = image.transpose(2, 0, 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.array([image, image, image])
        return image, mask

    def _get_patch(self, image: NDArray, x: int, y: int, pad_value: int) -> NDArray:
        """
        Gets the image patch starting at coordinates x/y. Patches
        always have the size of self.patch_size.
        Edges get padded with white pixels. The image can be grayscale or RGB.
        """
        h, w = image.shape[:2]
        x_start = x
        y_start = y
        x_end = min(x_start + self.patch_size, w)
        y_end = min(y_start + self.patch_size, h)
        patch_content_width = x_end - x_start
        patch_content_height = y_end - y_start

        grayscale_dim = 2
        if len(image.shape) == grayscale_dim:
            # Grayscale, create a zero patch
            patch = np.ones((self.patch_size, self.patch_size), dtype=image.dtype) * pad_value
        else:
            # RGB, create an empty white patch
            patch = np.ones((self.patch_size, self.patch_size, 3), dtype=image.dtype) * pad_value  # type: ignore

        # Copy valid region from source image to patch
        patch[0:patch_content_height, 0:patch_content_width] = image[y_start:y_end, x_start:x_end]

        return patch

    @abstractmethod
    def _build_label(self, image: NDArray, mask: str) -> NDArray:
        pass

    def _prepare_image(self, image: NDArray) -> NDArray:
        return image

    def __len__(self) -> int:
        return len(self.ids)


class D2DenseDataset(SegmentationBaseDataset):

    def _get_mask(self, image_name: str) -> str:
        return image_name.replace(".png", "_seg.png").replace("/images/", "/segmentation/")

    def _build_label(self, image: NDArray, path: str) -> NDArray:
        mask = np.array(Image.open(self._get_mask(path)))
        mask = reconstruct_lines_between_staffs(image, mask)
        # Create a blank mask to remap the class values
        mask_remap = np.zeros_like(mask)

        # Remap the mask according to the dynamically created class map
        for class_value, new_value in sorted(CLASS_CHANNEL_MAP.items()):
            if class_value in HALF_WHOLE_NOTE:
                temp_canvas = fill_hole(mask, class_value)
                mask_remap[temp_canvas == 1] = new_value
            if class_value in DEF.STAFF:
                temp_canvas = np.zeros_like(mask_remap)
                temp_canvas[mask == class_value] = 1
                temp_canvas = make_lines_stronger(temp_canvas, (3, 3))
                mask_remap[temp_canvas == 1] = new_value
            else:
                mask_remap[mask == class_value] = new_value
        return mask_remap


class CvcMuscimaDataset(SegmentationBaseDataset):

    def _get_staff_mask(self, image_name: str) -> str:
        return image_name.replace("/image/", "/gt/")

    def _get_symbol_mask(self, image_name: str) -> str:
        return image_name.replace("/image/", "/symbol/")

    def _build_label(self, image: NDArray, path: str) -> NDArray:
        mask = np.array(Image.open(self._get_staff_mask(path))).astype(np.uint8)
        mask = make_lines_stronger(mask, (3, 3))
        symbol = np.array(Image.open(self._get_symbol_mask(path))).astype(np.uint8)
        mask[symbol == 1] = 2
        return mask

    def _prepare_image(self, image: NDArray) -> NDArray:
        # Invert colors
        return 255 - image


def create_horizontal_gray_gradient(width: int, height: int) -> NDArray:
    # Create a gradient from 0 to 255 horizontally
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    # Convert to 3-channel grayscale image
    return cv2.merge([gradient, gradient, gradient])


def create_vertical_gray_gradient(width: int, height: int) -> NDArray:
    gradient = np.tile(np.linspace(0, 255, height, dtype=np.uint8), (width, 1)).T
    return cv2.merge([gradient, gradient, gradient])


def blend_with_gradient(image: NDArray, gradient: NDArray, alpha: float = 0.3) -> NDArray:
    gradient_resized = cv2.resize(gradient, (image.shape[1], image.shape[0]))
    blended = cv2.addWeighted(image, 1 - alpha, gradient_resized, alpha, 0)
    return blended


class AddGrayGradient(A.ImageOnlyTransform):
    def __init__(
        self,
        alpha: float = 0.3,
        direction: str = "horizontal",
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply, p)
        self.alpha = alpha
        self.direction = direction

    def apply(self, image: NDArray, **params: Any) -> NDArray:
        h, w = image.shape[:2]
        if self.direction == "horizontal":
            gradient = create_horizontal_gray_gradient(w, h)
        else:
            gradient = create_vertical_gray_gradient(w, h)
        return blend_with_gradient(image, gradient, self.alpha)


# training set images augmentation
def get_training_augmentation() -> Any:
    train_transform = [
        AddGrayGradient(alpha=0.4, direction="vertical", p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


def random_split(
    feat_files: list[str], train_val_split: float = 0.1
) -> tuple[list[str], list[str]]:
    random.shuffle(feat_files)
    split_idx = round(train_val_split * len(feat_files))
    train_files = feat_files[split_idx:]
    val_files = feat_files[:split_idx]
    return train_files, val_files


def visualize_dataset(dataset: SegmentationBaseDataset) -> None:
    for i in range(10):
        index = random.randint(0, len(dataset))
        image, mask = dataset[index]
        image = np.transpose(image, (1, 2, 0))
        cv2.imwrite(f"{i}_image.png", image)
        cv2.imwrite(f"{i}_mask.png", 255.0 * mask / (CHANNEL_NUM - 1))
        eprint(f"{i}_mask.png", set(np.unique(mask)))


def train_segnet(visualize: bool = False) -> None:

    script_location = os.path.dirname(os.path.realpath(__file__))
    git_root = Path(script_location).parent.parent.absolute()
    dataset_root = os.path.join(git_root, "datasets")
    dense_root = os.path.join(dataset_root, "ds2_dense")

    run_id = get_run_id()
    model_destination = os.path.join(git_root, "homr", "segmentation", f"segnet_{run_id}.pth")
    eprint("Starting training of ", model_destination)

    images_dir = os.path.join(dense_root, "images")
    images = [os.path.join(images_dir, file) for file in os.listdir(images_dir)]
    train_images, val_images = random_split(images)

    train_dataset = D2DenseDataset(train_images, augmentation=get_training_augmentation())

    if visualize:
        visualize_dataset(train_dataset)

    validation_dataset = D2DenseDataset(val_images, augmentation=None)

    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=False, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=24, shuffle=False, num_workers=4)

    model = create_segnet()

    trainer = pl.Trainer(max_epochs=3, log_every_n_steps=1)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )
    torch.save(model.state_dict(), model_destination)
    eprint(f"Saved model to {model_destination}")


def train_unet(visualize: bool = False) -> None:
    """
    Segnet is layer #5 and #6 have replaced unet.
    """
    eprint("Unet is unused and the training code might be removed in future")

    script_location = os.path.dirname(os.path.realpath(__file__))
    git_root = Path(script_location).parent.parent.absolute()
    dataset_root = os.path.join(git_root, "datasets")
    cvc_root = os.path.join(dataset_root, "CvcMuscima-Distortions")

    run_id = get_run_id()
    model_destination = os.path.join(git_root, "homr", "segmentation", f"unet_{run_id}.pth")
    eprint("Starting training of ", model_destination)

    images = glob.glob(cvc_root + "/**/image/*.png", recursive=True)
    train_images, val_images = random_split(images)

    train_dataset = CvcMuscimaDataset(train_images, augmentation=get_training_augmentation())

    if visualize:
        visualize_dataset(train_dataset)

    validation_dataset = CvcMuscimaDataset(val_images, augmentation=None)

    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=False, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=24, shuffle=False, num_workers=4)

    model = create_unet()

    trainer = pl.Trainer(max_epochs=3, log_every_n_steps=1)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )
    torch.save(model.state_dict(), model_destination)
    eprint(f"Saved model to {model_destination}")


if __name__ == "__main__":
    train_unet(True)
