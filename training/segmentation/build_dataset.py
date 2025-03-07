import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from homr.resize import calc_target_image_size
from homr.simple_logging import eprint
from training.download import download_cvs_musicma, download_deep_scores
from training.segmentation.dense_dataset_definitions import (
    DENSE_DATASET_DEFINITIONS as DEF,
)

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
staff_dataset = os.path.join(dataset_root, "staffs_segmentation")


def get_cvc_data_paths(dataset_path: str) -> list[list[str]]:
    """Returns: [image, staff, symbol]"""
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

    return data


def get_deep_score_data_paths(dataset_path: str) -> list[list[str]]:
    """Returns: [image, segmentation] where segmentation is color coded"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found, download the dataset first.")

    imgs = os.listdir(os.path.join(dataset_path, "images"))
    paths = []
    for img in imgs:
        image_path = os.path.join(dataset_path, "images", img)
        seg_path = os.path.join(dataset_path, "segmentation", img.replace(".png", "_seg.png"))
        paths.append([image_path, seg_path])

    return paths


def read_image_and_resize(path: str, gray: bool = True):
    image = Image.open(path)
    if gray:
        image = image.convert("L")
    tar_w, tar_h = calc_target_image_size(image)
    if tar_w == image.size[0] and tar_h == image.size[1]:
        return np.array(image)
    return np.array(image.resize((tar_w, tar_h), Image.NEAREST))


def create_staff_mask(img):
    """Morph close a staff image until a number of components remain which likely are the staffs"""
    kernel_size = 5
    kernel_increment = 5
    stable_count = 0
    max_iterations = 20
    initial_number_of_labels, _ = cv2.connectedComponents(img, connectivity=8)
    last_num_labels = -1

    for _ in range(max_iterations):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        num_labels, _ = cv2.connectedComponents(closed, connectivity=8)
        if num_labels < initial_number_of_labels / 5 and last_num_labels == num_labels:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count > 1:
            return closed
        kernel_size += kernel_increment
        last_num_labels = num_labels

    raise ValueError("Failed to create mask")


def extract_narrow_tall_objects(binary_img, max_width=30, min_height=100):
    """
    Find braces and brackets, which are usually tall items.
    """

    binary_img = (binary_img > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    filtered_img = np.zeros_like(binary_img)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if w <= max_width and h >= min_height:
            filtered_img[labels == i] = 255

    return filtered_img


def remove_small_horizontal_elements(binary_img, max_horizontal_length=10):
    kernel = np.ones((max_horizontal_length, 1), np.uint8)
    eroded = cv2.erode(binary_img, kernel, iterations=1)
    cleaned_img = cv2.dilate(eroded, kernel, iterations=1)
    return cleaned_img


def split_image_into_patches(image: np.array, patch_size=512) -> list:
    patches = []
    h, w = image.shape[:2]

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i : i + patch_size, j : j + patch_size]

            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = np.pad(
                    patch,
                    ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)),
                    mode="constant",
                    constant_values=256,
                )

            patches.append(patch)

    return patches


def process_cvc_data(i, image_path, staff_path, symbol_path, staff_dataset):
    """Process one CVC dataset entry and save patches."""
    try:
        image = 255 - read_image_and_resize(image_path)
        staff_lines = read_image_and_resize(staff_path)
        symbol = read_image_and_resize(symbol_path)
        staff_mask = create_staff_mask(staff_lines)
        brackets_mask = extract_narrow_tall_objects(symbol)
        total_mask = np.zeros_like(staff_mask)
        total_mask[staff_mask > 0] = 128
        total_mask[brackets_mask > 0] = 255

        image_patches = split_image_into_patches(image)
        mask_patches = split_image_into_patches(total_mask)

        for j, (image_patch, mask_patch) in enumerate(
            zip(image_patches, mask_patches, strict=False)
        ):
            cv2.imwrite(os.path.join(staff_dataset, f"{i}_{j}_cvc_img.png"), image_patch)
            cv2.imwrite(os.path.join(staff_dataset, f"{i}_{j}_cvc_mask.png"), mask_patch)
    except Exception as e:
        eprint("Error at ", image_path, e)


def process_deep_score_data(i, image_path, masks_path, staff_dataset):
    """Process one deep score dataset entry and save patches."""
    try:
        image = read_image_and_resize(image_path)
        masks_color_encoded = read_image_and_resize(masks_path, gray=False)
        staff_lines = np.zeros_like(masks_color_encoded, np.uint8)
        staff_lines[masks_color_encoded == DEF.STAFF] = 255
        brackets_mask = extract_narrow_tall_objects(remove_small_horizontal_elements(255 - image))
        staff_mask = create_staff_mask(staff_lines)
        total_mask = np.zeros_like(staff_mask)
        total_mask[staff_mask > 0] = 128
        total_mask[brackets_mask > 0] = 255

        image_patches = split_image_into_patches(image)
        mask_patches = split_image_into_patches(total_mask)

        for j, (image_patch, mask_patch) in enumerate(
            zip(image_patches, mask_patches, strict=False)
        ):
            cv2.imwrite(os.path.join(staff_dataset, f"{i}_{j}_d2_img.png"), image_patch)
            cv2.imwrite(os.path.join(staff_dataset, f"{i}_{j}_d2_mask.png"), mask_patch)
    except Exception as e:
        eprint("Error at ", image_path, e)


def build_dataset():
    if os.path.exists(staff_dataset):
        return staff_dataset
    recreate_dataset()
    return staff_dataset


def recreate_dataset():
    cvc = get_cvc_data_paths(download_cvs_musicma())
    d2 = get_deep_score_data_paths(download_deep_scores())

    if os.path.exists(staff_dataset):
        shutil.rmtree(staff_dataset)
    os.makedirs(staff_dataset)

    with ProcessPoolExecutor() as executor:
        futures = []

        for i, [image_path, staff_path, symbol_path] in enumerate(cvc):
            futures.append(
                executor.submit(
                    process_cvc_data, i, image_path, staff_path, symbol_path, staff_dataset
                )
            )

        for i, [image_path, masks_path] in enumerate(d2):
            futures.append(
                executor.submit(process_deep_score_data, i, image_path, masks_path, staff_dataset)
            )

        for future in tqdm(futures, desc="Processing images", total=len(futures)):
            future.result()


if __name__ == "__main__":
    recreate_dataset()
