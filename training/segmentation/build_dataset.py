import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from homr.simple_logging import eprint
from training.download import (
    download_backgrounds,
    download_cvs_musicma,
    download_deep_scores,
)
from training.segmentation.dense_dataset_definitions import (
    DENSE_DATASET_DEFINITIONS as DEF,
)

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
staff_dataset = os.path.join(dataset_root, "staffs_segmentation")

min_number_of_non_background_pixels = 20

file_limit = -1


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


def get_background_paths(dataset_path: str) -> list[str]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found, download the dataset first.")

    imgs = os.listdir(os.path.join(dataset_path, "OriginalImages"))
    paths = []
    for img in imgs:
        image_path = os.path.join(dataset_path, "OriginalImages", img)
        paths.append(image_path)

    return paths


def create_noisy_canvas(width, height, gray=True):
    """Creates a noisy canvas with uniform noise."""
    if gray:
        return np.random.randint(0, 256, (height, width), dtype=np.uint8)
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def create_blank_canvas(width, height, gray=True):
    """Creates a blank."""
    if gray:
        return np.zeros((height, width), dtype=np.uint8)
    return np.zeros((height, width, 3), dtype=np.uint8)


backgrounds = get_background_paths(download_backgrounds())


def pick_random_background(width=1024, height=1024):
    """Selects a random background image, resizes it, and crops a 1024x1024 region."""
    bg_path = random.choice(backgrounds)
    bg_image = Image.open(bg_path).convert("L")

    # Resize to a random value between 1024 and 4096
    random_size = random.randint(1024, 4096)
    bg_image = bg_image.resize((random_size, random_size), Image.BILINEAR)

    # Crop a random 1024x1024 segment
    x_start = random.randint(0, random_size - width)
    y_start = random.randint(0, random_size - height)
    bg_image = bg_image.crop((x_start, y_start, x_start + width, y_start + height))

    return np.array(bg_image)


def read_image_and_resize(path: str, offset: list[int], gray: bool = True, mask: bool = False):
    image = Image.open(path)
    if gray:
        image = image.convert("L")

    # Define target size
    tar_w, tar_h = 1024, 1024
    image = image.resize((tar_w, tar_h), Image.NEAREST)
    image = np.array(image)

    # Decide whether to use a background (20% chance)
    chance_for_random_background = 0.2
    if mask:
        background = create_blank_canvas(tar_w, tar_h)
    elif random.random() < chance_for_random_background:
        background = pick_random_background(tar_w, tar_h)
        if background is None:
            background = create_noisy_canvas(tar_w, tar_h, gray)
    else:
        background = create_noisy_canvas(tar_w, tar_h, gray)

    # Compute new image position
    top = offset[0]
    left = offset[1]
    bottom = offset[2]
    right = offset[3]

    new_h = tar_h - top - bottom
    new_w = tar_w - left - right

    # Ensure new dimensions are valid
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    image = Image.fromarray(image).resize((new_w, new_h), Image.NEAREST)
    image = np.array(image)

    # Place image onto background
    background[top : top + new_h, left : left + new_w] = image

    return background


def flood_fill(img):
    im_floodfill = img.copy()
    h, w = img.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood fill from point (0,0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert flood filled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine original image with inverted flood-filled image
    filled_image = img | im_floodfill_inv
    return filled_image


def create_staff_mask(img):
    """Morph close a staff image until a number of components remain which likely are the staffs"""
    kernel_size = 1
    kernel_increment = 1
    max_iterations = 20
    initial_number_of_labels, _ = cv2.connectedComponents(img, connectivity=8)
    best_number_of_labels = -1

    for _ in range(max_iterations):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        num_labels, _ = cv2.connectedComponents(closed, connectivity=8)
        if best_number_of_labels > 0 and best_number_of_labels != num_labels:
            # We are now at the point where increasing the kernel merges too many elements
            # so we go back by one step
            kernel_size -= kernel_increment

            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            kernel = np.ones((20, 20), np.uint8)
            closed = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel)
            closed = flood_fill(closed)
            return closed
        elif num_labels <= initial_number_of_labels / 5 and best_number_of_labels < 0:
            best_number_of_labels = num_labels
        kernel_size += kernel_increment

    raise ValueError(
        "Failed to create mask, starting from initial labels " + str(initial_number_of_labels)
    )


def create_staff_line_masks(img):
    """
    Expands the items in the binary image:
    - 4 pixels wider (2 pixels to the left, 2 pixels to the right)
    - 2 pixels thicker (1 pixel up, 1 pixel down)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    expanded_img = cv2.dilate(img, kernel, iterations=1)
    return expanded_img


def extract_narrow_tall_objects(binary_img, max_width=5, min_height=20):
    """
    Find braces and brackets, which are usually tall items.
    """

    binary_img = (binary_img > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    binary_img = cv2.erode(binary_img, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    filtered_img = np.zeros_like(binary_img)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if w <= max_width and h >= min_height:
            filtered_img[labels == i] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    expanded_img = cv2.dilate(filtered_img, kernel, iterations=1)
    return expanded_img


def overlay_mask_on_image(grayscale_img, mask_img):
    """
    Creates an RGB image from the grayscale image and overlays the mask as a transparent red layer.
    The intensity of the red corresponds to the mask value.
    """
    # Convert grayscale image to RGB
    rgb_image = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)

    # Normalize the mask to range [0, 255] if needed
    mask_normalized = cv2.normalize(mask_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create a red channel overlay with transparency
    red_overlay = np.zeros_like(rgb_image, dtype=np.uint8)
    red_overlay[:, :, 2] = mask_normalized  # Assign mask intensity to red channel

    # Blend the original image and the red overlay (alpha blending)
    alpha = mask_normalized.astype(float) / 255.0  # Convert mask to transparency values [0,1]
    blended_image = (rgb_image * (1 - alpha[:, :, None]) + red_overlay * alpha[:, :, None]).astype(
        np.uint8
    )

    return blended_image


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
                raise ValueError("Image has an invalid size " + str(image.shape))

            patches.append(patch)

    return patches


def process_cvc_data(i, image_path, staff_path, symbol_path, staff_dataset):
    """Process one CVC dataset entry and save patches."""
    try:
        offset = get_random_offsets()
        image = 255 - read_image_and_resize(image_path, offset)
        staff_lines = read_image_and_resize(staff_path, offset, mask=True)
        symbol = read_image_and_resize(symbol_path, offset, mask=True)
        staff_mask = create_staff_mask(staff_lines)
        staff_lines = create_staff_line_masks(staff_lines)
        brackets_mask = extract_narrow_tall_objects(symbol)
        total_mask = np.zeros_like(staff_mask)
        total_mask[staff_mask > 0] = 128
        total_mask[staff_lines > 0] = 196
        total_mask[brackets_mask > 0] = 255

        image_patches = split_image_into_patches(image)
        mask_patches = split_image_into_patches(total_mask)

        for j, (image_patch, mask_patch) in enumerate(
            zip(image_patches, mask_patches, strict=False)
        ):
            number_of_pixels_total = mask_patch.shape[0] * mask_patch.shape[1]

            contains_only_background = np.sum(mask_patch != 0) < 0.1 * number_of_pixels_total
            if contains_only_background:
                continue
            cv2.imwrite(os.path.join(staff_dataset, f"{i}_{j}_cvc_img.png"), image_patch)
            cv2.imwrite(os.path.join(staff_dataset, f"{i}_{j}_cvc_mask.png"), mask_patch)
            if file_limit > 0:
                cv2.imwrite(
                    os.path.join(staff_dataset, "debug", f"{i}_{j}_cvc_debug.png"),
                    overlay_mask_on_image(image_patch, mask_patch),
                )
    except Exception as e:
        eprint("Error at ", image_path, symbol_path, e)


def process_deep_score_data(i, image_path, masks_path, staff_dataset):
    """Process one deep score dataset entry and save patches."""
    try:
        offset = get_random_offsets()
        image = read_image_and_resize(image_path, offset)
        masks_color_encoded = read_image_and_resize(masks_path, offset, gray=False, mask=True)
        staff_lines = np.zeros_like(masks_color_encoded, np.uint8)
        staff_lines[masks_color_encoded == DEF.STAFF] = 255
        brackets_mask = extract_narrow_tall_objects(remove_small_horizontal_elements(255 - image))
        staff_mask = create_staff_mask(staff_lines)
        staff_lines = create_staff_line_masks(staff_lines)
        total_mask = np.zeros_like(staff_mask)
        total_mask[staff_mask > 0] = 128
        total_mask[staff_lines > 0] = 196
        total_mask[brackets_mask > 0] = 255

        image_patches = split_image_into_patches(image)
        mask_patches = split_image_into_patches(total_mask)

        for j, (image_patch, mask_patch) in enumerate(
            zip(image_patches, mask_patches, strict=False)
        ):
            contains_only_background = np.sum(mask_patch != 0) < min_number_of_non_background_pixels
            if contains_only_background:
                continue
            cv2.imwrite(os.path.join(staff_dataset, f"{i}_{j}_d2_img.png"), image_patch)
            cv2.imwrite(os.path.join(staff_dataset, f"{i}_{j}_d2_mask.png"), mask_patch)
            if file_limit > 0:
                cv2.imwrite(
                    os.path.join(staff_dataset, "debug", f"{i}_{j}_d2_debug.png"),
                    overlay_mask_on_image(image_patch, mask_patch),
                )
    except Exception as e:
        eprint("Error at ", image_path, masks_path, e)


def get_random_offsets() -> list[int]:
    chance_for_no_offsets = 0.2
    if random.random() < chance_for_no_offsets:
        return [0, 0, 0, 0]

    return [
        random.randint(0, 50),
        random.randint(0, 50),
        random.randint(0, 50),
        random.randint(0, 50),
    ]


def build_dataset():
    if os.path.exists(staff_dataset):
        return staff_dataset
    recreate_dataset()
    return staff_dataset


def recreate_dataset():
    cvc = get_cvc_data_paths(download_cvs_musicma())
    d2 = get_deep_score_data_paths(download_deep_scores())
    if file_limit > 0:
        cvc = cvc[0:100]
        d2 = d2[0:100]

    if os.path.exists(staff_dataset):
        shutil.rmtree(staff_dataset)
    os.makedirs(staff_dataset)
    if file_limit > 0:
        os.makedirs(os.path.join(staff_dataset, "debug"))

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
