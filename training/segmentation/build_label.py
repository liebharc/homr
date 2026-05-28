import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

from homr.simple_logging import eprint
from homr.type_definitions import NDArray
from training.segmentation.dense_dataset_definitions import (
    DENSE_DATASET_DEFINITIONS as DEF,
)

HALF_WHOLE_NOTE = DEF.NOTEHEADS_HOLLOW + DEF.NOTEHEADS_WHOLE + [42]


def fill_hole(gt: NDArray, tar_color: int) -> NDArray:
    if tar_color not in HALF_WHOLE_NOTE:
        raise ValueError("The color is not a notehead color")
    tar = np.where(gt == tar_color, 1, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(tar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)

        # Scan by row
        for yi in range(y, y + h):
            cur = x
            cand_y = []
            cand_x = []
            while cur <= x + w:
                if tar[yi, cur] > 0:
                    break
                cur += 1
            while cur <= x + w:
                if tar[yi, cur] == 0:
                    break
                cur += 1
            while cur <= x + w:
                if tar[yi, cur] > 0:
                    break
                cand_y.append(yi)
                cand_x.append(cur)
                cur += 1
            if cur <= x + w:
                tar[np.array(cand_y), np.array(cand_x)] = 1

        # Scan by column
        for xi in range(x, x + w):
            cur = y
            cand_y = []
            cand_x = []
            while cur <= y + h:
                if tar[cur, xi] > 0:
                    break
                cur += 1
            while cur <= y + h:
                if tar[cur, xi] == 0:
                    break
                cur += 1
            while cur <= y + h:
                if tar[cur, xi] > 0:
                    break
                cand_y.append(cur)
                cand_x.append(xi)
                cur += 1
            if cur <= y + h:
                tar[np.array(cand_y), np.array(cand_x)] = 1

    return tar


def find_example(
    dataset_path: str, color: int, max_count: int = 100, mark_value: int = 255
) -> NDArray | None:
    files = os.listdir(dataset_path)
    random.shuffle(files)
    for ff in files[:max_count]:
        path = os.path.join(dataset_path, ff)
        img = Image.open(path)
        arr = np.array(img).astype(np.uint8)
        if color in arr:
            return np.where(arr == color, mark_value * np.ones_like(arr), arr)

    return None


def reconstruct_lines_between_staffs(image: NDArray, mask: NDArray) -> NDArray:
    """
    Step 1: Find tallest barline in the mask
    Step 2: Find vertical black lines in the image with height >= 2x tallest barline
    Return a binary mask highlighting those tall lines
    """
    height, width = mask.shape
    result = np.zeros((height, width), dtype=np.uint8)

    barline_mask = np.isin(mask, DEF.ALL_BARLINES).astype(np.uint8) * 255

    # Use connected components to find barlines
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(barline_mask)

    max_barline_height = 0
    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]
        max_barline_height = max(max_barline_height, h)

    # Threshold for long lines in the RGB image
    min_required_height = max(2 * max_barline_height, 100)

    # Convert RGB to grayscale, then to binary black pixel mask
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_pixels = (gray_img == 0).astype(np.uint8) * 255

    # Morphological operation to connect vertical structures
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_required_height))
    connected = cv2.morphologyEx(black_pixels, cv2.MORPH_OPEN, vertical_kernel)

    # Find connected components again to isolate tall vertical lines
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(connected)

    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]
        if h >= min_required_height:
            result[labels == i] = 1

    mask[(result == 1) & (mask == 0)] = DEF.BRACKETS[0]

    return mask


if __name__ == "__main__":
    from pathlib import Path

    script_location = os.path.dirname(os.path.realpath(__file__))
    git_root = Path(script_location).parent.parent.absolute()
    dataset_root = os.path.join(git_root, "datasets")
    seg_folder = os.path.join(dataset_root, "ds2_dense", "segmentation")
    color = int(sys.argv[1])
    with_background = find_example(seg_folder, color)
    if with_background is None:
        eprint("Found no examples")
    else:
        cv2.imwrite("example.png", with_background)
