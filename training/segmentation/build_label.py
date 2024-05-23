import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

from homr.simple_logging import eprint
from homr.type_definitions import NDArray
from training.segmentation.constant_min import CHANNEL_NUM, CLASS_CHANNEL_MAP
from training.segmentation.dense_dataset_definitions import (
    DENSE_DATASET_DEFINITIONS as DEF,
)

HALF_WHOLE_NOTE = DEF.NOTEHEADS_HOLLOW + DEF.NOTEHEADS_WHOLE + [42]


# ruff: noqa: C901, PLR0912
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


def build_label(
    seg_path: str, strenghten_channels: dict[int, tuple[int, int]] | None = None
) -> NDArray:
    img = Image.open(seg_path)
    arr = np.array(img)
    color_set = set(np.unique(arr))
    color_set.remove(0)  # Remove background color from the candidates

    total_chs = CHANNEL_NUM
    output = np.zeros(arr.shape + (total_chs,))

    output[..., 0] = np.where(arr == 0, 1, 0)
    for color in color_set:
        ch = CLASS_CHANNEL_MAP.get(color, 0)
        if (ch != 0) and color in HALF_WHOLE_NOTE:
            note = fill_hole(arr, color)
            output[..., ch] += note
        elif ch != 0:
            output[..., ch] += np.where(arr == color, 1, 0)
    if strenghten_channels is not None:
        for ch in strenghten_channels.keys():
            output[..., ch] = make_symbols_stronger(output[..., ch], strenghten_channels[ch])
        # The background channel is 1 if all other channels are 0
        background_ch = np.ones((arr.shape[0], arr.shape[1]))
        for ch in range(1, total_chs):
            background_ch = np.where(output[..., ch] == 1, 0, background_ch)
        output[..., 0] = background_ch
    return output


def close_lines(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    # Use hough transform to find lines
    width = img.shape[1]
    lines = cv2.HoughLinesP(
        img, 1, np.pi / 180, threshold=width // 32, minLineLength=width // 16, maxLineGap=50
    )
    if lines is not None:
        angles = []
        # Draw lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)
        mean_angle = np.mean(angles)
        # Draw lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            is_horizontal = abs(angle - mean_angle) < np.pi / 16
            if is_horizontal:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)  # type: ignore
    else:
        eprint("No lines found")

    return img


def make_symbols_stronger(img: NDArray, kernel_size: tuple[int, int] = (5, 5)) -> NDArray:
    """
    Dilates the symbols to make them stronger
    """
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def find_example(
    dataset_path: str, color: int, max_count: int = 100, mark_value: int = 200
) -> NDArray | None:
    files = os.listdir(dataset_path)
    random.shuffle(files)
    for ff in files[:max_count]:
        path = os.path.join(dataset_path, ff)
        img = Image.open(path)
        arr = np.array(img)
        if color in arr:
            return np.where(arr == color, mark_value, arr)

    return None


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
        cv2.imwrite("example.png", 255 * with_background)
