import numpy as np
from PIL import Image

from .types import NDArray


def calc_target_image_size(image: Image.Image) -> tuple[int, int]:
    # Estimate target size with number of pixels.
    # Best number would be 3M~4.35M pixels.
    w, h = image.size
    pixels = w * h
    target_size_min = 2.5 * 1024 * 1024
    target_size_max = 3.5 * 1024 * 1024
    if target_size_min <= pixels <= target_size_max:
        return w, h
    lb = target_size_min / pixels
    ub = target_size_max / pixels
    ratio = pow((lb + ub) / 2, 0.5)
    tar_w = round(ratio * w)
    tar_h = round(ratio * h)
    return tar_w, tar_h


def resize_image(image_arr: NDArray) -> NDArray:
    image = Image.fromarray(image_arr)
    tar_w, tar_h = calc_target_image_size(image)
    if tar_w == image_arr.shape[1] and tar_h == image_arr.shape[0]:
        return image_arr

    return np.array(image.resize((tar_w, tar_h)))
