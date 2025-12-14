import numpy as np
from PIL import Image

from homr.type_definitions import NDArray


def calc_target_image_size(width: int, height: int) -> tuple[int, int]:
    """
    Target a fixed width while preserving aspect ratio.
    """
    target_width = 1920

    if width == target_width:
        return width, height

    ratio = target_width / width
    tar_w = target_width
    tar_h = round(height * ratio)

    return tar_w, tar_h


def resize_image(image_arr: NDArray) -> NDArray:
    image = Image.fromarray(image_arr)
    tar_w, tar_h = calc_target_image_size(image.size[0], image.size[1])
    if tar_w == image_arr.shape[1] and tar_h == image_arr.shape[0]:
        return image_arr
    return np.array(image.resize((tar_w, tar_h)))
