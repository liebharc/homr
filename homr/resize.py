import numpy as np
from PIL import Image

from homr.simple_logging import eprint
from homr.type_definitions import NDArray


def calc_target_image_size(image: Image.Image) -> tuple[int, int]:
    # Estimate target size with number of pixels.
    # Best number would be 3M~4.35M pixels.
    w, h = image.size
    pixels = w * h
    target_size_min = 3.0 * 1000 * 1000
    target_size_max = 4.35 * 1000 * 1000
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
        eprint("Keeping original size of", tar_w, "x", tar_h)
        return image_arr

    eprint(
        "Resizing input from", image_arr.shape[1], "x", image_arr.shape[0], "to", tar_w, "x", tar_h
    )
    return np.array(image.resize((tar_w, tar_h)))
