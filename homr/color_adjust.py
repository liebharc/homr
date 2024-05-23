import math

import cv2
import numpy as np
import scipy  # type: ignore

from homr.simple_logging import eprint
from homr.type_definitions import NDArray


def get_dominant_color(
    gray_scale: NDArray, color_range: range, default: int | None = None
) -> int | None:
    if gray_scale.dtype != np.uint8:
        raise Exception("Wrong image dtype")

    # Create a mask for values in the range [min_val, max_val]
    mask = (gray_scale >= color_range.start) & (gray_scale <= color_range.stop)

    # Apply the mask to the grayscale image
    masked_gray_scale = gray_scale[mask]
    if masked_gray_scale.size == 0:
        return default

    bins = np.bincount(masked_gray_scale.flatten())
    center_of_mass = scipy.ndimage.measurements.center_of_mass(bins)[0]

    return int(center_of_mass)


def apply_clahe(channel: NDArray) -> NDArray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(channel)


def remove_background_from_channel(channel: NDArray, block_size: int) -> tuple[NDArray, NDArray]:
    """
    Divides the image into blocks of size block_size and calculates
    the dominant color of each block. The dominant color is then
    used to create a background image, which is then used to divide the
    original image. The result is an image with a more uniform background.
    """
    x_range = range(0, channel.shape[0], block_size)
    y_range = range(0, channel.shape[1], block_size)
    background_pixels = np.zeros(
        [math.ceil(x_range.stop / block_size), math.ceil(y_range.stop / block_size)], dtype=np.uint8
    )
    color_range = range(150, 254)
    background = get_dominant_color(channel, color_range)
    for i, row in enumerate(x_range):
        for j, col in enumerate(y_range):
            idx = (row, col)
            block_idx = get_block_index(channel.shape, idx, block_size)
            background_pixels[i, j] = get_dominant_color(
                channel[block_idx], color_range, background
            )
    background_blurred = cv2.blur(background_pixels, (3, 3))
    color_white = 255
    valid_background = background_blurred < color_white  # type: ignore
    max_background = int(np.max(background_blurred[valid_background]))
    background_blurred[valid_background] += color_white - max_background
    result_background = cv2.resize(
        background_blurred, (channel.shape[1], channel.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    division = cv2.divide(channel, result_background, scale=color_white)

    return division, result_background


def get_block_index(
    image_shape: tuple[int, ...], yx: tuple[int, int], block_size: int
) -> tuple[NDArray, ...]:
    """
    Creates a grid of indices for a block of pixels around a given pixel.
    """
    y = np.arange(max(0, yx[0] - block_size), min(image_shape[0], yx[0] + block_size))
    x = np.arange(max(0, yx[1] - block_size), min(image_shape[1], yx[1] + block_size))
    return np.ix_(y, x)


def color_adjust(image: NDArray, block_size: int) -> tuple[NDArray, NDArray]:
    """
    Reduce the effect of uneven lighting on the image by dividing the image by its interpolated
    background.
    """
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image, background = remove_background_from_channel(image, block_size)
        return cv2.cvtColor(apply_clahe(image), cv2.COLOR_GRAY2BGR), background
    except Exception as e:
        eprint(e)
        return image, image
