import cv2
import numpy as np

from homr import constants
from homr.bounding_boxes import RotatedBoundingBox
from homr.type_definitions import NDArray


def prepare_bar_line_image(image: NDArray) -> NDArray:
    kernel = np.ones((5, 3), np.uint8)
    result = cv2.dilate(image, kernel, iterations=1)
    return result


def detect_bar_lines(
    bar_lines: list[RotatedBoundingBox], unit_size: float
) -> list[RotatedBoundingBox]:
    """
    Filters the bar line candidates based on their size.
    """
    result = []
    for bar_line in bar_lines:
        if bar_line.size[1] < constants.bar_line_min_height(unit_size):
            continue
        if bar_line.size[0] > constants.bar_line_max_width(unit_size):
            continue
        result.append(bar_line)
    return result
