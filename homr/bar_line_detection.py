import cv2
import numpy as np

from homr import constants
from homr.bounding_boxes import RotatedBoundingBox
from homr.model import BarLine, Staff
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


def add_bar_lines_to_staffs(staffs: list[Staff], bar_lines: list[RotatedBoundingBox]) -> None:
    """
    Adds the bar lines which reach from the top to the bottom line of a staff, centered
    on its middle line. A bar line through a whole system is added to each staff.
    """
    for bar_line in bar_lines:
        corners = cv2.boxPoints(bar_line.box)
        x = float(corners[:, 0].mean())
        top, bottom = float(corners[:, 1].min()), float(corners[:, 1].max())
        for staff in staffs:
            point = staff.get_at(x)
            if point is None:
                continue
            unit = (point.y[-1] - point.y[0]) / (len(point.y) - 1)
            if top <= point.y[0] + 0.5 * unit and bottom >= point.y[-1] - 0.5 * unit:
                middle = (point.y[0] + point.y[-1]) / 2
                staff.add_symbol(BarLine(bar_line, (x, middle)))
