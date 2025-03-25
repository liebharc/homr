import cv2
import numpy as np
from scipy.signal import medfilt  # type: ignore

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import Staff, StaffPoint
from homr.type_definitions import NDArray


def preprocess_image(staff_image: NDArray) -> NDArray:
    """Preprocess the image: apply Gaussian blur and binarization."""
    blurred = cv2.GaussianBlur(staff_image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def get_staff_lines(binary_image: NDArray) -> list[int]:
    """Find staff lines using a vertical projection profile."""
    projection = np.sum(binary_image, axis=1)
    peaks = np.argpartition(projection, -5)[-5:]  # Get 5 highest peaks
    return sorted(peaks)


def refine_lines(binary_image: NDArray) -> list[list[int]]:
    """Refine staff lines by tracking them across the width of the image."""
    height, width = binary_image.shape
    tracked_lines: list[list[int]] = [[] for _ in range(5)]
    debug_image = np.zeros_like(binary_image)

    for x in range(width):
        column = binary_image[:, x]
        detected = get_staff_lines(column.reshape(-1, 1))
        debug_image[detected, x] = 255
        lines_in_a_staff = 5
        if len(detected) == lines_in_a_staff:
            for i, y in enumerate(detected):
                tracked_lines[i].append(y)

    return [medfilt(line, kernel_size=5) for line in tracked_lines]


def construct_staff_from_lines(staff_image: NDArray) -> Staff:
    binary_image = preprocess_image(staff_image)
    refined_lines = refine_lines(staff_image)
    height, width = binary_image.shape

    staff_grid = []
    for x in range(width):
        y_positions = [float(refined_lines[i][x]) for i in range(5)]
        angle = 0.0  # Can be estimated if needed
        staff_grid.append(StaffPoint(float(x), y_positions, angle))

    return Staff(staff_grid)


def construct_staff_from_lines_in_area(
    staff_image: NDArray, area: RotatedBoundingBox
) -> Staff | None:
    x1, y1, x2, y2 = area.to_bounding_box().box
    cropped_image = staff_image[y1:y2, x1:x2]
    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        return None

    staff = construct_staff_from_lines(cropped_image)

    staff_grid = []
    for staff_point in staff.grid:
        staff_grid.append(
            StaffPoint(staff_point.x + x1, [y + y1 for y in staff_point.y], staff_point.angle)
        )

    return Staff(staff_grid)
