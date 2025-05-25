import sys
import traceback

import cv2
import numpy as np

from homr import constants
from homr.bounding_boxes import BoundingBox, create_lines
from homr.debug import Debug
from homr.model import MultiStaff, Staff, StaffPoint
from homr.simple_logging import eprint
from homr.staff_detection import detect_staff
from homr.type_definitions import NDArray


def save_staff_positions(
    multi_staffs: list[MultiStaff], shape: tuple[int, ...], file_path: str
) -> None:
    staff_coordinates = []
    for multi_staff in multi_staffs:
        for staff in multi_staff.staffs:
            x1, y1, x2, y2 = staff.min_x, staff.min_y, staff.max_x, staff.max_y
            width = x2 - x1
            height = y2 - y1
            centerx = x1 + width / 2
            centery = y1 + height / 2
            img_height, img_width, _ = shape
            coordinate = (
                "0 "
                + str(centerx / img_width)
                + " "
                + str(centery / img_height)
                + " "
                + str(width / img_width)
                + " "
                + str(height / img_height)
            )
            staff_coordinates.append(coordinate + "\n")
    with open(file_path, "w") as text_file:
        text_file.writelines(staff_coordinates)


def load_staff_positions(
    debug: Debug, image: NDArray, file_path: str, selected_staff: int = -1
) -> list[MultiStaff]:
    staffs = []
    img_height, img_width, _ = image.shape

    with open(file_path) as text_file:
        lines = text_file.readlines()

    for line_index, line in enumerate(lines):
        try:
            parts = line.strip().split()
            if len(parts) != constants.number_of_lines_on_a_staff:
                continue  # Ignore invalid lines

            _, norm_centerx, norm_centery, norm_width, norm_height = map(float, parts)

            width = norm_width * img_width
            height = norm_height * img_height
            centerx = norm_centerx * img_width
            centery = norm_centery * img_height

            x1 = int(centerx - width / 2)
            y1 = int(centery - height / 2)
            x2 = int(x1 + width)
            y2 = int(y1 + height)

            bounding_box = BoundingBox([x1, y1, x2, y2], np.array([]))
            crop_area = bounding_box.increase_size_in_each_dimension(100, image.shape)
            staff_img = crop_area.blank_everything_outside_of_box(image)
            if selected_staff >= 0 and line_index != selected_staff:
                staff = dummy_staff_from_rect(bounding_box, image.shape)
                if staff is not None:
                    staffs.append(MultiStaff([staff], []))
                continue

            staff = detect_staff_simple(debug, staff_img, crop_area)
            if staff is None:
                staff = dummy_staff_from_rect(bounding_box, image.shape)
            min_staff_width = 10
            if staff is not None and staff.max_x - staff.min_x > min_staff_width:
                staffs.append(MultiStaff([staff], []))
        except Exception as e:
            eprint("Skipping staff due to error:", e)
            traceback.print_exc(file=sys.stderr)

    return staffs


def detect_staff_simple(debug: Debug, img: NDArray, crop_area: BoundingBox) -> Staff | None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )

    # Create horizontal structuring element and apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=2)
    horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_DILATE, kernel, iterations=2)
    horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_CLOSE, kernel, iterations=2)
    debug.write_threshold_image("horizontal_lines", 255 * horizontal_lines)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_DILATE, kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_CLOSE, kernel, iterations=2)
    debug.write_threshold_image("vertical_lines", 255 * vertical_lines)

    # Detect lines using Hough Transform
    staff_lines = create_lines(horizontal_lines, threshold=30, min_line_length=30)
    bar_lines = create_lines(vertical_lines, threshold=20, min_line_length=20, max_line_gap=15)
    staff_lines = [line.ensure_min_dimension(3, 3) for line in staff_lines]
    bar_lines = [line.ensure_min_dimension(3, 3) for line in bar_lines]
    bar_lines = [bar.make_box_taller_keep_center(5) for bar in bar_lines]

    crop_box = crop_area.box
    min_x = max(crop_box[0] + 100, 0)
    max_x = crop_box[3] - 100

    staffs = detect_staff(debug, horizontal_lines, staff_lines, [], bar_lines)
    if len(staffs) > 0:
        return staffs[0].extend_to_x_range(min_x, max_x)
    return None


def dummy_staff_from_rect(box: BoundingBox, shape: tuple[int, ...]) -> Staff | None:
    x1, y1, x2, y2 = box.box
    x1 = max(0, x1 - 50)
    x2 = min(shape[0], x2 + 50)
    points = []
    height = y2 - y1
    for x in range(int(x1), int(x2), 10):
        yValues = [(i * height / 4) + y1 for i in range(5)]
        points.append(StaffPoint(x, yValues, 0))
    if len(points) == 0:
        return None
    return Staff(points)


if __name__ == "__main__":
    import sys

    img = cv2.imread(sys.argv[1])
    staff = detect_staff_simple(
        Debug(img, "input.jpg", True), img, BoundingBox([100, 100, 500, 500], np.array([]))
    )

    if staff is None:
        print("No staff lines detected.")  # noqa: T201
    else:
        print(staff)  # noqa: T201
