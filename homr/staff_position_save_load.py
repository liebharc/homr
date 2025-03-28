import numpy as np

from homr import constants
from homr.bounding_boxes import BoundingBox
from homr.model import MultiStaff, Staff, StaffPoint


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


def load_staff_positions(shape: tuple[int, ...], file_path: str) -> list[MultiStaff]:
    staffs = []
    img_height, img_width, _ = shape

    with open(file_path) as text_file:
        lines = text_file.readlines()

    for line in lines:
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
        staff = dummy_staff_from_rect(bounding_box, shape)
        staffs.append(MultiStaff([staff], []))

    return staffs


def dummy_staff_from_rect(box: BoundingBox, shape: tuple[int, ...]) -> Staff:
    x1, y1, x2, y2 = box.box
    x1 = max(0, x1 - 50)
    x2 = min(shape[0], x2 + 50)
    points = []
    height = y2 - y1
    for x in range(int(x1), int(x2), 10):
        yValues = [(i * height / 4) + y1 for i in range(5)]
        points.append(StaffPoint(x, yValues, 0))
    return Staff(points)
