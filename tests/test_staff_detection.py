import unittest

import cv2
import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import Staff, StaffPoint
from homr.staff_detection import (
    RawStaff,
    StaffAnchor,
    StaffLineSegment,
    connect_staff_lines,
    filter_neighbor_page,
    find_horizontal_lines,
    merge_staffs_in_same_row,
)


def makeBoundingBox(x: float, y: float) -> RotatedBoundingBox:
    w, h = 40.0, 2.0
    angle = 0.0
    return RotatedBoundingBox(((x, y), (w, h), angle), np.array([]))


def make_staff(min_x: float, max_x: float, top: float) -> Staff:
    y = [top + 10 * i for i in range(5)]
    return Staff([StaffPoint(min_x, y, 0), StaffPoint(max_x, y, 0)])


def make_raw_staff(min_x: float, max_x: float, top: float) -> RawStaff:
    """A staff with a unit size of 10 and one fragment per line."""
    lines = []
    for i in range(5):
        box = ((float(min_x + max_x) / 2, top + 10 * i), (float(max_x - min_x), 2.0), 0.0)
        contour = cv2.boxPoints(box).astype(np.int32).reshape(-1, 1, 2)
        lines.append(StaffLineSegment(i, [RotatedBoundingBox(box, contour)]))
    clef = RotatedBoundingBox(((min_x + 10, top + 20), (20, 40), 0), np.array([]))
    return RawStaff(0, lines, [StaffAnchor(lines, clef)])


class TestStaffDetection(unittest.TestCase):
    def test_merge_staffs_in_same_row(self) -> None:
        left = make_raw_staff(100, 500, 100)
        after_clef_change = make_raw_staff(600, 1500, 100)
        next_row = make_raw_staff(100, 500, 300)
        shifted_by_a_line = make_raw_staff(600, 1500, 310)
        neighbor_page = make_raw_staff(1800, 2500, 100)

        result = merge_staffs_in_same_row(
            [left, after_clef_change, next_row, shifted_by_a_line, neighbor_page]
        )

        self.assertEqual(4, len(result))
        merged = next(staff for staff in result if staff.min_y < 200 and staff.max_x < 1600)
        self.assertAlmostEqual(100, merged.min_x, delta=2)
        self.assertAlmostEqual(1500, merged.max_x, delta=2)

    def test_filter_neighbor_page(self) -> None:
        page = [make_staff(500, 1500, top) for top in (100, 300, 500)]
        indented = make_staff(700, 1500, 700)
        neighbor_page = make_staff(100, 350, 300)

        result = filter_neighbor_page([*page, indented, neighbor_page])

        self.assertEqual([*page, indented], result)

    def test_connect_staff_lines(self) -> None:
        lines = [makeBoundingBox(100, 100), makeBoundingBox(50, 100), makeBoundingBox(150, 100)]
        result = connect_staff_lines(lines, 5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].staff_fragments, [lines[1], lines[0], lines[2]])

    def test_find_horizontal_lines_without_lines(self) -> None:
        empty = np.zeros((200, 20), dtype=np.uint8)
        self.assertEqual([], find_horizontal_lines(empty, 10))
