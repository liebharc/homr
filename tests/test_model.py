import unittest

import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import MultiStaff, Staff, StaffPoint


def make_staff(number: int) -> Staff:
    y_points = [10 * i + 100 * float(number) for i in range(5)]
    return Staff([StaffPoint(0.0, y_points, 0)])


def make_connection(number: int) -> RotatedBoundingBox:
    rect = ((float(number), float(number)), (number, number), float(number))
    contours = np.empty((0, 0))
    return RotatedBoundingBox(rect, contours)


class TestModel(unittest.TestCase):

    def test_multi_staff_merge(self) -> None:
        staff1 = MultiStaff(
            [make_staff(1), make_staff(2)], [make_connection(1), make_connection(2)]
        )
        staff2 = MultiStaff(
            [staff1.staffs[1], make_staff(3)], [staff1.connections[1], make_connection(3)]
        )
        result1 = staff1.merge(staff2)
        self.assertEqual(result1.staffs, [staff1.staffs[0], staff1.staffs[1], staff2.staffs[1]])
        self.assertEqual(
            result1.connections,
            [staff1.connections[0], staff1.connections[1], staff2.connections[1]],
        )

        result2 = staff2.merge(staff1)
        self.assertEqual(result2.staffs, [staff1.staffs[0], staff2.staffs[0], staff2.staffs[1]])
        self.assertEqual(
            result2.connections,
            [staff2.connections[0], staff2.connections[1], staff1.connections[0]],
        )
