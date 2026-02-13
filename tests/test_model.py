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

    def test_create_grandstaffs_even(self) -> None:
        staffs = [make_staff(i) for i in range(4)]
        connections = [make_connection(i) for i in range(4)]
        ms = MultiStaff(staffs, connections)

        result = ms.create_grandstaffs()

        self.assertEqual(len(result.staffs), 2)
        self.assertEqual(result.staffs[0].min_y, staffs[0].min_y)
        self.assertEqual(result.staffs[0].max_y, staffs[1].max_y)
        self.assertEqual(result.staffs[1].min_y, staffs[2].min_y)
        self.assertEqual(result.staffs[1].max_y, staffs[3].max_y)
        self.assertEqual(result.connections, connections)

    def test_create_grandstaffs_odd(self) -> None:
        staffs = [make_staff(i) for i in range(5)]
        connections = [make_connection(i) for i in range(5)]
        ms = MultiStaff(staffs, connections)

        result = ms.create_grandstaffs()

        self.assertEqual(len(result.staffs), 3)
        self.assertEqual(result.staffs[0].min_y, staffs[0].min_y)
        self.assertEqual(result.staffs[0].max_y, staffs[0].max_y)
        self.assertEqual(result.staffs[1].min_y, staffs[1].min_y)
        self.assertEqual(result.staffs[1].max_y, staffs[2].max_y)
        self.assertEqual(result.staffs[2].min_y, staffs[3].min_y)
        self.assertEqual(result.staffs[2].max_y, staffs[4].max_y)
        self.assertEqual(result.connections, connections)

    def test_create_grandstaffs_empty(self) -> None:
        ms = MultiStaff([], [])

        result = ms.create_grandstaffs()

        self.assertIs(result, ms)
