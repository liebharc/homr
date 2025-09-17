import unittest

import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.staff_detection import connect_staff_lines


def makeBoundingBox(x: float, y: float) -> RotatedBoundingBox:
    w, h = 40.0, 2.0
    angle = 0.0
    return RotatedBoundingBox(((x, y), (w, h), angle), np.array([]))


class TestStaffDetection(unittest.TestCase):

    def test_connect_staff_lines(self) -> None:
        lines = [makeBoundingBox(100, 100), makeBoundingBox(50, 100), makeBoundingBox(150, 100)]
        result = connect_staff_lines(lines, 5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].staff_fragments, [lines[1], lines[0], lines[2]])
