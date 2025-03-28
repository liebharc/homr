import unittest

import numpy as np

from homr.bounding_boxes import BoundingBox, BoundingEllipse, RotatedBoundingBox

empty = np.array([])


class TestBoundingBoxes(unittest.TestCase):

    unit_size = 3

    def test_is_overlapping_rotated_box_extrapolated(self) -> None:
        box1 = RotatedBoundingBox(((100, 200), (10, 10), 0), empty)
        touching_box = RotatedBoundingBox(((110, 200), (10, 10), 0), empty)
        samle_line_box = RotatedBoundingBox(((140, 200), (10, 10), 0), empty)
        different_line_box = RotatedBoundingBox(((220, 300), (10, 10), 0), empty)
        self.assertTrue(box1.is_overlapping_extrapolated(touching_box, self.unit_size))
        self.assertTrue(box1.is_overlapping_extrapolated(samle_line_box, self.unit_size))
        self.assertFalse(box1.is_overlapping_extrapolated(different_line_box, self.unit_size))

        box2 = RotatedBoundingBox(((100, 200), (10, 10), 45), empty)
        touching_box = RotatedBoundingBox(((110, 200), (10, 20), 50), empty)
        samle_line_box = RotatedBoundingBox(((110, 200), (10, 10), 45), empty)
        different_line_box = RotatedBoundingBox(((140, 220), (10, 10), 0), empty)
        self.assertTrue(box2.is_overlapping_extrapolated(touching_box, self.unit_size))
        self.assertFalse(box2.is_overlapping_extrapolated(samle_line_box, self.unit_size))
        self.assertTrue(box2.is_overlapping_extrapolated(different_line_box, self.unit_size))

    def test_is_overlapping_rotated_box_with_rotated_box(self) -> None:
        box1 = RotatedBoundingBox(((100, 200), (10, 10), 0), empty)
        touching_box = RotatedBoundingBox(((110, 200), (10, 10), 0), empty)
        inside_box = RotatedBoundingBox(((105, 200), (10, 10), 0), empty)
        far_away_box = RotatedBoundingBox(((200, 200), (10, 10), 0), empty)
        crossing_box = RotatedBoundingBox(((105, 205), (10, 10), 90), empty)

        self.assertTrue(box1.is_overlapping(touching_box))
        self.assertTrue(box1.is_overlapping(inside_box))
        self.assertFalse(box1.is_overlapping(far_away_box))
        self.assertTrue(box1.is_overlapping(crossing_box))

    def test_is_overlapping_rotated_box_with_ellipse(self) -> None:
        box1 = RotatedBoundingBox(((100, 200), (10, 10), 0), empty)
        touching_box = BoundingEllipse(((110, 200), (10, 10), 0), empty)
        inside_box = BoundingEllipse(((105, 200), (10, 10), 0), empty)
        far_away_box = BoundingEllipse(((200, 200), (10, 10), 0), empty)

        self.assertTrue(box1.is_overlapping(touching_box))
        self.assertTrue(box1.is_overlapping(inside_box))
        self.assertFalse(box1.is_overlapping(far_away_box))

        box2 = RotatedBoundingBox(
            ((570.1167602539062, 506.98968505859375), (2, 60), -5.042449951171875), empty
        )
        ellipse2 = BoundingEllipse(((536.93896484375, 470.5845947265625), (13, 17), 5), empty)
        self.assertFalse(box2.is_overlapping(ellipse2))

    def test_partial_overlap(self) -> None:
        box1 = BoundingBox((10, 10, 50, 50), contours=empty)
        box2 = BoundingBox((30, 30, 70, 70), contours=empty)
        overlap_area = box1.get_overlapping_area_size(box2)
        expected_area = (50 - 30) * (50 - 30)
        self.assertEqual(overlap_area, expected_area)

    def test_no_overlap(self) -> None:
        box1 = BoundingBox((10, 10, 50, 50), contours=empty)
        box3 = BoundingBox((100, 100, 150, 150), contours=empty)
        overlap_area = box1.get_overlapping_area_size(box3)
        self.assertEqual(overlap_area, 0)

    def test_full_overlap(self) -> None:
        box1 = BoundingBox((10, 10, 50, 50), contours=empty)
        box4 = BoundingBox((10, 10, 50, 50), contours=empty)
        overlap_area = box1.get_overlapping_area_size(box4)
        expected_area = (50 - 10) * (50 - 10)
        self.assertEqual(overlap_area, expected_area)
