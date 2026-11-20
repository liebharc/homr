# ruff: noqa: S101

import unittest
from unittest import mock

import cv2
import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import BarLine, Staff, StaffPoint
from homr.number_recognition import (
    CLASSES,
    OTHER,
    Component,
    _full_measure_rest,
    _measures,
    _merge_overlapping,
    measure_intervals,
    read_numbers,
    staff_lines,
)
from homr.transformer.vocabulary import EncodedSymbol


def _staff_image(barlines: list[int]) -> np.ndarray:
    """A staff with unit size 12, the top line at 100 and the given barlines."""
    image = np.full((256, 1000), 255, np.uint8)
    for i in range(5):
        cv2.line(image, (40, 100 + 12 * i), (960, 100 + 12 * i), 0, 1)
    for x in barlines:
        cv2.line(image, (x, 100), (x, 148), 0, 2)
    return image


def _staff(barlines: list[int], tops: tuple[int, ...] = (100,)) -> Staff:
    """The staff of _staff_image, with the given barlines."""
    grid = [
        StaffPoint(x, [top + 12.0 * i for top in tops for i in range(5)], 0)
        for x in range(40, 961, 10)
    ]
    staff = Staff(grid)
    for x in barlines:
        box = RotatedBoundingBox(((x, 124), (2, 48), 0), np.zeros((0, 1, 2), np.int32))
        staff.add_symbol(BarLine(box))
    return staff


def _symbols(rhythms: str) -> list[EncodedSymbol]:
    return [EncodedSymbol(r) for r in rhythms.split()]


class TestNumberRecognition(unittest.TestCase):
    def test_staff_lines(self) -> None:
        lines = staff_lines(_staff([]), 256)
        assert lines is not None
        self.assertEqual((lines.top, lines.unit), (100, 12))

    def test_staff_lines_of_a_grand_staff_are_closest_to_the_center(self) -> None:
        lines = staff_lines(_staff([], tops=(20, 100)), 256)
        assert lines is not None
        self.assertEqual(lines.top, 100)

    def test_measure_intervals(self) -> None:
        image = _staff_image([300, 600, 959])
        lines = staff_lines(_staff([]), image.shape[0])
        assert lines is not None
        intervals, matches = measure_intervals(
            image, lines, [600, 300, 959], 3, starts_with_barline=False
        )
        self.assertTrue(matches)
        self.assertEqual([round(a / 10) for a, _ in intervals], [4, 30, 60])

    def test_measures_ignore_barlines_before_the_first_note(self) -> None:
        symbols = _symbols("clef_G2 repeatStart rest_2 barline note_4 note_4 barline rest_2")
        self.assertEqual(_measures(symbols), [[0, 2], [4, 5], [7]])

    def test_full_measure_rest(self) -> None:
        symbols = _symbols("rest_2 barline rest_8 barline note_4 chord note_4")
        measures = _measures(symbols)
        self.assertEqual(_full_measure_rest(symbols, measures[0], None), 0)
        # A short rest, e.g. the part of a measure before a repeat sign
        self.assertIsNone(_full_measure_rest(symbols, measures[1], None))
        self.assertIsNone(_full_measure_rest(symbols, measures[2], None))

    def test_merge_overlapping_joins_parts_of_a_digit(self) -> None:
        top_bar = Component(10, 10, 12, 3)
        body = Component(10, 14, 12, 12)
        neighbor = Component(25, 10, 10, 16)
        merged = _merge_overlapping([top_bar, body, neighbor], max_gap=2)
        self.assertEqual(merged, [Component(10, 10, 12, 16), neighbor])

    def test_multirest_without_number_lasts_one_measure(self) -> None:
        symbols = _symbols("clef_G2 rest_4m barline note_4 note_4 barline")
        no_digits = mock.Mock()
        no_digits.predict.side_effect = lambda crops: np.eye(len(CLASSES))[[OTHER] * len(crops)]
        with mock.patch("homr.number_recognition._get_classifier", return_value=no_digits):
            result = read_numbers(_staff_image([500, 959]), symbols, _staff([500, 959]))
        self.assertEqual(result[1].rhythm, "measureRest")

    def test_rests_pointing_to_the_same_measure_get_no_number(self) -> None:
        # Two barline intervals for three measures: the rests are placed by attention
        image = _staff_image([500, 959])
        cv2.rectangle(image, (265, 60), (277, 82), 0, -1)  # a "number" above the first measure
        fives = mock.Mock()
        fives.predict.side_effect = lambda crops: np.eye(len(CLASSES))[[5] * len(crops)]

        def read(attention_x: list[float]) -> list[str]:
            symbols = _symbols("clef_G2 rest_1 barline rest_1 barline note_4 note_4 barline")
            for index, x in zip([1, 3], attention_x, strict=True):
                symbols[index].coordinates = (x, 120.0)
            with mock.patch("homr.number_recognition._get_classifier", return_value=fives):
                return [s.rhythm for s in read_numbers(image, symbols, _staff([500, 959]))]

        self.assertEqual(read([200, 700])[1], "rest_5m")
        self.assertEqual(read([200, 300])[1:4:2], ["rest_1", "rest_1"])
