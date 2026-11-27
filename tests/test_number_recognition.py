# ruff: noqa: S101

import unittest
from unittest import mock

import cv2
import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import BarLine, Staff, StaffPoint
from homr.number_recognition import (
    CROP_HEIGHT,
    CROP_WIDTH,
    DIGIT_CLASSES,
    Reading,
    _full_measure_rest,
    _measures,
    crop_at,
    decode_digits,
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


def _reader(reading: Reading) -> mock.Mock:
    """A reader which reads the same for every crop."""
    reader = mock.Mock()
    reader.read.side_effect = lambda crops: [reading] * len(crops)
    return reader


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

    def test_misread_multirest_bar_is_a_full_measure_rest(self) -> None:
        symbols = _symbols("rest_8 chord rest_2 barline rest_2 note_4")
        measures = _measures(symbols)
        self.assertEqual(_full_measure_rest(symbols, measures[0], None), 2)
        self.assertIsNone(_full_measure_rest(symbols, measures[1], None))

    def test_numbered_rest_drops_the_other_rests_of_its_measure(self) -> None:
        symbols = _symbols("clef_G2 rest_2 chord rest_8 barline note_4 note_4 barline")
        sevens = _reader(Reading(None, None, 7, 0.9, 0.9))
        with mock.patch("homr.number_recognition._get_reader", return_value=sevens):
            result = read_numbers(_staff_image([500, 959]), symbols, _staff([500, 959]))
        self.assertEqual([s.rhythm for s in result][:3], ["clef_G2", "rest_7m", "barline"])

    def test_decode_digits(self) -> None:
        def slots(*classes: int) -> np.ndarray:
            return np.eye(DIGIT_CLASSES)[list(classes)]

        self.assertEqual(decode_digits(slots(0, 4))[0], 3)
        self.assertEqual(decode_digits(slots(2, 3))[0], 12)
        self.assertEqual(decode_digits(slots(0, 2, 1))[0], 10)
        self.assertIsNone(decode_digits(slots(0, 0))[0])
        # A gap or a leading zero is no number
        self.assertIsNone(decode_digits(slots(2, 0, 3))[0])
        self.assertIsNone(decode_digits(slots(1, 4))[0])

    def test_crop_at_puts_the_staff_at_a_fixed_place(self) -> None:
        image = _staff_image([])
        lines = staff_lines(_staff([]), image.shape[0])
        assert lines is not None
        crop = crop_at(image, 500, lines)
        self.assertEqual(crop.shape, (CROP_HEIGHT, CROP_WIDTH))
        rows = np.where((crop < 128).mean(axis=1) > 0.9)[0]  # noqa: PLR2004
        # The top line is 5 units, the bottom line 9 units below the top of the crop
        self.assertEqual([rows.min(), rows.max()], [40, 72])

    def test_time_signature_gets_numerator(self) -> None:
        symbols = _symbols("clef_G2 timeSignature/4 note_4 barline")
        symbols[1].coordinates = (100.0, 124.0)
        reader = _reader(Reading(3, 4, None, 0.9, 0.9))
        with mock.patch("homr.number_recognition._get_reader", return_value=reader):
            result = read_numbers(_staff_image([959]), symbols, _staff([959]))
        self.assertEqual(result[1].rhythm, "timeSignature3/4")

    def test_multirest_without_number_lasts_one_measure(self) -> None:
        symbols = _symbols("clef_G2 rest_4m barline note_4 note_4 barline")
        no_digits = _reader(Reading(None, None, None, 0.9, 0.9))
        with mock.patch("homr.number_recognition._get_reader", return_value=no_digits):
            result = read_numbers(_staff_image([500, 959]), symbols, _staff([500, 959]))
        self.assertEqual(result[1].rhythm, "measureRest")

    def test_rests_pointing_to_the_same_measure_get_no_number(self) -> None:
        # Two barline intervals for three measures: the rests are placed by attention
        image = _staff_image([500, 959])
        fives = _reader(Reading(None, None, 5, 0.9, 0.9))

        def read(attention_x: list[float]) -> list[str]:
            symbols = _symbols("clef_G2 rest_1 barline rest_1 barline note_4 note_4 barline")
            for index, x in zip([1, 3], attention_x, strict=True):
                symbols[index].coordinates = (x, 120.0)
            with mock.patch("homr.number_recognition._get_reader", return_value=fives):
                return [s.rhythm for s in read_numbers(image, symbols, _staff([500, 959]))]

        self.assertEqual(read([200, 700])[1], "rest_5m")
        self.assertEqual(read([200, 300])[1:4:2], ["rest_1", "rest_1"])
