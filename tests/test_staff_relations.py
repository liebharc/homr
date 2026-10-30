import unittest

import numpy as np

from homr.model import InputPredictions, Staff, StaffPoint
from homr.staff_relations import (
    channels,
    crop_height,
    crop_width,
    features,
    group_staffs,
    horizontal_pairs,
    join_staff_pieces,
    page_inputs,
    vertical_pairs,
)
from training.staff_relations.render_parts import keep_part, remove_lyrics


def make_staff(min_x: float, max_x: float, top: float) -> Staff:
    y = [top + 10 * i for i in range(5)]
    return Staff([StaffPoint(x, y, 0) for x in range(int(min_x), int(max_x) + 1, 10)])


separate = [0.9, 0.05, 0.05]
same_system = [0.1, 0.8, 0.1]
grand_staff = [0.1, 0.2, 0.7]


class TestStaffRelations(unittest.TestCase):
    def test_vertical_pairs_skip_staffs_in_the_same_row(self) -> None:
        staffs = [make_staff(0, 400, 0), make_staff(500, 600, 0), make_staff(0, 600, 100)]
        self.assertEqual(vertical_pairs(staffs), [(0, 2), (1, 2)])

    def test_page_inputs_shapes(self) -> None:
        staffs = [make_staff(100, 800, 100), make_staff(100, 800, 250)]
        image = np.full((500, 1000), 255, dtype=np.uint8)
        mask = np.zeros((500, 1000), dtype=np.uint8)
        predictions = InputPredictions(image, image, mask, mask, mask, mask, mask)
        pairs, crops, feats = page_inputs(predictions, staffs)
        self.assertEqual(pairs, [(0, 1)])
        self.assertEqual(crops.shape, (1, channels, crop_height, crop_width))
        self.assertEqual(feats.shape, (1, features))

    def test_group_staffs(self) -> None:
        staffs = [make_staff(0, 600, 100 * i) for i in range(6)]
        pairs = [(i, i + 1) for i in range(5)]
        probabilities = np.array([same_system, grand_staff, separate, same_system, grand_staff])
        systems = group_staffs(staffs, pairs, probabilities)
        self.assertEqual(
            [[s.is_grandstaff for s in m.staffs] for m in systems], [[False, True]] * 2
        )

    def test_a_staff_is_in_at_most_one_grand_staff(self) -> None:
        staffs = [make_staff(0, 600, 100 * i) for i in range(3)]
        pairs = [(0, 1), (1, 2)]
        probabilities = np.array([[0.1, 0.3, 0.6], grand_staff])
        systems = group_staffs(staffs, pairs, probabilities)
        self.assertEqual([[s.is_grandstaff for s in m.staffs] for m in systems], [[False, True]])

    def test_keep_part(self) -> None:
        musicxml = (
            '<part-list><score-part id="P1"><part-name>Voice</part-name></score-part>'
            '<score-part id="P2"><part-name>Piano</part-name></score-part></part-list>'
            '<part id="P1"><measure><print new-system="yes"/></measure></part>'
            '<part id="P2"><measure/></part>'
        )
        voice = keep_part(musicxml, 0) or ""
        self.assertNotIn("P2", voice)
        self.assertNotIn("new-system", voice)
        self.assertIn('<part id="P1">', voice)
        self.assertIsNone(keep_part('<score-part id="P1"></score-part>', 0))

    def test_remove_lyrics_keeps_the_lyric_font(self) -> None:
        musicxml = (
            '<defaults><lyric-font font-size="10"/></defaults>'
            '<note><pitch/><lyric number="1"><text>la</text></lyric></note>'
        )
        self.assertEqual(
            remove_lyrics(musicxml),
            '<defaults><lyric-font font-size="10"/></defaults><note><pitch/></note>',
        )

    def test_join_staff_pieces(self) -> None:
        staffs = [make_staff(0, 300, 0), make_staff(350, 600, 2), make_staff(0, 600, 100)]
        pairs = horizontal_pairs(staffs)
        self.assertEqual(pairs, [(0, 1)])
        joined = join_staff_pieces(staffs, pairs, np.array([[0.1, 0.9]]))
        self.assertEqual([(s.min_x, s.max_x) for s in joined], [(0, 600), (0, 600)])
        kept = join_staff_pieces(staffs, pairs, np.array([[0.9, 0.1]]))
        self.assertEqual(len(kept), 3)
