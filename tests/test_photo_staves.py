import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.omr_datasets.photo_look import NO_CURVE, Look, spots
from training.omr_datasets.photo_staves import (
    BLUR_LIMIT,
    answers,
    match_staff,
    staff_blur,
    targets,
    too_blurry,
)

NOTE = (
    "<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration>"
    "<type>whole</type></note>"
)
ATTRIBUTES = (
    "<attributes><divisions>1</divisions><key><fifths>0</fifths></key>"
    "<time><beats>4</beats><beat-type>4</beat-type></time>"
    "<clef><sign>G</sign><line>2</line></clef></attributes>"
)
BOX = [100.0, 200.0, 2100.0, 260.0]


def one_part(bars: int, first: str = "") -> str:
    measures = f'<measure number="1">{ATTRIBUTES}{first}{NOTE}</measure>' + "".join(
        f'<measure number="{n}">{NOTE}</measure>' for n in range(2, bars + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><score-partwise version="3.1"><part-list>'
        '<score-part id="P1"><part-name>Voice</part-name></score-part></part-list>'
        f'<part id="P1">{measures}</part></score-partwise>'
    )


def bar_count(tokens: str) -> int:
    return sum(1 for line in tokens.splitlines() if line.split()[0] == "barline")


class TestPhotoStaves(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def write(self, xml: str) -> str:
        path = Path(self.folder.name) / "song.musicxml"
        path.write_text(xml, encoding="utf-8")
        return str(path)

    def test_answers_cut_by_bars(self) -> None:
        systems = [{"bars": 5, "staffs": [BOX]}, {"bars": 3, "staffs": [BOX]}]
        tokens, staves = answers(self.write(one_part(8)), systems)
        self.assertEqual(staves, [1])
        assert tokens is not None  # noqa: S101
        self.assertEqual(set(tokens), {(0, 0), (1, 0)})
        self.assertEqual([bar_count(tokens[0, 0] or ""), bar_count(tokens[1, 0] or "")], [5, 3])
        self.assertTrue(
            (tokens[1, 0] or "").startswith("clef_G2")
        )  # each staff starts with its clef

    def test_answers_that_dont_fill_the_staves_are_none(self) -> None:
        tokens, _ = answers(self.write(one_part(8)), [{"bars": 4, "staffs": [BOX]}])
        self.assertIsNone(tokens)

    def test_unreadable_song_is_skipped_not_raised(self) -> None:
        path = self.write(one_part(4)[:-40])  # cut off: not valid MusicXML
        self.assertEqual(answers(path, [{"bars": 4, "staffs": [BOX]}]), (None, []))

    def test_a_grand_staff_is_one_target(self) -> None:
        low = [100.0, 300.0, 2100.0, 360.0]
        found = targets([{"bars": 4, "staffs": [BOX, low]}], [2])
        self.assertEqual(found, [((0, 0), [100.0, 200.0, 2100.0, 360.0])])

    def test_match_staff_needs_both_ends_on_a_drawn_staff(self) -> None:
        drawn = [((0, 0), BOX)]
        self.assertEqual(match_staff(((110, 230), (1100, 230), (2090, 230)), drawn), (0, 0))
        # the left end 200 px off: more than 4% of the staff's 2000 px
        self.assertIsNone(match_staff(((300, 230), (1100, 230), (2090, 230)), drawn))
        # the middle above the staff
        self.assertIsNone(match_staff(((110, 150), (1100, 150), (2090, 150)), drawn))

    def test_too_blurry_at_the_limit(self) -> None:
        self.assertTrue(too_blurry(BLUR_LIMIT * 10 + 0.01, 10))
        self.assertFalse(too_blurry(BLUR_LIMIT * 10 - 0.01, 10))

    def test_staff_blur_reads_the_field(self) -> None:
        shape = (480, 640)
        rows, cols = spots(shape)
        blur = np.zeros((rows, cols, 3), np.float32)
        blur[..., 1] = 0.8
        blur[:, : cols // 2, 0] = 1.0
        blur[:, cols // 2 :, 0] = 3.0
        look = Look(blur, np.ones((rows, cols), np.float32), (0.0, 1.0), NO_CURVE)
        self.assertAlmostEqual(staff_blur(look, (100, 240), shape), 1.0, delta=0.1)
        self.assertAlmostEqual(staff_blur(look, (590, 240), shape), 3.0, delta=0.1)
