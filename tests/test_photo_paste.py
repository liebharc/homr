import unittest
import xml.etree.ElementTree as ET

import numpy as np

from training.omr_datasets.photo_maps import FH, FW, PhotoConditions
from training.omr_datasets.photo_paste import (
    bar_count,
    engrave_on_template,
    measures_from,
    put_on_photo,
)


def piano_score(bars: int) -> str:
    """A grand-staff piano score: whole notes in both hands."""

    def note(step: str, octave: int, staff: int) -> str:
        return (
            f"<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>"
            f"<duration>4</duration><voice>{staff}</voice><type>whole</type><staff>{staff}</staff></note>"
        )

    measures = []
    for n in range(1, bars + 1):
        attributes = (
            "<attributes><divisions>1</divisions><key><fifths>0</fifths></key>"
            "<time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves>"
            '<clef number="1"><sign>G</sign><line>2</line></clef>'
            '<clef number="2"><sign>F</sign><line>4</line></clef></attributes>'
            if n == 1
            else ""
        )
        measures.append(
            f'<measure number="{n}">{attributes}{note("E", 4, 1)}'
            f"<backup><duration>4</duration></backup>{note('C', 3, 2)}</measure>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><score-partwise version="3.1"><part-list>'
        '<score-part id="P1"><part-name>Piano</part-name></score-part></part-list>'
        f'<part id="P1">{"".join(measures)}</part></score-partwise>'
    )


def changing_score() -> str:
    """One part, 8 bars: G clef and C major, then from bar 3 on F clef and D major."""
    note = (
        "<note><pitch><step>D</step><octave>3</octave></pitch><duration>4</duration>"
        "<type>whole</type></note>"
    )
    first = (
        "<attributes><divisions>1</divisions><key><fifths>0</fifths></key>"
        "<time><beats>4</beats><beat-type>4</beat-type></time>"
        "<clef><sign>G</sign><line>2</line></clef></attributes>"
    )
    change = (
        "<attributes><key><fifths>2</fifths></key>"
        "<clef><sign>F</sign><line>4</line></clef></attributes>"
    )
    measures = "".join(
        f'<measure number="{n}">{first if n == 1 else change if n == 3 else ""}{note}</measure>'
        for n in range(1, 9)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><score-partwise version="3.1"><part-list>'
        '<score-part id="P1"><part-name>Cello</part-name></score-part></part-list>'
        f'<part id="P1">{measures}</part></score-partwise>'
    )


class TestPhotoPaste(unittest.TestCase):
    def test_bars_from_the_middle_keep_the_clef_key_and_time_in_force(self) -> None:
        part = ET.fromstring(measures_from(changing_score(), 4, 3)).find("part")  # noqa: S314
        assert part is not None  # noqa: S101
        measures = part.findall("measure")
        self.assertEqual([m.get("number") for m in measures], ["5", "6", "7"])
        attributes = measures[0].find("attributes")
        assert attributes is not None  # noqa: S101
        self.assertEqual(attributes.findtext("divisions"), "1")
        self.assertEqual(attributes.findtext("key/fifths"), "2")
        self.assertEqual(attributes.findtext("time/beats"), "4")
        self.assertEqual(attributes.findtext("clef/sign"), "F")

    def test_bar_count_is_the_first_parts_bars(self) -> None:
        self.assertEqual(bar_count(piano_score(12)), 12)

    def test_a_page_can_start_at_a_later_bar(self) -> None:
        page = engrave_on_template(piano_score(12), "Leipzig", 40, start=5)
        assert page is not None  # noqa: S101
        part = ET.fromstring(page.musicxml).find("part")  # noqa: S314
        assert part is not None  # noqa: S101
        self.assertEqual(part.findall("measure")[0].get("number"), "6")
        self.assertEqual(sum(s["bars"] for s in page.systems), 7)

    def test_engraved_page_lists_its_bars_and_staves(self) -> None:
        page = engrave_on_template(piano_score(8), "Leipzig", 40)
        self.assertIsNotNone(page)
        assert page is not None  # noqa: S101
        self.assertEqual(page.flat.shape, (FH, FW))
        self.assertEqual(sum(s["bars"] for s in page.systems), 8)
        self.assertTrue(all(len(s["staffs"]) == 2 for s in page.systems))
        for system in page.systems:
            for left, top, right, bottom in system["staffs"]:
                self.assertTrue(0 <= left < right <= FW and 0 <= top < bottom <= FH)
                line = page.flat[int(top) - 1 : int(top) + 2, int(left) : int(right)]
                self.assertLess(int(line.min()), 128)  # the top staff line is drawn there

    def test_too_short_a_song_gives_no_page(self) -> None:
        self.assertIsNone(engrave_on_template(piano_score(2), "Bravura", 55))

    def test_put_on_photo_only_changes_the_page(self) -> None:
        h, w = 120, 160
        photo = np.random.default_rng(1).integers(150, 230, (h, w, 3), np.uint8)
        on_page = np.zeros((h, w), bool)
        on_page[:, : w // 2] = True
        qx, qy = np.meshgrid(np.linspace(0, FW - 1, w), np.linspace(0, FH - 1, h))
        map_x = np.where(on_page, qx, -1).astype(np.float32)
        map_y = np.where(on_page, qy, -1).astype(np.float32)
        cond = PhotoConditions(photo, map_x, map_y, on_page, photo.astype(np.float32), 0.3, 2.0)
        flat = np.full((FH, FW), 255, np.uint8)
        flat[:, : FW // 4] = 0  # ink on the left quarter of the page
        result = put_on_photo(photo, cond, flat)
        self.assertTrue(np.array_equal(result[:, w // 2 :], photo[:, w // 2 :]))
        self.assertLess(float(result[:, : w // 8].mean()), float(photo[:, : w // 8].mean()) - 50)
