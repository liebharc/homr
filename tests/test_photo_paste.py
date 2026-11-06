import unittest

import numpy as np

from training.omr_datasets.photo_maps import FH, FW, PhotoConditions
from training.omr_datasets.photo_paste import engrave_on_template, put_on_photo


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


class TestPhotoPaste(unittest.TestCase):
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
