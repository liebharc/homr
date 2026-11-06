import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.photo_staves import answers, match_staff, targets

NOTE = (
    "<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration>"
    "<type>whole</type></note>"
)
ATTRIBUTES = (
    "<attributes><divisions>1</divisions><key><fifths>0</fifths></key>"
    "<time><beats>4</beats><beat-type>4</beat-type></time>"
    "<clef><sign>G</sign><line>2</line></clef></attributes>"
)
OCTAVE_SHIFT = (
    '<direction><direction-type><octave-shift type="down" size="8"/></direction-type></direction>'
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
        self.assertTrue((tokens[1, 0] or "").startswith("clef_G2"))  # each staff starts with its clef

    def test_answers_that_dont_fill_the_staves_are_none(self) -> None:
        tokens, _ = answers(self.write(one_part(8)), [{"bars": 4, "staffs": [BOX]}])
        self.assertIsNone(tokens)

    def test_unreadable_song_is_skipped_not_raised(self) -> None:
        path = self.write(one_part(4, first=OCTAVE_SHIFT))
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
