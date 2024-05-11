import unittest

from homr.results import ClefType, ResultClef, ResultPitch, move_pitch_to_clef


class TestResultModel(unittest.TestCase):

    def test_change_staff(self) -> None:
        treble = ResultClef(ClefType.TREBLE, 1)
        bass = ResultClef(ClefType.BASS, 1)
        self.assertEqual(
            str(move_pitch_to_clef(treble.get_reference_pitch(), treble, bass)),
            str(bass.get_reference_pitch()),
        )
        self.assertEqual(
            str(move_pitch_to_clef(bass.get_reference_pitch(), bass, treble)),
            str(treble.get_reference_pitch()),
        )

        self.assertEqual(str(move_pitch_to_clef(ResultPitch("E", 2, 1), bass, treble)), "C4#")
        self.assertEqual(str(move_pitch_to_clef(ResultPitch("F", 2, 0), bass, treble)), "D4♮")
        self.assertEqual(str(move_pitch_to_clef(ResultPitch("G", 2, -1), bass, treble)), "E4b")
        self.assertEqual(str(move_pitch_to_clef(ResultPitch("A", 2, None), bass, treble)), "F4")

        self.assertEqual(str(move_pitch_to_clef(ResultPitch("B", 2, 1), bass, treble)), "G4#")
        self.assertEqual(str(move_pitch_to_clef(ResultPitch("C", 3, 0), bass, treble)), "A4♮")
        self.assertEqual(str(move_pitch_to_clef(ResultPitch("D", 3, -1), bass, treble)), "B4b")
        self.assertEqual(str(move_pitch_to_clef(ResultPitch("E", 3, None), bass, treble)), "C5")

        self.assertEqual(str(move_pitch_to_clef(ResultPitch("F", 3, 1), bass, treble)), "D5#")
        self.assertEqual(str(move_pitch_to_clef(ResultPitch("G", 3, 0), bass, treble)), "E5♮")
        self.assertEqual(str(move_pitch_to_clef(ResultPitch("A", 3, -1), bass, treble)), "F5b")
        self.assertEqual(str(move_pitch_to_clef(ResultPitch("B", 3, None), bass, treble)), "G5")

        self.assertEqual(str(move_pitch_to_clef(ResultPitch("C", 4, 1), bass, treble)), "A5#")
