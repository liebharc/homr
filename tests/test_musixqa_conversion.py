import unittest
from fractions import Fraction

from homr.transformer.vocabulary import EncodedSymbol
from training.datasets.convert_musixqa import (
    convert_duration,
    convert_piece_to_homr,
    interleave_staves,
    parse_pitch,
)


class TestMusixQAConversion(unittest.TestCase):

    def test_parse_pitch(self):
        self.assertEqual(parse_pitch("C4"), ("C4", "_"))
        self.assertEqual(parse_pitch("F#4"), ("F4", "#"))
        self.assertEqual(parse_pitch("Bb3"), ("B3", "b"))
        self.assertEqual(parse_pitch("rest"), (".", "."))
        self.assertEqual(parse_pitch(""), (".", "."))

    def test_convert_duration(self):
        self.assertEqual(convert_duration("1/4"), "4")
        self.assertEqual(convert_duration("1/8"), "8")
        self.assertEqual(convert_duration("1/2"), "2")
        self.assertEqual(convert_duration("3/8"), "4.")
        self.assertEqual(convert_duration("7/16"), "4..")
        with self.assertRaises(ValueError):
            convert_duration("1/5")

    def test_interleave_staves_simple(self):
        bars = [
            {
                "staves": {
                    "treble": [
                        {"duration": "1/4", "pitch": "C4"},
                        {"duration": "1/4", "pitch": "E4"},
                    ],
                    "bass": [{"duration": "1/2", "pitch": "C3"}],
                }
            }
        ]
        # is_grandstaff removed. Passing is_last_system=True to get bolddoublebarline
        result = interleave_staves(bars, is_last_system=True)

        # Expected:
        # Offset 0: note_4 C4 _, note_2 C3 _
        # Offset 1/4: note_4 E4 _
        # Offset 1/2: bolddoublebarline (last bar of last system)

        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 2)
        # Convert symbols to strings for comparison
        res_str = [[str(s) for s in chord] for chord in result]

        self.assertEqual(res_str[0], ["note_4 C4 _ _ upper", "note_2 C3 _ _ upper"])
        self.assertEqual(res_str[1], ["note_4 E4 _ _ upper"])
        self.assertEqual(res_str[2], ["bolddoublebarline . . . ."])

    def test_interleave_staves_intermediate_system(self):
        bars = [{"staves": {"treble": [{"duration": "1/4", "pitch": "C4"}]}}]
        # Not last system -> normal barline
        result = interleave_staves(bars, is_last_system=False)
        res_str = [[str(s) for s in chord] for chord in result]
        self.assertEqual(res_str[1], ["barline . . . ."])

    def test_interleave_staves_with_ties(self):
        bars = [
            {
                "staves": {
                    "treble": [
                        {"duration": "1/4", "pitch": "C4", "tie": True},
                        {"duration": "1/4", "pitch": "C4", "tie": True},
                        {"duration": "1/4", "pitch": "C4"},
                    ]
                }
            }
        ]
        result = interleave_staves(bars, is_last_system=True)
        res_str = [[str(s) for s in chord] for chord in result]

        # Expected:
        # Note 1: tieStart
        # Note 2: tieStart_tieStop
        # Note 3: tieStop
        self.assertEqual(res_str[0], ["note_4 C4 _ tieStart upper"])
        self.assertEqual(res_str[1], ["note_4 C4 _ tieStart_tieStop upper"])
        self.assertEqual(res_str[2], ["note_4 C4 _ tieStop upper"])
        self.assertEqual(res_str[3], ["bolddoublebarline . . . ."])

    def test_interleave_staves_with_repeats(self):
        bars = [
            {"repeat": "start", "staves": {"treble": [{"duration": "1/4", "pitch": "G4"}]}},
            {"repeat": "end", "staves": {"treble": [{"duration": "1/4", "pitch": "G4"}]}},
        ]
        result = interleave_staves(bars)
        res_str = [[str(s) for s in chord] for chord in result]

        # Expected:
        # Bar 1: repeatStart, note_4 G4 _, barline
        # Bar 2: note_4 G4 _, repeatEnd
        self.assertEqual(res_str[0], ["repeatStart . . . ."])
        self.assertEqual(res_str[1], ["note_4 G4 _ _ upper"])
        self.assertEqual(res_str[2], ["barline . . . ."])
        self.assertEqual(res_str[3], ["note_4 G4 _ _ upper"])
        self.assertEqual(res_str[4], ["repeatEnd . . . ."])

    def test_convert_piece_to_homr(self):
        piece_data = {"key": "G Major", "time_signature": "4/4"}
        bars = [
            {
                "staves": {
                    "treble": [{"duration": "1/4", "pitch": "G4"}],
                    "bass": [{"duration": "1/4", "pitch": "G3"}],
                }
            }
        ]
        # is_first_system=True, is_last_system=True
        tokens = convert_piece_to_homr(piece_data, bars, is_first_system=True, is_last_system=True)

        # Expected structure:
        # clef_F4 _ _ _ upper (since bass present, simplified logic)
        # keySignature_1 . . . .
        # timeSignature/4 . . . . (since first system)
        # bolddoublebarline . . . . (last bar)

        self.assertIn("clef_F4 _ _ _ upper", tokens)
        self.assertNotIn("clef_G2", tokens)  # Should only be one clef
        self.assertIn("keySignature_1", tokens)
        self.assertIn("timeSignature/4", tokens)
        self.assertIn("note_4 G4 _ _ upper&note_4 G3 _ _ upper", tokens)
        self.assertIn("bolddoublebarline . . . .", tokens)

    def test_convert_piece_to_homr_no_time_sig(self):
        piece_data = {"key": "C Major", "time_signature": "4/4"}
        bars = [{"staves": {"treble": [{"duration": "1/4", "pitch": "C4"}]}}]

        # Not first system, IS last system
        tokens = convert_piece_to_homr(piece_data, bars, is_first_system=False, is_last_system=True)

        self.assertNotIn("timeSignature/4", tokens)
        self.assertIn("clef_G2 _ _ _ upper", tokens)
        self.assertIn("bolddoublebarline . . . .", tokens)

    def test_repeat_start_replaces_barline(self):
        bars = [
            {"staves": {"treble": [{"duration": "1/4", "pitch": "C4"}]}},
            {"repeat": "start", "staves": {"treble": [{"duration": "1/4", "pitch": "D4"}]}},
        ]
        result = interleave_staves(bars)
        # Bar 0 ends with barline.
        # Bar 1 starts with repeatStart.
        # Expected sequence: C4 -> repeatStart -> D4 -> barline
        # The barline after C4 should be removed.
        res_str = [[str(s) for s in chord] for chord in result]

        self.assertEqual(res_str[0], ["note_4 C4 _ _ upper"])
        self.assertEqual(res_str[1], ["repeatStart . . . ."])
        self.assertEqual(res_str[2], ["note_4 D4 _ _ upper"])
        self.assertEqual(res_str[3], ["barline . . . ."])

    def test_repeat_end_replaces_barline(self):
        # Verify repeatEnd works as intended (replaces barline/bolddoublebarline)
        bars = [{"repeat": "end", "staves": {"treble": [{"duration": "1/4", "pitch": "C4"}]}}]

        # Test 1: Normal bar (not last system). Should use repeatEnd instead of barline.
        result = interleave_staves(bars, is_last_system=False)
        res_str = [[str(s) for s in chord] for chord in result]
        # C4 -> repeatEnd
        self.assertEqual(res_str[0], ["note_4 C4 _ _ upper"])
        self.assertEqual(res_str[1], ["repeatEnd . . . ."])
        # Ensure no barline appended after
        self.assertEqual(len(res_str), 2)

        # Test 2: Last system. Should use repeatEnd instead of bolddoublebarline.
        result = interleave_staves(bars, is_last_system=True)
        res_str = [[str(s) for s in chord] for chord in result]
        self.assertEqual(res_str[1], ["repeatEnd . . . ."])
        self.assertNotIn(["bolddoublebarline . . . ."], res_str)


if __name__ == "__main__":
    unittest.main()
