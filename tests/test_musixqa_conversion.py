import unittest
from fractions import Fraction
from training.datasets.convert_musixqa import (
    parse_pitch,
    convert_duration,
    interleave_staves,
    convert_piece_to_homr
)
from homr.transformer.vocabulary import EncodedSymbol

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
                        {"duration": "1/4", "pitch": "E4"}
                    ],
                    "bass": [
                        {"duration": "1/2", "pitch": "C3"}
                    ]
                }
            }
        ]
        # is_grandstaff = True
        result = interleave_staves(bars, True)
        
        # Expected:
        # Offset 0: note_4 C4 _, note_2 C3 _
        # Offset 1/4: note_4 E4 _
        # Offset 1/2: barline
        
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 2)
        # Convert symbols to strings for comparison
        res_str = [[str(s) for s in chord] for chord in result]
        
        self.assertEqual(res_str[0], ["note_4 C4 _ _ upper", "note_2 C3 _ _ lower"])
        self.assertEqual(res_str[1], ["note_4 E4 _ _ upper"])
        self.assertEqual(res_str[2], ["barline . . . ."])

    def test_interleave_staves_with_ties(self):
        bars = [
            {
                "staves": {
                    "treble": [
                        {"duration": "1/4", "pitch": "C4", "tie": True},
                        {"duration": "1/4", "pitch": "C4", "tie": True},
                        {"duration": "1/4", "pitch": "C4"}
                    ]
                }
            }
        ]
        result = interleave_staves(bars, False)
        res_str = [[str(s) for s in chord] for chord in result]
        
        # Expected:
        # Note 1: tieStart
        # Note 2: tieStart_tieStop
        # Note 3: tieStop
        self.assertEqual(res_str[0], ["note_4 C4 _ tieStart upper"])
        self.assertEqual(res_str[1], ["note_4 C4 _ tieStart_tieStop upper"])
        self.assertEqual(res_str[2], ["note_4 C4 _ tieStop upper"])
        self.assertEqual(res_str[3], ["barline . . . ."])

    def test_interleave_staves_with_repeats(self):
        bars = [
            {
                "repeat": "start",
                "staves": {"treble": [{"duration": "1/4", "pitch": "G4"}]}
            },
            {
                "repeat": "end",
                "staves": {"treble": [{"duration": "1/4", "pitch": "G4"}]}
            }
        ]
        result = interleave_staves(bars, False)
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
        piece_data = {
            "key": "G Major",
            "time_signature": "4/4"
        }
        bars = [
            {
                "staves": {
                    "treble": [{"duration": "1/4", "pitch": "G4"}],
                    "bass": [{"duration": "1/4", "pitch": "G3"}]
                }
            }
        ]
        tokens = convert_piece_to_homr(piece_data, bars, True)
        
        # Expected structure:
        # clef_G2 _ _ _ upper, clef_F4 _ _ _ lower, keySignature_1 . . . ., timeSignature/4 . . . ., note_4 G4 _ _ upper&note_4 G3 _ _ lower, barline . . . .
        
        self.assertIn("clef_G2 . . . upper", tokens)
        self.assertIn("clef_F4 . . . lower", tokens)
        self.assertIn("keySignature_1", tokens)
        self.assertIn("timeSignature/4", tokens)
        self.assertIn("note_4 G4 _ _ upper&note_4 G3 _ _ lower", tokens)
        self.assertIn("barline . . . .", tokens)

if __name__ == "__main__":
    unittest.main()
