import unittest
from fractions import Fraction

from homr.transformer.vocabulary import (
    EncodedSymbol,
    kern_to_symbol_duration,
    remove_duplicated_symbols,
)
from training.transformer.training_vocabulary import (
    read_token_lines,
    token_lines_to_str,
)


def from_rhythm(rhythm: str) -> EncodedSymbol:
    return EncodedSymbol(rhythm)


def remove_tuplet(rhythm: str) -> str:
    return EncodedSymbol(rhythm).remove_tuplet().rhythm


def get_duration(rhythm: str) -> Fraction:
    return EncodedSymbol(rhythm).get_duration().fraction


class TestVocabulary(unittest.TestCase):
    def test_is_tuplet(self) -> None:
        self.assertTrue(from_rhythm("note_12").is_tuplet())
        self.assertTrue(from_rhythm("note_12.").is_tuplet())
        self.assertFalse(from_rhythm("note_4").is_tuplet())
        self.assertTrue(from_rhythm("note_28").is_tuplet())
        self.assertTrue(from_rhythm("note_20").is_tuplet())
        self.assertFalse(from_rhythm("note_16").is_tuplet())
        self.assertFalse(from_rhythm("note_32").is_tuplet())
        self.assertFalse(from_rhythm("clef_F4").is_tuplet())
        self.assertTrue(from_rhythm("rest_12").is_tuplet())
        self.assertFalse(from_rhythm("rest_4").is_tuplet())

    def test_remove_tuplet(self) -> None:
        self.assertEqual(remove_tuplet("note_12"), "note_8")
        self.assertEqual(remove_tuplet("note_12."), "note_8.")
        self.assertEqual(remove_tuplet("note_4"), "note_4")
        self.assertEqual(remove_tuplet("note_28"), "note_16")
        self.assertEqual(remove_tuplet("note_20"), "note_16")
        self.assertEqual(remove_tuplet("note_16"), "note_16")
        self.assertEqual(remove_tuplet("note_32"), "note_32")
        self.assertEqual(remove_tuplet("clef_F4"), "clef_F4")
        self.assertEqual(remove_tuplet("rest_12"), "rest_8")
        self.assertEqual(remove_tuplet("rest_4"), "rest_4")

    def test_remove_redudant_clefs_keys_and_time_signatures(self) -> None:
        symbols = [
            EncodedSymbol("clef_G2", position="upper"),
            EncodedSymbol("clef_F4", position="lower"),
            EncodedSymbol("keySignature_0"),
            EncodedSymbol("timeSignature_/4"),
            EncodedSymbol("note_4"),
            EncodedSymbol("keySignature_0"),
            EncodedSymbol("timeSignature_/4"),
            EncodedSymbol("clef_G2", position="upper"),
            EncodedSymbol("clef_G2", position="lower"),
        ]
        result = remove_duplicated_symbols(symbols)
        self.assertEqual(
            result,
            [
                EncodedSymbol("clef_G2", position="upper"),
                EncodedSymbol("clef_F4", position="lower"),
                EncodedSymbol("keySignature_0"),
                EncodedSymbol("timeSignature_/4"),
                EncodedSymbol("note_4"),
                EncodedSymbol("clef_G2", position="lower"),
            ],
        )

    def test_remove_duplicates_in_chord(self) -> None:
        tokens_str = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_1 . . . .
timeSignature/4 . . . .
note_12 B5 _ tieStop upper&note_12 G5 _ slurStart_tieStop upper&note_12 B5 _ tieStop upper
note_12 G5 _ _ upper&note_12 D5 # staccato upper
note_12 D5 # _ upper&note_12 B4 _ staccato upper&note_24 D5 # _ upper
note_2 B4 _ slurStart_slurStop upper&note_4 G4 _ _ upper&note_4 B3 _ _ lower&note_4 D3 # _ lower
note_4 A4 _ _ upper&note_4 F4 # slurStop upper&note_4 B3 _ _ lower&note_4 D3 # _ lower
barline . . . ."""
        tokens = read_token_lines(tokens_str.splitlines())
        result = remove_duplicated_symbols(tokens)
        expected = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_1 . . . .
timeSignature/4 . . . .
note_12 B5 _ slurStart_tieStop upper&note_12 G5 _ _ upper
note_12 G5 _ staccato upper&note_12 D5 # _ upper
note_12 D5 # staccato upper&note_12 B4 _ _ upper
note_2 B4 _ slurStart_slurStop upper&note_4 G4 _ _ upper&note_4 B3 _ _ lower&note_4 D3 # _ lower
note_4 A4 _ slurStop upper&note_4 F4 # _ upper&note_4 B3 _ _ lower&note_4 D3 # _ lower
barline . . . ."""

        self.maxDiff = None
        self.assertEqual(token_lines_to_str(result), expected)

    def test_durations(self) -> None:
        self.assertEqual(get_duration("note_0"), Fraction(1))
        self.assertEqual(get_duration("note_1"), Fraction(1))
        self.assertEqual(get_duration("note_2"), Fraction(1, 2))
        self.assertEqual(get_duration("note_4"), Fraction(1, 4))
        self.assertEqual(get_duration("note_8"), Fraction(1, 8))
        self.assertEqual(get_duration("note_16"), Fraction(1, 16))
        self.assertEqual(get_duration("note_2."), Fraction(3, 4))
        self.assertEqual(get_duration("note_4."), Fraction(3, 8))
        self.assertEqual(get_duration("note_8."), Fraction(3, 16))
        self.assertEqual(get_duration("note_16."), Fraction(3, 32))
        self.assertEqual(get_duration("note_4.."), Fraction(7, 16))
        self.assertEqual(get_duration("note_12"), Fraction(1, 12))

    def test_remove_incorrect_tuplets(self) -> None:
        tokens = [
            EncodedSymbol("clef_G2", "_", "_", "_", "upper"),
            EncodedSymbol("keySignature_1", ".", ".", ".", "."),
            EncodedSymbol("note_8", "F4", "#", "slurStart", "upper"),
            EncodedSymbol("note_8", "E4", "_", "slurStop", "upper"),
            EncodedSymbol("note_24", "E4", "_", "slurStart", "upper"),
            EncodedSymbol("note_24", "F4", "#", "_", "upper"),
            EncodedSymbol("note_12", "G4", "_", "tieStart", "upper"),
            EncodedSymbol("note_4", "G4", "_", "slurStop_tieStop", "upper"),
            EncodedSymbol("note_8", "F4", "#", "slurStart", "upper"),
            EncodedSymbol("note_8", "E4", "_", "slurStop", "upper"),
            EncodedSymbol("barline", ".", ".", ".", "."),
            EncodedSymbol("note_1", "D4", "_", "_", "upper"),
            EncodedSymbol("chord", ".", ".", ".", "."),
            EncodedSymbol("rest_0", "D4", "_", "_", "upper"),
            EncodedSymbol("barline", ".", ".", ".", "."),
            EncodedSymbol("note_4", "F4", "#", "slurStop_tieStart", "upper"),
            EncodedSymbol("note_8", "F4", "#", "tieStop", "upper"),
            EncodedSymbol("note_16", "G4", "_", "slurStart", "upper"),
            EncodedSymbol("note_16", "F4", "#", "_", "upper"),
            EncodedSymbol("note_16", "E4", "_", "_", "upper"),
            EncodedSymbol("note_16", "F4", "#", "_", "upper"),
            EncodedSymbol("note_8", "D4", "_", "slurStop", "upper"),
            EncodedSymbol("barline", ".", ".", ".", "."),
            EncodedSymbol("note_2.", "D5", "_", "_", "upper"),
            EncodedSymbol("note_4", "F4", "N", "_", "upper"),
            EncodedSymbol("barline", ".", ".", ".", "."),
        ]
        result = remove_duplicated_symbols(tokens)
        expected = [
            EncodedSymbol("clef_G2", "_", "_", "_", "upper"),
            EncodedSymbol("keySignature_1", ".", ".", ".", "."),
            EncodedSymbol("note_8", "F4", "#", "slurStart", "upper"),
            EncodedSymbol("note_8", "E4", "_", "slurStop", "upper"),
            EncodedSymbol("note_16", "E4", "_", "slurStart", "upper"),
            EncodedSymbol("note_16", "F4", "#", "_", "upper"),
            EncodedSymbol("note_8", "G4", "_", "tieStart", "upper"),
            EncodedSymbol("note_4", "G4", "_", "slurStop_tieStop", "upper"),
            EncodedSymbol("note_8", "F4", "#", "slurStart", "upper"),
            EncodedSymbol("note_8", "E4", "_", "slurStop", "upper"),
            EncodedSymbol("barline", ".", ".", ".", "."),
            EncodedSymbol("note_1", "D4", "_", "_", "upper"),
            EncodedSymbol("barline", ".", ".", ".", "."),
            EncodedSymbol("note_4", "F4", "#", "slurStop_tieStart", "upper"),
            EncodedSymbol("note_8", "F4", "#", "tieStop", "upper"),
            EncodedSymbol("note_16", "G4", "_", "slurStart", "upper"),
            EncodedSymbol("note_16", "F4", "#", "_", "upper"),
            EncodedSymbol("note_16", "E4", "_", "_", "upper"),
            EncodedSymbol("note_16", "F4", "#", "_", "upper"),
            EncodedSymbol("note_8", "D4", "_", "slurStop", "upper"),
            EncodedSymbol("barline", ".", ".", ".", "."),
            EncodedSymbol("note_2.", "D5", "_", "_", "upper"),
            EncodedSymbol("note_4", "F4", "N", "_", "upper"),
            EncodedSymbol("barline", ".", ".", ".", "."),
        ]

        self.assertEqual(token_lines_to_str(result), token_lines_to_str(expected))

    def test_only_keep_lower_staff_if_there_is_a_clef(self) -> None:
        tokens = [
            EncodedSymbol("clef_G2", "_", "_", "_", "upper"),
            EncodedSymbol("keySignature_1", ".", ".", ".", "."),
            EncodedSymbol("note_8", "F4", "#", "slurStart", "upper"),
            EncodedSymbol("note_8", "E4", "_", "slurStop", "upper"),
            EncodedSymbol("note_24", "E4", "_", "slurStart", "upper"),
            EncodedSymbol("note_24", "F4", "#", "_", "lower"),
        ]
        result = remove_duplicated_symbols(tokens)
        expected = [
            EncodedSymbol("clef_G2", "_", "_", "_", "upper"),
            EncodedSymbol("keySignature_1", ".", ".", ".", "."),
            EncodedSymbol("note_8", "F4", "#", "slurStart", "upper"),
            EncodedSymbol("note_8", "E4", "_", "slurStop", "upper"),
            EncodedSymbol("note_24", "E4", "_", "slurStart", "upper"),
            EncodedSymbol("note_24", "F4", "#", "_", "upper"),
        ]

        self.assertEqual(token_lines_to_str(result), token_lines_to_str(expected))

    def test_tuplet_duration(self) -> None:
        for kern in ["6", "12"]:
            duration = kern_to_symbol_duration(kern)
            self.assertEqual(duration.normal_notes, 2)
            self.assertEqual(duration.actual_notes, 3)

    def test_normal_duration(self) -> None:
        for kern in ["2", "4", "8"]:
            duration = kern_to_symbol_duration(kern)
            self.assertEqual(duration.normal_notes, 1)
            self.assertEqual(duration.actual_notes, 1)
