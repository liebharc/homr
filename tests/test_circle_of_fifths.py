import unittest

from homr.circle_of_fifths import (
    maintain_accidentals_during_measure,
    strip_naturals,
)
from homr.transformer.vocabulary import EncodedSymbol


class TestCircleOfFifths(unittest.TestCase):

    def test_maintain_accidentals_during_measure_with_key_and_barlines(self) -> None:
        symbols = [
            EncodedSymbol("keySignature_1"),
            EncodedSymbol("note_2", "F4", "_"),
            EncodedSymbol("note_4", "G4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("barline"),
            EncodedSymbol("note_1", "G4", "_"),
        ]
        result = maintain_accidentals_during_measure(symbols)
        expected = [
            EncodedSymbol("keySignature_1"),
            EncodedSymbol(
                "note_2", "F4", "_"
            ),  # the PrIMus datset encodes the keys already correctly
            EncodedSymbol("note_4", "G4", "#"),
            EncodedSymbol("note_4", "G4", "#"),
            EncodedSymbol("barline"),
            EncodedSymbol("note_1", "G4", "_"),
        ]
        self.assertEqual(result, expected)

    def test_maintain_accidentals_during_measure(self) -> None:
        symbols = [
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("note_4", "A4", "_"),
            EncodedSymbol("note_4", "F4", "_"),
        ]
        result = maintain_accidentals_during_measure(symbols)
        expected = [
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("note_4", "A4", "_"),
            EncodedSymbol("note_4", "F4", "#"),
        ]
        self.assertEqual(result, expected)

    def test_strip_naturals(self) -> None:
        symbols = [
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("note_4", "A4", "_"),
            EncodedSymbol("note_4", "F5", "N"),
        ]
        result = strip_naturals(symbols)
        expected = [
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("note_4", "A4", "_"),
            EncodedSymbol("note_4", "F5", "_"),
        ]
        self.assertEqual(result, expected)
