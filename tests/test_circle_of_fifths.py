import unittest

from homr.circle_of_fifths import (
    convert_engraving_to_sounding_representation,
    convert_sounding_to_engraving_representation,
)
from homr.transformer.vocabulary import EncodedSymbol


class TestCircleOfFifths(unittest.TestCase):

    def test_convert_sounding_to_engraving_representation(self) -> None:
        symbols = [
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("note_4", "A4", "_"),
            EncodedSymbol("note_4", "F5", "_"),
        ]
        result = convert_sounding_to_engraving_representation(symbols)
        expected = [
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("note_4", "A4", "_"),
            EncodedSymbol("note_4", "F5", "N"),
        ]
        self.assertEqual(result, expected)

    def test_convert_engraving_to_sounding_representation(self) -> None:
        symbols = [
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("note_4", "A4", "_"),
            EncodedSymbol("note_4", "F5", "_"),
        ]
        result = convert_engraving_to_sounding_representation(symbols)
        expected = [
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("note_4", "A4", "_"),
            EncodedSymbol("note_4", "F5", "#"),
        ]
        self.assertEqual(result, expected)
