import unittest

from homr.circle_of_fifths import strip_naturals
from homr.transformer.vocabulary import EncodedSymbol


class TestCircleOfFifths(unittest.TestCase):

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
