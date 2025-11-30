import unittest

from homr.transformer.vocabulary import EncodedSymbol
from training.transformer.training_vocabulary import (
    token_lines_to_str,
)

data = """clef_G1 _ _ _ upper&clef_F3 _ _ _ lower
keySignature_7 . . . .
timeSignature/4 . . . .
note_32 D5 # _ upper
clef_C2 _ _ _ lower
repeatEnd . . . ."""


class TestTrainingVocabulary(unittest.TestCase):
    def test_sort_token_chords(self) -> None:
        chord = [
            EncodedSymbol("note_8", "C4", articulation="staccatissimo", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_8", "E5", position="lower"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_16", "E4", articulation="tieStart_tenuto", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_32", "D4", articulation="tieStart_slurStop", position="upper"),
        ]
        result = token_lines_to_str(chord)
        self.maxDiff = None
        expected = (
            "note_16 E4 . slurStop_staccatissimo_tenuto_tieStart upper&note_32 D4 . _ upper"
            "&note_8 C4 . _ upper&note_8 E5 . . lower"
        )
        self.assertEqual(result, expected)
