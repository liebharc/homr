import unittest

from homr.transformer.vocabulary import EncodedSymbol, Vocabulary
from training.transformer.training_vocabulary import (
    CLEF_ANCHORS,
    max_ledger_lines,
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
            EncodedSymbol("note_16", "E4", articulation="tenuto", slur="slurStart", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_32", "D4", slur="slurStart_slurStop", position="upper"),
        ]
        result = token_lines_to_str(chord)
        self.maxDiff = None
        expected = (
            "note_16 E4 . staccatissimo_tenuto slurStart_slurStop upper&note_32 D4 . _ _ upper"
            "&note_8 C4 . _ _ upper&note_8 E5 . . . lower"
        )
        self.assertEqual(result, expected)

    def test_clef_anchor_coverage(self) -> None:
        vocabulary = Vocabulary()
        clef_tokens = {token for token in vocabulary.rhythm if token.startswith("clef_")}
        self.assertEqual(clef_tokens, set(CLEF_ANCHORS.keys()))

    def test_max_ledger_lines(self) -> None:
        tokens = [
            EncodedSymbol("note_4", "C6", ".", "_", "upper"),
            EncodedSymbol("note_4", "C4", ".", "_", "lower"),
        ]

        # Upper anchor is clef_G2 (B4) and lower anchor is clef_F4 (D3).
        # C6 is 2 ledger lines above the treble staff and C4 is 1 ledger line above the bass staff.
        self.assertEqual(max_ledger_lines(tokens), 2)
