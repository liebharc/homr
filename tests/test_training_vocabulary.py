import unittest

from homr.transformer.vocabulary import EncodedSymbol, build_state
from training.transformer.training_vocabulary import (
    read_token_lines,
    to_decoder_branches,
    token_lines_to_str,
)

data = """clef_G1 _ _ _ upper&clef_F3 _ _ _ lower
keySignature_7 . . . .
timeSignature/4 . . . .
note_32 D5 # _ upper
clef_C2 _ _ _ lower
repeatEnd . . . ."""


class TestTrainingVocabulary(unittest.TestCase):
    def test_track_state(self) -> None:
        inv_state_vocab = {v: k for k, v in build_state().items()}
        tokens = read_token_lines(data.splitlines())
        result = to_decoder_branches(tokens)
        npresult = result.states.numpy()
        states = [inv_state_vocab[s] for s in npresult if s != 0]
        self.assertEqual(
            [
                "keySignature_0+clef_G2+clef_F4",
                "keySignature_0+clef_G2+clef_F4",
                "keySignature_0+clef_G1+clef_F4",
                "keySignature_0+clef_G1+clef_F4",
                "keySignature_0+clef_G1+clef_F3",
                "keySignature_7+clef_G1+clef_F3",
                "keySignature_7+clef_G1+clef_F3",
                "keySignature_7+clef_G1+clef_F3",
                "keySignature_7+clef_G1+clef_C2",
                "keySignature_7+clef_G1+clef_C2",
            ],
            states,
        )

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
