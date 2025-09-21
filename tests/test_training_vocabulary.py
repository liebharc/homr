import unittest

from homr.transformer.vocabulary import build_state
from training.transformer.training_vocabulary import (
    read_token_lines,
    to_decoder_branches,
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
