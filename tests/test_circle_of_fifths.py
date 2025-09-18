import unittest

import numpy as np

from homr.circle_of_fifths import (
    agnostic_to_semantic_accidentals,
    semantic_to_agnostic_accidentals,
)
from training.transformer.training_vocabulary import (
    read_token_lines,
    token_lines_to_str,
)

empty = np.array([])


semantic = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_7 . . . .
timeSignature/4 . . . .
note_32 D5 # _ upper&note_8 B3 # _ lower&note_8 B2 # _ lower
note_32 C5 # _ upper
note_16 B4 # _ upper
note_32 C5 # _ upper&note_8 A3 # _ lower&note_8 A2 # _ lower
note_32 B4 # _ upper
note_16 A4 # _ upper
note_8 C5 # _ upper&note_8 A4 # _ upper&note_8 E3 # _ lower&note_8 E2 # _ lower
note_8 B4 # _ upper&note_8 G4 ## _ upper&note_8 E3 # _ lower&note_8 E2 # _ lower
barline . . . .
note_2 A4 # _ upper&note_2 A2 # _ lower
repeatEnd . . . ."""

agnostic = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_7 . . . .
timeSignature/4 . . . .
note_32 D5 _ _ upper&note_8 B3 _ _ lower&note_8 B2 _ _ lower
note_32 C5 _ _ upper
note_16 B4 _ _ upper
note_32 C5 _ _ upper&note_8 A3 _ _ lower&note_8 A2 _ _ lower
note_32 B4 _ _ upper
note_16 A4 _ _ upper
note_8 C5 _ _ upper&note_8 A4 _ _ upper&note_8 E3 _ _ lower&note_8 E2 _ _ lower
note_8 B4 _ _ upper&note_8 G4 ## _ upper&note_8 E3 _ _ lower&note_8 E2 _ _ lower
barline . . . .
note_2 A4 _ _ upper&note_2 A2 _ _ lower
repeatEnd . . . ."""


class TestCircleOfFifths(unittest.TestCase):
    def test_semantic_to_agnostic(self) -> None:
        tokens = read_token_lines(semantic.splitlines())
        result = semantic_to_agnostic_accidentals(tokens)
        actual = token_lines_to_str(result)
        self.assertEqual(agnostic, actual)

    @unittest.skip("needs to be defined after we have a working model")
    def test_agnostic_to_semantic(self) -> None:
        tokens = read_token_lines(agnostic.splitlines())
        result = agnostic_to_semantic_accidentals(tokens)
        actual = token_lines_to_str(result)
        self.assertEqual(semantic, actual)
