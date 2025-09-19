import unittest

from training.datasets.primus_semantic_parser import convert_primus_semantic_to_tokens
from training.transformer.training_vocabulary import (
    check_token_lines,
    token_lines_to_str,
)

example = "clef-G2	keySignature-FM	timeSignature-3/8	note-F4_quarter.	barline	note-A5_quarter.	barline	gracenote-A5_eighth	note-G5_quarter.	barline	note-Bb5_eighth	note-A5_eighth	note-G5_eighth	barline	note-F5_quarter.	tie	barline	note-F5_quarter.	barline	note-A4_quarter.	barline"  # noqa: E501


class TestPrimusSemanticToTokens(unittest.TestCase):
    def test_conversion(self) -> None:
        result = convert_primus_semantic_to_tokens(example)
        check_token_lines(result)
        as_string = token_lines_to_str(result)
        expected = """clef_G2 _ _ _ upper
keySignature_-1 . . . .
timeSignature/8 . . . .
note_4. F4 _ _ upper
barline . . . .
note_4. A5 _ _ upper
barline . . . .
note_8G A5 _ _ upper
note_4. G5 _ _ upper
barline . . . .
note_8 B5 b _ upper
note_8 A5 _ _ upper
note_8 G5 _ _ upper
barline . . . .
note_4. F5 _ _ upper
tieSlur . . . .
barline . . . .
note_4. F5 _ _ upper
barline . . . .
note_4. A4 _ _ upper
barline . . . ."""
        self.assertEqual(as_string, expected)
