import unittest

from homr.transformer.split_merge_symbols import merge_symbols, split_symbols


def split_merge(merged: str) -> list[str]:
    actuallift, actualpitch, actualrhythm, _actualnotes = split_symbols([merged.replace("+", "\t")])
    merged_again = merge_symbols(actualrhythm, actualpitch, actuallift)
    return merged_again


class TestMergeSymbolsPrimus(unittest.TestCase):

    def test_merge(self) -> None:
        actual = split_merge(
            "clef-C1 timeSignature-C/ note-G4_double_whole note-G4_whole note-A4_whole. note-G4_half note-G4_half note-F#4_half note-G4_double_whole note-G4_half"  # noqa: E501
        )
        self.assertEqual(
            actual,
            [
                "clef-C1+timeSignature-C/+note-G4_breve+note-G4_whole+note-A4_whole.+note-G4_half+note-G4_half+note-F4#_half+note-G4_breve+note-G4_half"
            ],
        )
