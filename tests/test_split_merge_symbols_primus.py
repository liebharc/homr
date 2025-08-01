import unittest

from homr.transformer.split_merge_symbols import merge_symbols, split_symbols


def split_merge(merged: str) -> list[str]:
    actuallift, actualpitch, actualrhythm, actualmodfier, _actualnotes = split_symbols(
        [merged.replace("+", "\t")]
    )
    merged_again = merge_symbols(actualrhythm, actualpitch, actuallift, actualmodfier)
    return merged_again


class TestMergeSymbolsPrimus(unittest.TestCase):

    def test_merge(self) -> None:
        actual = split_merge(
            "clef-C1 timeSignature-C/ note-G4_double_whole note-G4_whole note-A4_whole. note-G4_half note-G4_half note-F#4_half note-G4_double_whole note-G4_half"  # noqa: E501
        )
        self.assertEqual(
            actual,
            [
                "clef-C1+timeSignature-/2+note-G4_breve+note-G4_whole+note-A4_whole.+note-G4_half+note-G4_half+note-F4#_half+note-G4_breve+note-G4_half"
            ],
        )

    def test_multirest_gets_a_valid_modifier(self) -> None:
        actual = "clef-G2 keySignature-GM timeSignature-3/8       multirest-52    barline note-G4_quarter.        barline note-B4_quarter.        barline note-D5_quarter gracenote-C5_eighth     note-B4_sixteenth  note-A4_thirty_second   note-G4_thirty_second   barline note-E5_eighth  note-F#5_eighth note-G5_eighth  barline gracenote-D5_eighth     note-C5_quarter.        barline    note-B4_quarter rest-eighth     barline note-G4_sixteenth       note-A4_sixteenth       note-B4_eighth  note-A4_eighth  barline note-A4_quarter"  # noqa: E501

        _actuallift, _actualpitch, _actualrhythm, actualmodfier, _actualnotes = split_symbols(
            [actual.replace("+", "\t")]
        )

        valid_symbols = ("mod_null", "nonote", "mod_dot")
        invalid_modifiers = [s for s in actualmodfier[0] if s not in valid_symbols]
        self.assertEqual(invalid_modifiers, [])
