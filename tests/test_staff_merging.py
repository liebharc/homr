import unittest

from homr.transformer.vocabulary import EncodedSymbol, empty
from training.datasets.staff_merging import create_chord_over_two_staffs


class TestStaffMerging(unittest.TestCase):
    def test_chord_merging(self) -> None:

        result = create_chord_over_two_staffs(
            [
                EncodedSymbol("keySignature_0", empty, empty, empty, "upper"),
                EncodedSymbol("timeSignature/4"),
                EncodedSymbol("clef_G2", empty, empty, empty, "upper"),
                EncodedSymbol("note_16", "F4", empty, empty, "upper"),
                EncodedSymbol("clef_F4", empty, empty, empty, "lower"),
                EncodedSymbol("rest_4", empty, empty, empty, "upper"),
                EncodedSymbol("repeatStart"),
            ]
        )

        self.assertEqual(
            [r.rhythm for r in result],
            [
                "repeatStart",
                "clef_G2",
                "chord",
                "clef_F4",
                "keySignature_0",
                "timeSignature/4",
                "note_16",
                "chord",
                "rest_4",
            ],
        )
