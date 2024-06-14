import unittest

from homr import constants
from homr.results import (
    ClefType,
    ResultClef,
    ResultDuration,
    ResultMeasure,
    ResultNote,
    ResultNoteGroup,
    ResultPitch,
    ResultRest,
    ResultStaff,
    ResultTimeSignature,
)
from homr.tr_omr_parser import TrOMRParser


class TestTrOmrParser(unittest.TestCase):

    unit_size = 3

    def test_parsing(self) -> None:
        data = "clef-G2+keySignature-FM+timeSignature-4/4+note-A4_half+note-B4_half+barline+note-A4_quarter.+note-G4_eighth+note-F4_quarter+note-G4_quarter+barline"  # noqa: E501
        expected = ResultStaff(
            [
                ResultMeasure(
                    [
                        ResultClef(ClefType.TREBLE, -1),
                        ResultTimeSignature("4/4"),
                        ResultNote(
                            ResultPitch("A", 4, None),
                            ResultDuration(2 * constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("B", 4, None),
                            ResultDuration(2 * constants.duration_of_quarter, False),
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        ResultNote(
                            ResultPitch("A", 4, None),
                            ResultDuration(int(constants.duration_of_quarter * 1.5), True),
                        ),
                        ResultNote(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter // 2, False),
                        ),
                        ResultNote(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                    ]
                ),
            ]
        )

        parser = TrOMRParser()
        actual = parser.parse_tr_omr_output(data)
        self.assertEqual(actual, expected)

    def test_parsing_no_final_bar_line(self) -> None:
        data = "clef-G2+keySignature-FM+timeSignature-4/4+note-A4_half+note-B4_half+barline+note-A4_quarter.+note-G4_eighth+note-F4_quarter+note-G4_quarter"  # noqa: E501
        expected = ResultStaff(
            [
                ResultMeasure(
                    [
                        ResultClef(ClefType.TREBLE, -1),
                        ResultTimeSignature("4/4"),
                        ResultNote(
                            ResultPitch("A", 4, None),
                            ResultDuration(2 * constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("B", 4, None),
                            ResultDuration(2 * constants.duration_of_quarter, False),
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        ResultNote(
                            ResultPitch("A", 4, None),
                            ResultDuration(int(constants.duration_of_quarter * 1.5), True),
                        ),
                        ResultNote(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter // 2, False),
                        ),
                        ResultNote(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                    ]
                ),
            ]
        )

        parser = TrOMRParser()
        actual = parser.parse_tr_omr_output(data)
        self.assertEqual(actual, expected)

    def test_rest_parsing(self) -> None:
        data = "note-E5_sixteenth|rest-eighth+note-A2_eighth|note-E4_eighth+rest-eighth"
        expected = ResultStaff(
            [
                ResultMeasure(
                    [
                        ResultNote(
                            ResultPitch("E", 5, None),
                            ResultDuration(constants.duration_of_quarter // 4, False),
                        ),
                        ResultNoteGroup(
                            [
                                ResultNote(
                                    ResultPitch("A", 2, None),
                                    ResultDuration(constants.duration_of_quarter // 2, False),
                                ),
                                ResultNote(
                                    ResultPitch("E", 4, None),
                                    ResultDuration(constants.duration_of_quarter // 2, False),
                                ),
                            ]
                        ),
                        ResultRest(ResultDuration(constants.duration_of_quarter // 2, False)),
                    ]
                ),
            ]
        )

        parser = TrOMRParser()
        actual = parser.parse_tr_omr_output(data)
        self.assertEqual(actual, expected)

    def test_note_group_parsing(self) -> None:
        data = "clef-G2+keySignature-CM+note-D4_quarter+note-E4_quarter+note-F4_quarter+note-G4_quarter+barline+note-D4_half+note-D4_half|note-G4_half+barline+note-E4_quarter+note-F4_quarter+note-G4_quarter+note-A4_quarter+barline+note-E4_half+note-E4_half|note-A4_half+barline+note-F4_quarter+note-G4_quarter+note-A4_quarter+note-B4_quarter+barline+note-F4_half+note-F4_half|note-B4_half+barline"  # noqa: E501
        expected = ResultStaff(
            [
                ResultMeasure(
                    [
                        ResultClef(ClefType.TREBLE, 0),
                        ResultNote(
                            ResultPitch("D", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("E", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        ResultNote(
                            ResultPitch("D", 4, None),
                            ResultDuration(constants.duration_of_quarter * 2, False),
                        ),
                        ResultNoteGroup(
                            [
                                ResultNote(
                                    ResultPitch("D", 4, None),
                                    ResultDuration(constants.duration_of_quarter * 2, False),
                                ),
                                ResultNote(
                                    ResultPitch("G", 4, None),
                                    ResultDuration(constants.duration_of_quarter * 2, False),
                                ),
                            ]
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        ResultNote(
                            ResultPitch("E", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("A", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        ResultNote(
                            ResultPitch("E", 4, None),
                            ResultDuration(constants.duration_of_quarter * 2, False),
                        ),
                        ResultNoteGroup(
                            [
                                ResultNote(
                                    ResultPitch("E", 4, None),
                                    ResultDuration(constants.duration_of_quarter * 2, False),
                                ),
                                ResultNote(
                                    ResultPitch("A", 4, None),
                                    ResultDuration(constants.duration_of_quarter * 2, False),
                                ),
                            ]
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        ResultNote(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("A", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("B", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        ResultNote(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter * 2, False),
                        ),
                        ResultNoteGroup(
                            [
                                ResultNote(
                                    ResultPitch("F", 4, None),
                                    ResultDuration(constants.duration_of_quarter * 2, False),
                                ),
                                ResultNote(
                                    ResultPitch("B", 4, None),
                                    ResultDuration(constants.duration_of_quarter * 2, False),
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

        parser = TrOMRParser()
        actual = parser.parse_tr_omr_output(data)
        self.assertEqual(actual, expected)

    def test_accidental_parsing(self) -> None:
        data = "clef-G2+keySignature-DM+note-D4_quarter+note-E4_quarter+note-F4_quarter+note-G4_quarter+note-C5_quarter+note-C5#_quarter+note-A4#_quarter+barline"  # noqa: E501
        expected = ResultStaff(
            [
                ResultMeasure(
                    [
                        ResultClef(ClefType.TREBLE, 2),
                        ResultNote(
                            ResultPitch("D", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("E", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("C", 5, None),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("C", 5, 1),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                        ResultNote(
                            ResultPitch("A", 4, 1),
                            ResultDuration(constants.duration_of_quarter, False),
                        ),
                    ]
                ),
            ]
        )

        parser = TrOMRParser()
        actual = parser.parse_tr_omr_output(data)
        self.assertEqual(actual, expected)
