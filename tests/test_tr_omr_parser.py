import unittest

from homr import constants
from homr.results import (
    ClefType,
    DurationModifier,
    ResultChord,
    ResultClef,
    ResultDuration,
    ResultMeasure,
    ResultNote,
    ResultPitch,
    ResultStaff,
    ResultTimeSignature,
)
from homr.tr_omr_parser import TrOMRParser


def single_note(pitch: ResultPitch, duration: ResultDuration) -> ResultChord:

    return ResultChord(
        duration,
        [
            ResultNote(
                pitch,
                duration,
            )
        ],
    )


def note_chord(notes: list[ResultNote]) -> ResultChord:
    return ResultChord(notes[0].duration, notes)


class TestTrOmrParser(unittest.TestCase):

    unit_size = 3

    def test_parsing(self) -> None:
        data = "clef-G2+keySignature-FM+timeSignature-/4+note-A4_half+note-B4_half+barline+note-A4_quarter.+note-G4_eighth+note-F4_quarter+note-G4_quarter+barline"  # noqa: E501
        expected = ResultStaff(
            [
                ResultMeasure(
                    [
                        ResultClef(ClefType.treble(), -1),
                        ResultTimeSignature(1, 4),
                        single_note(
                            ResultPitch("A", 4, None),
                            ResultDuration(2 * constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("B", 4, None),
                            ResultDuration(2 * constants.duration_of_quarter),
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        single_note(
                            ResultPitch("A", 4, None),
                            ResultDuration(
                                int(constants.duration_of_quarter), DurationModifier.DOT
                            ),
                        ),
                        single_note(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter // 2),
                        ),
                        single_note(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                    ]
                ),
            ]
        )

        parser = TrOMRParser()
        actual = parser.parse_tr_omr_output(data)
        self.assertEqual(actual, expected)

    def test_parsing_no_final_bar_line(self) -> None:
        data = "clef-G2+keySignature-FM+timeSignature-/4+note-A4_half+note-B4_half+barline+note-A4_quarter.+note-G4_eighth+note-F4_quarter+note-G4_quarter"  # noqa: E501
        expected = ResultStaff(
            [
                ResultMeasure(
                    [
                        ResultClef(ClefType.treble(), -1),
                        ResultTimeSignature(1, 4),
                        single_note(
                            ResultPitch("A", 4, None),
                            ResultDuration(2 * constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("B", 4, None),
                            ResultDuration(2 * constants.duration_of_quarter),
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        single_note(
                            ResultPitch("A", 4, None),
                            ResultDuration(constants.duration_of_quarter, DurationModifier.DOT),
                        ),
                        single_note(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter // 2),
                        ),
                        single_note(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter),
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
                        single_note(
                            ResultPitch("E", 5, None),
                            ResultDuration(constants.duration_of_quarter // 4),
                        ),
                        note_chord(
                            [
                                ResultNote(
                                    ResultPitch("A", 2, None),
                                    ResultDuration(constants.duration_of_quarter // 2),
                                ),
                                ResultNote(
                                    ResultPitch("E", 4, None),
                                    ResultDuration(constants.duration_of_quarter // 2),
                                ),
                            ]
                        ),
                        ResultChord(ResultDuration(constants.duration_of_quarter // 2), []),
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
                        ResultClef(ClefType.treble(), 0),
                        single_note(
                            ResultPitch("D", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("E", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        single_note(
                            ResultPitch("D", 4, None),
                            ResultDuration(
                                constants.duration_of_quarter * 2,
                            ),
                        ),
                        note_chord(
                            [
                                ResultNote(
                                    ResultPitch("D", 4, None),
                                    ResultDuration(constants.duration_of_quarter * 2),
                                ),
                                ResultNote(
                                    ResultPitch("G", 4, None),
                                    ResultDuration(constants.duration_of_quarter * 2),
                                ),
                            ]
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        single_note(
                            ResultPitch("E", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("A", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        single_note(
                            ResultPitch("E", 4, None),
                            ResultDuration(constants.duration_of_quarter * 2),
                        ),
                        note_chord(
                            [
                                ResultNote(
                                    ResultPitch("E", 4, None),
                                    ResultDuration(constants.duration_of_quarter * 2),
                                ),
                                ResultNote(
                                    ResultPitch("A", 4, None),
                                    ResultDuration(constants.duration_of_quarter * 2),
                                ),
                            ]
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        single_note(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("A", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("B", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                    ]
                ),
                ResultMeasure(
                    [
                        single_note(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter * 2),
                        ),
                        note_chord(
                            [
                                ResultNote(
                                    ResultPitch("F", 4, None),
                                    ResultDuration(constants.duration_of_quarter * 2),
                                ),
                                ResultNote(
                                    ResultPitch("B", 4, None),
                                    ResultDuration(constants.duration_of_quarter * 2),
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
                        ResultClef(ClefType.treble(), 2),
                        single_note(
                            ResultPitch("D", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("E", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("F", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("G", 4, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("C", 5, None),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("C", 5, 1),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                        single_note(
                            ResultPitch("A", 4, 1),
                            ResultDuration(constants.duration_of_quarter),
                        ),
                    ]
                ),
            ]
        )

        parser = TrOMRParser()
        actual = parser.parse_tr_omr_output(data)
        self.assertEqual(actual, expected)

    def test_parsing_chords_with_rests(self) -> None:
        data = "rest_quarter|note-A4_half|note-B4_half"
        expected = ResultStaff(
            [
                ResultMeasure(
                    [
                        note_chord(
                            [
                                ResultNote(
                                    ResultPitch("A", 4, None),
                                    ResultDuration(2 * constants.duration_of_quarter),
                                ),
                                ResultNote(
                                    ResultPitch("B", 4, None),
                                    ResultDuration(2 * constants.duration_of_quarter),
                                ),
                            ]
                        )
                    ]
                )
            ]
        )
        parser = TrOMRParser()
        actual = parser.parse_tr_omr_output(data)
        self.assertEqual(actual, expected)

    def test_parse_chords_with_unexpected_symbols(self) -> None:
        data = "note-A4_half|barline"
        expected = ResultStaff(
            [
                ResultMeasure(
                    [
                        single_note(
                            ResultPitch("A", 4, None),
                            ResultDuration(2 * constants.duration_of_quarter),
                        ),
                    ]
                )
            ]
        )
        parser = TrOMRParser()
        actual = parser.parse_tr_omr_output(data)
        self.assertEqual(actual, expected)
