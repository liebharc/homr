import unittest
from typing import Any

from homr import constants
from homr.results import ResultChord, ResultDuration, ResultNote, ResultPitch
from homr.xml_generator import build_note_group

quarter = ResultDuration(constants.duration_of_quarter)
half = ResultDuration(2 * constants.duration_of_quarter)

a4 = ResultPitch("A", 4, None)
d4 = ResultPitch("D", 4, None)


class TestMusicXmlGen(unittest.TestCase):

    def test_chord(self) -> None:
        self.maxDiff = None
        chord = ResultChord(half, [ResultNote(a4, half), ResultNote(d4, half)])
        note = build_note_group(chord)
        expected = """
XMLNote([XMLPitch([XMLStep(value: A)]),XMLDuration(value: 32)]),
XMLNote([XMLChord(),XMLPitch([XMLStep(value: D)]),XMLDuration(value: 32)])
"""
        self.assertEqual(self._xml_to_str(note), self._norm_expected(expected))

    def test_chord_is_shorter_than_notes(self) -> None:
        self.maxDiff = None
        chord = ResultChord(quarter, [ResultNote(a4, half), ResultNote(d4, half)])
        note = build_note_group(chord)
        expected = """
XMLNote([XMLPitch([XMLStep(value: A)]),XMLDuration(value: 32)]),
XMLNote([XMLChord(),XMLPitch([XMLStep(value: D)]),XMLDuration(value: 32)]),
XMLBackup([XMLDuration(value: 16)])
"""
        self.assertEqual(self._xml_to_str(note), self._norm_expected(expected))

    def test_chord_is_longer_than_notes(self) -> None:
        self.maxDiff = None
        chord = ResultChord(half, [ResultNote(a4, quarter), ResultNote(d4, quarter)])
        note = build_note_group(chord)
        expected = """
XMLNote([XMLPitch([XMLStep(value: A)]),XMLDuration(value: 16)]),
XMLNote([XMLChord(),XMLPitch([XMLStep(value: D)]),XMLDuration(value: 16)]),
XMLForward([XMLDuration(value: 16)])
"""
        self.assertEqual(self._xml_to_str(note), self._norm_expected(expected))

    def test_chords_of_different_duration(self) -> None:
        self.maxDiff = None
        chord = ResultChord(quarter, [ResultNote(a4, quarter), ResultNote(d4, half)])
        note = build_note_group(chord)
        expected = """
XMLNote([XMLPitch([XMLStep(value: A)]),XMLDuration(value: 16)]),
XMLBackup([XMLDuration(value: 16)]),
XMLNote([XMLPitch([XMLStep(value: D)]),XMLDuration(value: 32)]),
XMLBackup([XMLDuration(value: 16)])
"""
        self.assertEqual(self._xml_to_str(note), self._norm_expected(expected))

    def _norm_expected(self, expected: str) -> str:
        return "[" + expected.replace("\n", "") + "]"

    def _xml_to_str(self, xml: Any) -> str:
        def recurse(node_or_list: Any) -> str:
            if isinstance(node_or_list, list):
                return (
                    "["
                    + ",".join(recurse(child) for child in node_or_list if child is not None)
                    + "]"
                )

            node = node_or_list
            name = node.__class__.__name__

            ignore_nodes = ("XMLAlter", "XMLOctave", "XMLVoice", "XMLType", "XMLStaff")

            if name in ignore_nodes:
                return ""
            value = getattr(node, "value_", None)

            if hasattr(node, "children"):
                children = node.children
            elif hasattr(node, "get_children"):
                children = node.get_children()
            else:
                children = []

            child_strs = [recurse(child) for child in children if child is not None]
            child_strs = [child for child in child_strs if child != ""]

            parts = []
            if value is not None and value != "":
                parts.append(f"value: {value}")
            if child_strs:
                parts.append(f"[{','.join(child_strs)}]")

            if parts:
                return f"{name}({','.join(parts)})"
            else:
                return f"{name}()"

        return recurse(xml)
