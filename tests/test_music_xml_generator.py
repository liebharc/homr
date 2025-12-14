# ruff: noqa: E501

import re
import unittest
from typing import Any

from homr.music_xml_generator import SymbolChord, XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol
from training.transformer.training_vocabulary import (
    read_token_lines,
)


class TestMusicXmlGenerator(unittest.TestCase):
    """
    MusicXML testing is mostly covered by training/validate_music_xml_conversion.py
    This script requires that the data sets are downloaded and converted and uses
    the data sets to check that back and forth conversion works.
    """

    def test_chord_with_different_duratons(self) -> None:
        tabi_measure_18_upper = """clef_G2 . . . upper
keySignature_4 . . . .
timeSignature/8 . . . .
note_4. G3 # _ upper &note_4. C4 # _ upper&note_16 E4 # _ upper
note_16 F4 # _ upper
note_4 E4 # _ upper
note_8 E4 # _ upper
note_8 C4 # _ upper
note_8 D4 # _ upper
barline . . . ."""
        tokens = read_token_lines(tabi_measure_18_upper.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        actual = self._xml_to_str(xml)
        expected = """XMLScorePartwise([XMLWork([XMLWorkTitle()]),
XMLPart([XMLMeasure([XMLAttributes([XMLDivisions(value: 4)]),
XMLAttributes([XMLKey([XMLFifths(value: 4)]),
XMLTime([XMLBeats(value: 6),
XMLBeatType(value: 8)]),
XMLClef([XMLSign(value: G),
XMLLine(value: 2)])]),
XMLNote([XMLPitch([XMLStep(value: E)]),
XMLDuration(value: 1),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 1)]),
XMLNote([XMLPitch([XMLStep(value: G)]),
XMLDuration(value: 6),
XMLVoice(value: 2),
XMLDot(),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLChord(),
XMLPitch([XMLStep(value: C)]),
XMLDuration(value: 6),
XMLVoice(value: 2),
XMLDot(),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 5)]),
XMLNote([XMLPitch([XMLStep(value: F)]),
XMLDuration(value: 1),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: E)]),
XMLDuration(value: 4),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: E)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: C)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: D)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()])])])])"""
        self.assertEqual(self._norm_expected(expected), actual)

    def test_grand_staff_generation(self) -> None:
        grandstaff = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_1 . . . .
timeSignature/4 . . . .
note_1 G4 _ _ upper&note_1 A3 # _ upper&rest_2 _ _ _ upper&note_4 G3 _ _ lower
rest_4 _ _ _ lower
note_2 E4 _ _ upper&note_2 C2 _ _ lower
barline . . . ."""
        tokens = read_token_lines(grandstaff.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        actual = self._xml_to_str(xml)
        expected = """XMLScorePartwise([XMLWork([XMLWorkTitle()]),
XMLPart([XMLMeasure([XMLAttributes([XMLDivisions(value: 1)]),
XMLAttributes([XMLKey([XMLFifths(value: 1)]),
XMLTime([XMLBeats(value: 4),
XMLBeatType(value: 4)]),
XMLClef([XMLSign(value: G),
XMLLine(value: 2)]),
XMLClef([XMLSign(value: F),
XMLLine(value: 4)])]),
XMLNote([XMLRest(),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 2)]),
XMLNote([XMLPitch([XMLStep(value: G)]),
XMLDuration(value: 4),
XMLVoice(value: 2),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLChord(),
XMLPitch([XMLStep(value: A)]),
XMLDuration(value: 4),
XMLVoice(value: 2),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 4)]),
XMLNote([XMLPitch([XMLStep(value: G)]),
XMLDuration(value: 1),
XMLVoice(value: 1),
XMLStaff(value: 2),
XMLNotations()]),
XMLNote([XMLRest(),
XMLDuration(value: 1),
XMLVoice(value: 1),
XMLStaff(value: 2),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: E)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 2)]),
XMLNote([XMLPitch([XMLStep(value: C)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 2),
XMLNotations()])])])])"""
        self.assertEqual(self._norm_expected(expected), actual)

    def test_strip_articulations(self) -> None:
        chord = SymbolChord(
            [
                EncodedSymbol("note_8", articulation="staccatissimo"),
                EncodedSymbol("note_16", articulation="tieStart_slurStop_tenuto"),
                EncodedSymbol("note_32", articulation="tieStart"),
            ]
        )

        articulations, result = chord.strip_slur_ties()
        self.assertEqual(
            result.symbols,
            (
                [
                    EncodedSymbol("note_8", articulation="staccatissimo"),
                    EncodedSymbol("note_16", articulation="tenuto"),
                    EncodedSymbol("note_32", articulation="_"),
                ]
            ),
        )
        self.assertEqual(articulations, ["slurStop", "tieStart"])

    def test_begin_chord_with_standalone_rests(self) -> None:
        """
        If the lower position consists of a standalone rest then start the
        chord with this. That fixes an issue where the upper position
        consists of tuplets because in that case backups must not be used.

        See tabi.jpg measure 9 for an example.
        """
        chord = SymbolChord(
            [
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("rest_8", position="lower"),
            ]
        )
        first, second = chord.into_positions()

        self.assertEqual(first.symbols, [EncodedSymbol("rest_8", position="lower")])
        self.assertEqual(
            second.symbols,
            [
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("note_12", position="upper"),
            ],
        )

    def _norm_expected(self, expected: str) -> str:
        norm = expected.replace("\n", "")
        norm = re.sub(r",\s+", ",", norm)
        norm = re.sub(r"\[\s+", "[", norm)
        return norm

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

            ignore_nodes = (
                "XMLAlter",
                "XMLOctave",
                "XMLType",
                "XMLPartList",
                "XMLDefaults",
            )

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
