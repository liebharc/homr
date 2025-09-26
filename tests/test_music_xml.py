# ruff: noqa: E501

import re
import unittest
from typing import Any

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from training.datasets.music_xml_parser import music_xml_string_to_tokens
from training.transformer.training_vocabulary import (
    read_token_lines,
    token_lines_to_str,
)


class TestMusicXml(unittest.TestCase):
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
XMLTime([XMLBeats(value: 12),
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
XMLTime([XMLBeats(value: 16),
XMLBeatType(value: 4)]),
XMLClef([XMLSign(value: G),
XMLLine(value: 2)]),
XMLClef([XMLSign(value: F),
XMLLine(value: 4)])]),
XMLNote([XMLPitch([XMLStep(value: G)]),
XMLDuration(value: 1),
XMLVoice(value: 1),
XMLStaff(value: 2),
XMLNotations()]),
XMLBackup([XMLDuration(value: 1)]),
XMLNote([XMLRest(),
XMLDuration(value: 2),
XMLVoice(value: 2),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 2)]),
XMLNote([XMLPitch([XMLStep(value: G)]),
XMLDuration(value: 4),
XMLVoice(value: 3),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLChord(),
XMLPitch([XMLStep(value: A)]),
XMLDuration(value: 4),
XMLVoice(value: 3),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 3)]),
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
XMLNote([XMLChord(),
XMLPitch([XMLStep(value: C)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 2),
XMLNotations()])])])])"""
        self.assertEqual(self._norm_expected(expected), actual)

    def test_parse_xml_with_backup(self) -> None:
        self.maxDiff = None
        # lc4987672.musicxml, measure 22
        example = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
      <measure number="22">
    <attributes>
      <divisions>4</divisions>
      <key>
        <fifths>1</fifths>
        </key>
      <time symbol="common">
        <beats>4</beats>
        <beat-type>4</beat-type>
        </time>
      <staves>2</staves>
      <clef number="1">
        <sign>G</sign>
        <line>2</line>
        </clef>
      <clef number="2">
        <sign>F</sign>
        <line>4</line>
        </clef>
      </attributes>
    <note>
      <pitch>
        <step>A</step>
        <alter>1</alter>
        <octave>3</octave>
        </pitch>
      <duration>16</duration>
      <tie type="stop"/>
      <voice>1</voice>
      <type>whole</type>
      <staff>1</staff>
      <notations>
        <tied type="stop"/>
        </notations>
      </note>
    <note>
      <chord/>
      <pitch>
        <step>G</step>
        <octave>4</octave>
        </pitch>
      <duration>16</duration>
      <tie type="stop"/>
      <tie type="start"/>
      <voice>1</voice>
      <type>whole</type>
      <staff>1</staff>
      <notations>
        <tied type="stop"/>
        <tied type="start"/>
        </notations>
      </note>
    <backup>
      <duration>16</duration>
      </backup>
    <note>
      <rest/>
      <duration>8</duration>
      <voice>2</voice>
      <type>half</type>
      <staff>1</staff>
      </note>
    <direction placement="below">
      <direction-type>
        <dynamics default-x="2.78" default-y="-40.00" relative-x="3.29" relative-y="-50.00">
          <pp/>
          </dynamics>
        </direction-type>
      <staff>1</staff>
      <sound dynamics="36.67"/>
      </direction>
    <note>
      <pitch>
        <step>E</step>
        <octave>4</octave>
        </pitch>
      <duration>8</duration>
      <tie type="start"/>
      <voice>2</voice>
      <type>half</type>
      <stem>down</stem>
      <staff>1</staff>
      <notations>
        <tied type="start"/>
        </notations>
      </note>
    <backup>
      <duration>16</duration>
      </backup>
    <note>
      <pitch>
        <step>G</step>
        <octave>3</octave>
        </pitch>
      <duration>4</duration>
      <voice>5</voice>
      <type>quarter</type>
      <stem>down</stem>
      <staff>2</staff>
      <notations>
        <slur type="stop" number="1"/>
        </notations>
      </note>
    <note default-x="55.02" default-y="-298.07">
      <rest/>
      <duration>4</duration>
      <voice>5</voice>
      <type>quarter</type>
      <staff>2</staff>
      </note>
    <note default-x="139.48" default-y="-338.07">
      <pitch>
        <step>C</step>
        <octave>2</octave>
        </pitch>
      <duration>8</duration>
      <voice>5</voice>
      <type>half</type>
      <stem>up</stem>
      <staff>2</staff>
      </note>
    </measure>
  </part>
</score-partwise>
      """
        tokens = music_xml_string_to_tokens(example)
        flat_list = [x for xxs in tokens for xs in xxs for x in xs]
        token_str = token_lines_to_str(flat_list)
        expected = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_1 . . . .
timeSignature/4 . . . .
note_1 G4 _ tieStart_tieStop upper&note_1 A3 # tieStop upper&rest_2 _ _ _ upper&note_4 G3 _ slurStop lower
rest_4 _ _ _ lower
note_2 E4 _ tieStart upper&note_2 C2 _ _ lower
barline . . . ."""
        self.assertEqual(token_str, expected)

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
