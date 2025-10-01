# ruff: noqa: E501

import re
import unittest
from typing import Any

from training.datasets.music_xml_parser import music_xml_string_to_tokens
from training.transformer.training_vocabulary import (
    token_lines_to_str,
)


class TestMusicXmlParser(unittest.TestCase):
    """
    MusicXML testing is mostly covered by training/validate_music_xml_conversion.py
    This script requires that the data sets are downloaded and converted and uses
    the data sets to check that back and forth conversion works.
    """

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
note_1 G4 _ tieStart_tieStop upper&note_1 A3 # _ upper&rest_2 _ _ _ upper&note_4 G3 _ slurStop lower
rest_4 _ _ _ lower
note_2 E4 _ tieStart upper&note_2 C2 _ _ lower
barline . . . ."""
        self.assertEqual(token_str, expected)

    def test_tuplet_end_in_chord(self) -> None:
        self.maxDiff = None
        # lc6258930.musicxml, measure 24
        example = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
  <measure number="24" width="271.27">
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
      <note default-x="79.52" default-y="-122.93">
        <pitch>
          <step>G</step>
          <octave>5</octave>
          </pitch>
        <duration>4</duration>
        <tie type="stop"/>
        <voice>1</voice>
        <type>eighth</type>
        <time-modification>
          <actual-notes>3</actual-notes>
          <normal-notes>2</normal-notes>
          </time-modification>
        <stem>down</stem>
        <staff>1</staff>
        <beam number="1">begin</beam>
        <notations>
          <tied type="stop"/>
          <tuplet type="start" bracket="yes" placement="above"/>
          <slur type="start" number="1"/>
          </notations>
        </note>
      <note default-x="79.52" default-y="-112.93">
        <chord/>
        <pitch>
          <step>B</step>
          <octave>5</octave>
          </pitch>
        <duration>4</duration>
        <tie type="stop"/>
        <voice>1</voice>
        <type>eighth</type>
        <time-modification>
          <actual-notes>3</actual-notes>
          <normal-notes>2</normal-notes>
          </time-modification>
        <stem>down</stem>
        <staff>1</staff>
        <notations>
          <tied type="stop"/>
          </notations>
        </note>
      <note default-x="117.51" default-y="-137.93">
        <pitch>
          <step>D</step>
          <alter>1</alter>
          <octave>5</octave>
          </pitch>
        <duration>4</duration>
        <voice>1</voice>
        <type>eighth</type>
        <accidental>sharp</accidental>
        <time-modification>
          <actual-notes>3</actual-notes>
          <normal-notes>2</normal-notes>
          </time-modification>
        <stem>down</stem>
        <staff>1</staff>
        <beam number="1">continue</beam>
        <notations>
          <articulations>
            <staccato placement="above"/>
            </articulations>
          </notations>
        </note>
      <note default-x="117.51" default-y="-122.93">
        <chord/>
        <pitch>
          <step>G</step>
          <octave>5</octave>
          </pitch>
        <duration>4</duration>
        <voice>1</voice>
        <type>eighth</type>
        <time-modification>
          <actual-notes>3</actual-notes>
          <normal-notes>2</normal-notes>
          </time-modification>
        <stem>down</stem>
        <staff>1</staff>
        </note>
      <note default-x="155.50" default-y="-147.93">
        <pitch>
          <step>B</step>
          <octave>4</octave>
          </pitch>
        <duration>4</duration>
        <voice>1</voice>
        <type>eighth</type>
        <time-modification>
          <actual-notes>3</actual-notes>
          <normal-notes>2</normal-notes>
          </time-modification>
        <stem>down</stem>
        <staff>1</staff>
        <beam number="1">end</beam>
        <notations>
          <tuplet type="stop"/>
          <articulations>
            <staccato placement="above"/>
            </articulations>
          </notations>
        </note>
      <note default-x="155.50" default-y="-137.93">
        <chord/>
        <pitch>
          <step>D</step>
          <alter>1</alter>
          <octave>5</octave>
          </pitch>
        <duration>4</duration>
        <voice>1</voice>
        <type>eighth</type>
        <time-modification>
          <actual-notes>3</actual-notes>
          <normal-notes>2</normal-notes>
          </time-modification>
        <stem>down</stem>
        <staff>1</staff>
        </note>
      <note default-x="193.49" default-y="-147.93">
        <pitch>
          <step>B</step>
          <octave>4</octave>
          </pitch>
        <duration>24</duration>
        <voice>1</voice>
        <type>half</type>
        <stem>up</stem>
        <staff>1</staff>
        <notations>
          <slur type="stop" number="1"/>
          <slur type="start" number="1"/>
          </notations>
        </note>
      <direction placement="below">
        <direction-type>
          <wedge type="stop" number="1"/>
          </direction-type>
        <staff>1</staff>
        </direction>
      <backup>
        <duration>36</duration>
        </backup>
      <forward>
        <duration>12</duration>
        </forward>
      <note default-x="193.49" default-y="-157.93">
        <pitch>
          <step>G</step>
          <octave>4</octave>
          </pitch>
        <duration>12</duration>
        <voice>2</voice>
        <type>quarter</type>
        <stem>down</stem>
        <staff>1</staff>
        </note>
      <note default-x="231.48" default-y="-162.93">
        <pitch>
          <step>F</step>
          <alter>1</alter>
          <octave>4</octave>
          </pitch>
        <duration>12</duration>
        <voice>2</voice>
        <type>quarter</type>
        <stem>down</stem>
        <staff>1</staff>
        <notations>
          <slur type="stop" number="1"/>
          </notations>
        </note>
      <note default-x="231.48" default-y="-152.93">
        <chord/>
        <pitch>
          <step>A</step>
          <octave>4</octave>
          </pitch>
        <duration>12</duration>
        <voice>2</voice>
        <type>quarter</type>
        <stem>down</stem>
        <staff>1</staff>
        </note>
      <backup>
        <duration>36</duration>
        </backup>
      <note default-x="79.52" default-y="-305.83">
        <pitch>
          <step>B</step>
          <octave>1</octave>
          </pitch>
        <duration>12</duration>
        <voice>5</voice>
        <type>quarter</type>
        <stem>up</stem>
        <staff>2</staff>
        </note>
      <note default-x="193.49" default-y="-260.83">
        <pitch>
          <step>D</step>
          <alter>1</alter>
          <octave>3</octave>
          </pitch>
        <duration>12</duration>
        <voice>5</voice>
        <type>quarter</type>
        <accidental>sharp</accidental>
        <stem>down</stem>
        <staff>2</staff>
        </note>
      <note default-x="193.49" default-y="-235.83">
        <chord/>
        <pitch>
          <step>B</step>
          <octave>3</octave>
          </pitch>
        <duration>12</duration>
        <voice>5</voice>
        <type>quarter</type>
        <stem>down</stem>
        <staff>2</staff>
        </note>
      <note default-x="231.48" default-y="-260.83">
        <pitch>
          <step>D</step>
          <alter>1</alter>
          <octave>3</octave>
          </pitch>
        <duration>12</duration>
        <voice>5</voice>
        <type>quarter</type>
        <stem>down</stem>
        <staff>2</staff>
        </note>
      <note default-x="231.48" default-y="-235.83">
        <chord/>
        <pitch>
          <step>B</step>
          <octave>3</octave>
          </pitch>
        <duration>12</duration>
        <voice>5</voice>
        <type>quarter</type>
        <stem>down</stem>
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
note_12 B5 _ slurStart_tieStop upper&note_12 G5 _ _ upper&note_4 B1 _ _ lower
note_12 G5 _ staccato upper&note_12 D5 # _ upper
note_12 D5 # staccato upper&note_12 B4 _ _ upper
note_2 B4 _ slurStart_slurStop upper&note_4 G4 _ _ upper&note_4 B3 _ _ lower&note_4 D3 # _ lower
note_4 A4 _ slurStop upper&note_4 F4 # _ upper&note_4 B3 _ _ lower&note_4 D3 # _ lower
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
