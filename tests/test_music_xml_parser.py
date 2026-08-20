# ruff: noqa: E501

import re
import unittest
import xml.etree.ElementTree as ET
from typing import Any

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.transformer.training_vocabulary import (
    read_token_lines,
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
        expected = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_1 . . . . .
timeSignature/4 . . . . .
note_1 G4 _ _ slurStart_slurStop upper&note_1 A3 # _ _ upper&rest_2 _ _ _ _ upper2&note_4 G3 _ _ slurStop lower
rest_4 _ _ _ _ lower
note_2 E4 _ _ slurStart upper2&note_2 C2 _ _ _ lower
barline . . . . ."""
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
        expected = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_1 . . . . .
timeSignature/4 . . . . .
note_12 B5 _ _ slurStart_slurStop upper&note_12 G5 _ _ _ upper&note_4 B1 _ _ _ lower
note_12 G5 _ staccato _ upper&note_12 D5 # _ _ upper
note_12 D5 # staccato _ upper&note_12 B4 _ _ _ upper
note_2 B4 _ _ slurStart_slurStop upper&note_4 G4 _ _ _ upper2&note_4 B3 _ _ _ lower&note_4 D3 # _ _ lower
note_4 A4 _ _ slurStop upper2&note_4 F4 # _ _ upper2&note_4 B3 _ _ _ lower&note_4 D3 # _ _ lower
barline . . . . ."""
        self.assertEqual(token_str, expected)

    def test_arpeggiate(self) -> None:
        """Arpeggio on one staff shouldn't be propagated to the other
        one that starts on the same beat. It still should be filled in across
        the chord notes on its own staff."""
        self.maxDiff = None
        # Upper-staff chord (no arpeggio) and lower-staff chord (arpeggio on
        # one note only) sharing beat 0. like lc6810938.musicxml.
        example = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <note>
        <pitch><step>D</step><octave>4</octave></pitch>
        <duration>16</duration><voice>1</voice><type>whole</type><staff>1</staff>
      </note>
      <note>
        <chord/>
        <pitch><step>D</step><octave>5</octave></pitch>
        <duration>16</duration><voice>1</voice><type>whole</type><staff>1</staff>
      </note>
      <backup><duration>16</duration></backup>
      <note>
        <pitch><step>G</step><octave>2</octave></pitch>
        <duration>16</duration><voice>5</voice><type>whole</type><staff>2</staff>
        <notations><arpeggiate number="1"/></notations>
      </note>
      <note>
        <chord/>
        <pitch><step>D</step><octave>3</octave></pitch>
        <duration>16</duration><voice>5</voice><type>whole</type><staff>2</staff>
      </note>
      <note>
        <chord/>
        <pitch><step>B</step><octave>3</octave></pitch>
        <duration>16</duration><voice>5</voice><type>whole</type><staff>2</staff>
      </note>
    </measure>
  </part>
</score-partwise>
"""
        tokens = music_xml_string_to_tokens(example)
        flat_list = [x for xxs in tokens for xs in xxs for x in xs]
        token_str = token_lines_to_str(flat_list)
        expected = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_1 D5 _ _ _ upper&note_1 D4 _ _ _ upper&note_1 B3 _ arpeggiate _ lower&note_1 D3 _ _ _ lower&note_1 G2 _ _ _ lower
barline . . . . ."""
        self.assertEqual(token_str, expected)

    def test_two_voices_on_one_staff(self) -> None:
        """The first visible voice of a staff is the main voice, the others get a "2"
        suffix. Here voice 2 is written before voice 1 and voice 3 is folded into
        the second voice."""
        self.maxDiff = None
        example = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef number="1"><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>8</duration><voice>2</voice><type>half</type><staff>1</staff>
      </note>
      <note>
        <pitch><step>D</step><octave>4</octave></pitch>
        <duration>8</duration><voice>2</voice><type>half</type><staff>1</staff>
      </note>
      <backup><duration>16</duration></backup>
      <note>
        <pitch><step>G</step><octave>4</octave></pitch>
        <duration>16</duration><voice>1</voice><type>whole</type><staff>1</staff>
      </note>
      <backup><duration>16</duration></backup>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>16</duration><voice>3</voice><type>whole</type><staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>
"""
        tokens = music_xml_string_to_tokens(example)
        flat_list = [x for xxs in tokens for xs in xxs for x in xs]
        token_str = token_lines_to_str(flat_list)
        expected = """clef_G2 _ _ _ _ upper
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_1 G4 _ _ _ upper2&note_2 E4 _ _ _ upper&note_1 C4 _ _ _ upper2
note_2 D4 _ _ _ upper
barline . . . . ."""
        self.assertEqual(token_str, expected)

    def test_voice_assignment_is_per_staff_and_measure(self) -> None:
        """Hidden rests do not claim a voice; each staff and measure starts fresh."""
        example = """<score-partwise version="4.0"><part id="P1">
<measure number="1">
  <attributes>
    <divisions>1</divisions>
    <clef number="1"><sign>G</sign><line>2</line></clef>
    <clef number="2"><sign>F</sign><line>4</line></clef>
  </attributes>
  <note print-object="no">
    <rest/><duration>1</duration><voice>1</voice><type>quarter</type><staff>1</staff>
  </note>
  <backup><duration>1</duration></backup>
  <note>
    <rest/><duration>1</duration><voice>2</voice><type>quarter</type><staff>1</staff>
  </note>
  <backup><duration>1</duration></backup>
  <note>
    <pitch><step>C</step><octave>4</octave></pitch>
    <duration>1</duration><voice>1</voice><type>quarter</type><staff>1</staff>
  </note>
  <note>
    <chord/><pitch><step>E</step><octave>4</octave></pitch>
    <duration>1</duration><voice>1</voice><type>quarter</type><staff>1</staff>
  </note>
  <backup><duration>1</duration></backup>
  <note>
    <pitch><step>C</step><octave>3</octave></pitch>
    <duration>1</duration><voice>1</voice><type>quarter</type><staff>2</staff>
  </note>
  <backup><duration>1</duration></backup>
  <note>
    <rest/><duration>1</duration><voice>2</voice><type>quarter</type><staff>2</staff>
  </note>
</measure>
<measure number="2">
  <note>
    <pitch><step>D</step><octave>4</octave></pitch>
    <duration>1</duration><voice>1</voice><type>quarter</type><staff>1</staff>
  </note>
</measure>
</part></score-partwise>"""
        tokens = music_xml_string_to_tokens(example)
        symbols = [s for page in tokens for measure in page for s in measure]
        notes = {s.pitch: s.position for s in symbols if s.rhythm.startswith("note")}
        self.assertEqual(notes, {"C4": "upper2", "E4": "upper2", "C3": "lower", "D4": "upper"})
        rests = [s.position for s in symbols if s.rhythm.startswith("rest")]
        self.assertEqual(rests, ["upper", "lower2"])

    def test_round_trip_of_two_voices_on_one_staff(self) -> None:
        """tokens -> MusicXML -> tokens keeps the two upper voices apart.

        Which one ends up as `upper` and which as `upper2` may swap: the generator
        numbers the MusicXML voices by rhythmic layer (rebalance_measure_voices),
        while we take the first visible voice of a staff as the main voice.
        """
        self.maxDiff = None
        # Same input as test_music_xml_generator.test_two_voices_on_the_same_staff
        original = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 G4 _ _ _ upper&note_4 E4 _ _ _ upper2&note_1 C3 _ _ _ lower
note_4 D4 _ _ _ upper2
barline . . . . ."""
        xml = generate_xml(XmlGeneratorArguments(), [read_token_lines(original.splitlines())], "")
        tokens = music_xml_string_to_tokens(ET.tostring(xml, encoding="unicode"))
        flat_list = [x for xxs in tokens for xs in xxs for x in xs]
        positions = {s.pitch: s.position for s in flat_list if s.rhythm.startswith("note")}
        self.assertEqual(positions["C3"], "lower")
        self.assertEqual(positions["E4"], positions["D4"])
        self.assertEqual({positions["G4"], positions["E4"]}, {"upper", "upper2"})

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
