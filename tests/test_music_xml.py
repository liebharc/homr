import unittest

from training.datasets.music_xml_parser import music_xml_string_to_tokens
from training.transformer.training_vocabulary import token_lines_to_str


class TestMusicXml(unittest.TestCase):
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
        expected = """clef_G2 . . .
keySignature_1 . . .
timeSignature/4 . . .
note_1 A3 # _&note_1 G4 _ _&rest_2 _ _ _
note_2 E4 _ _
barline . . .
clef_F4 . . .
keySignature_1 . . .
timeSignature/4 . . .
note_4 G3 _ _
rest_4 _ _ _
note_2 C2 _ _
barline . . ."""
        self.assertEqual(token_str, expected)
