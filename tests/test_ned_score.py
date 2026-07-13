import unittest
import xml.etree.ElementTree as ET

from validation.ned_score import _split_grand_staff

_GRAND_STAFF_PART = """
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <staves>2</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch>
        <duration>1</duration><type>quarter</type><voice>1</voice><staff>1</staff></note>
      <note><pitch><step>C</step><octave>3</octave></pitch>
        <duration>1</duration><type>quarter</type><voice>2</voice><staff>2</staff></note>
    </measure>
  </part>
"""

_VOICE_PART = """
  <part id="P0">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <clef number="1"><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>E</step><octave>5</octave></pitch>
        <duration>1</duration><type>quarter</type><voice>1</voice><staff>1</staff></note>
    </measure>
  </part>
"""


def _wrap(parts_xml: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P0" /><score-part id="P1" /></part-list>
  {parts_xml}
</score-partwise>"""


class TestSplitGrandStaff(unittest.TestCase):
    def test_lone_grand_staff_part_still_splits(self) -> None:
        xml_text = _wrap(_GRAND_STAFF_PART)

        result = _split_grand_staff(xml_text)

        root = ET.fromstring(result)  # noqa: S314
        parts = root.findall("part")
        self.assertEqual(2, len(parts))
        self.assertEqual(1, len(parts[0].findall(".//note")))
        self.assertEqual(1, len(parts[1].findall(".//note")))
        part_ids = {p.get("id") for p in parts}
        score_part_ids = {sp.get("id") for sp in root.findall(".//score-part")}
        self.assertEqual(part_ids, score_part_ids)

    def test_grand_staff_alongside_other_part_also_splits(self) -> None:
        # A solo voice part plus a piano grand staff: two <part> elements in
        # the document, only one of which has <staves>2</staves>. This used
        # to be left untouched because the old check required exactly one
        # <part> in the whole document.
        xml_text = _wrap(_VOICE_PART + _GRAND_STAFF_PART)

        result = _split_grand_staff(xml_text)

        root = ET.fromstring(result)  # noqa: S314
        parts = root.findall("part")
        self.assertEqual(3, len(parts))
        note_counts = [len(p.findall(".//note")) for p in parts]
        self.assertEqual([1, 1, 1], note_counts)
        part_ids = [p.get("id") for p in parts]
        self.assertEqual(len(part_ids), len(set(part_ids)))
        score_part_ids = {sp.get("id") for sp in root.findall(".//score-part")}
        self.assertEqual(set(part_ids), score_part_ids)

    def test_no_grand_staff_returns_input_unchanged(self) -> None:
        xml_text = _wrap(_VOICE_PART)

        result = _split_grand_staff(xml_text)

        self.assertEqual(xml_text, result)

    def test_invalid_xml_returns_input_unchanged(self) -> None:
        self.assertEqual("not xml", _split_grand_staff("not xml"))
