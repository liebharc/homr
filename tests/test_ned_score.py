import unittest
import xml.etree.ElementTree as ET

from homr.transformer.vocabulary import EncodedSymbol
from validation.ned_score import _align_parts, _ned_from_parts, _split_grand_staff


def _sym(rhythm: str) -> EncodedSymbol:
    return EncodedSymbol(rhythm)


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


class TestAlignParts(unittest.TestCase):
    def test_reorders_reversed_parts_by_content(self) -> None:
        # kern lists [bass, treble] but the tool's split-grand-staff output
        # lists [treble, bass] for the same piece - the exact reversal found
        # in polish-scores samples like 43 and 52 (see memory).
        bass = [_sym("note_4"), _sym("note_8"), _sym("note_4")]
        treble = [_sym("note_8"), _sym("note_16"), _sym("note_16"), _sym("note_4")]
        kern_parts = [bass, treble]
        xml_parts = [treble, bass]

        aligned_kern, aligned_xml = _align_parts(kern_parts, xml_parts)

        self.assertEqual(aligned_kern, [bass, treble])
        self.assertEqual(aligned_xml, [bass, treble])

    def test_already_aligned_parts_are_unchanged(self) -> None:
        part_a = [_sym("note_4"), _sym("note_8")]
        part_b = [_sym("note_16"), _sym("note_4"), _sym("note_4")]

        aligned_kern, aligned_xml = _align_parts([part_a, part_b], [part_a, part_b])

        self.assertEqual(aligned_kern, [part_a, part_b])
        self.assertEqual(aligned_xml, [part_a, part_b])

    def test_ned_from_parts_ignores_part_order(self) -> None:
        # Same content as test_reorders_reversed_parts_by_content: a perfect,
        # zero-distance match once parts are correctly paired up. Before the
        # fix, comparing positionally (kern's bass against the tool's treble)
        # would have produced a large, spurious NED for identical content.
        bass = [_sym("note_4"), _sym("note_8"), _sym("note_4")]
        treble = [_sym("note_8"), _sym("note_16"), _sym("note_16"), _sym("note_4")]

        result = _ned_from_parts([bass, treble], [treble, bass])

        self.assertEqual(0, result.distance)
        self.assertEqual(0.0, result.ned)
