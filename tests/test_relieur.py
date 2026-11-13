import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from homr.relieur import get_children_of_type, process_concat


def _page(divisions: str, fifths: str, duration: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>{divisions}</divisions>
        <key><fifths>{fifths}</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch>
        <duration>{duration}</duration><type>whole</type></note>
    </measure>
  </part>
</score-partwise>
"""


def _merge(pages: list[str]) -> list[ET.Element]:
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for i, page in enumerate(pages):
            path = os.path.join(tmp, f"page{i}.musicxml")
            with open(path, "w", encoding="utf-8") as f:
                f.write(page)
            files.append(path)
        root, _, _ = process_concat(files)
    part = get_children_of_type(root, "part")[0]
    return get_children_of_type(part, "measure")


def _divisions(measure: ET.Element) -> str | None:
    for attrib in get_children_of_type(measure, "attributes"):
        for div in get_children_of_type(attrib, "divisions"):
            return div.text
    return None


def _fifths(measure: ET.Element) -> str | None:
    for attrib in get_children_of_type(measure, "attributes"):
        for key in get_children_of_type(attrib, "key"):
            return get_children_of_type(key, "fifths")[0].text
    return None


class TestRelieur(unittest.TestCase):

    def test_divisions_kept_when_they_change_back_to_the_first_pages_value(self) -> None:
        # Divisions 4, then 2, then 4 again: the third page's divisions must be kept,
        # otherwise its durations are read with the second page's divisions (doubled).
        measures = _merge([_page("4", "0", "16"), _page("2", "0", "8"), _page("4", "0", "16")])
        self.assertEqual([_divisions(m) for m in measures], ["4", "2", "4"])

    def test_divisions_removed_when_unchanged_from_previous_page(self) -> None:
        measures = _merge([_page("4", "0", "16"), _page("2", "0", "8"), _page("2", "0", "8")])
        self.assertEqual([_divisions(m) for m in measures], ["4", "2", None])

    def test_key_kept_when_it_changes_back(self) -> None:
        measures = _merge([_page("4", "0", "16"), _page("4", "-1", "16"), _page("4", "0", "16")])
        self.assertEqual([_fifths(m) for m in measures], ["0", "-1", "0"])
