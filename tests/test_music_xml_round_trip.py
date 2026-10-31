# ruff: noqa: S101
import xml.etree.ElementTree as ET

import pytest

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from training.omr_datasets.music_xml_parser import (
    music_xml_string_to_tokens,
    normalize_barlines_and_repeats,
)
from training.transformer.training_vocabulary import (
    read_token_lines,
    token_lines_to_str,
)

HEADER = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
timeSignature/4 . . . . .
"""


def _round_trip(tokens: str) -> str:
    xml = generate_xml(XmlGeneratorArguments(), [read_token_lines(tokens.splitlines())], "")
    parts = music_xml_string_to_tokens(ET.tostring(xml, encoding="unicode"))
    return token_lines_to_str([s for part in parts for s in normalize_barlines_and_repeats(part)])


@pytest.mark.parametrize(
    "body",
    [
        # Triplets against eighths in the other staff
        """note_12 E4 _ _ _ upper&note_8 C3 _ _ _ lower
note_12 F4 _ _ _ upper
note_8 D3 _ _ _ lower
note_12 G4 _ _ _ upper
note_2 A4 _ _ _ upper&note_2 E3 _ _ _ lower
note_4 B4 _ _ _ upper&note_4 F3 _ _ _ lower
barline . . . . .""",
        # A quarter and an eighth triplet, and a half and a quarter triplet
        """note_6 E4 _ _ _ upper&note_2 C3 _ _ _ lower
note_12 F4 _ _ _ upper
note_4 G4 _ _ _ upper
note_3 A4 _ _ _ upper&note_2 D3 _ _ _ lower
note_6 B4 _ _ _ upper
barline . . . . .""",
        # Triplets in both staves, the upper one starts with a rest in another voice
        """rest_12 _ _ _ _ upper&note_12 E2 _ _ _ lower
note_6 G4 _ _ _ upper&note_12 E4 _ _ _ upper2&note_12 E3 _ _ _ lower
note_12 F4 _ _ _ upper2&note_12 F3 _ _ _ lower
note_2. C5 _ _ _ upper&note_2. C4 _ _ _ upper2&note_2. C3 _ _ _ lower
barline . . . . .""",
        # A note which waits for the earlier note of its voice, not for the other staff
        """note_4 F4 _ _ _ upper&note_2 A2 _ _ _ lower
note_2 F3 _ _ _ upper
note_8. G3 _ _ _ upper&note_2 B2 _ _ _ lower
note_16 A3 _ _ _ upper
barline . . . . .""",
        # Two voices on one staff
        """note_4 E5 _ _ _ upper&note_4 A4 _ _ _ upper2&note_1 C3 _ _ _ lower
note_4 F5 _ _ _ upper&note_4 B4 _ _ _ upper2
note_2 G5 _ _ _ upper&note_2 C5 _ _ _ upper2
barline . . . . .""",
    ],
)
def test_round_trip(body: str) -> None:
    tokens = HEADER + body
    assert _round_trip(tokens) == tokens


def _score(measure: str) -> str:
    return f"""<score-partwise version="4.0"><part id="P1"><measure>
      <attributes><divisions>2</divisions><time><beats>2</beats><beat-type>4</beat-type></time>
      <staves>2</staves>
      <clef number="1"><sign>G</sign><line>2</line></clef>
      <clef number="2"><sign>F</sign><line>4</line></clef></attributes>
      {measure}
    </measure></part></score-partwise>"""


def _note(
    step: str, octave: int, duration: int, note_type: str, voice: int, staff: int, extra: str = ""
) -> str:
    return f"""<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>
      <duration>{duration}</duration><voice>{voice}</voice><type>{note_type}</type>
      {extra}<staff>{staff}</staff></note>"""


def _tokens(xml: str) -> str:
    parts = music_xml_string_to_tokens(xml)
    return token_lines_to_str([s for part in parts for measure in part for s in measure])


def test_note_without_notehead_takes_time() -> None:
    hidden = """<note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration>
      <voice>2</voice><type>quarter</type><notehead>none</notehead><staff>1</staff></note>"""
    xml = _score(
        _note("E", 5, 2, "quarter", 1, 1)
        + hidden
        + "<backup><duration>4</duration></backup>"
        + _note("C", 3, 4, "half", 5, 2)
    )
    assert "note_2 C3 _ _ _ lower" in _tokens(xml)


def test_cross_staff_notes_dont_make_a_main_voice() -> None:
    # Voice 1 has a note on the lower staff, the main voice there is still voice 5
    xml = _score(
        _note("C", 5, 1, "eighth", 1, 1, "<stem>up</stem>")
        + _note("C", 4, 1, "eighth", 1, 2, "<stem>up</stem>")
        + _note("D", 5, 2, "quarter", 1, 1, "<stem>up</stem>")
        + "<backup><duration>4</duration></backup>"
        + _note("C", 3, 2, "quarter", 5, 2, "<stem>down</stem>")
        + _note("D", 3, 2, "quarter", 5, 2, "<stem>down</stem>")
    )
    tokens = _tokens(xml)
    assert "note_4 C3 _ _ _ lower" in tokens
    assert "note_8 C4 _ _ _ lower2" in tokens
