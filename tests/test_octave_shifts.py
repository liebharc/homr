# ruff: noqa: S101
import xml.etree.ElementTree as ET

import pytest

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from training.omr_datasets.convert_lieder import MeasureCutter
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.transformer.training_vocabulary import (
    read_token_lines,
    token_lines_to_str,
)


def _measure(content: str, attributes: str = "") -> str:
    return f"""<measure>
      <attributes><divisions>1</divisions>{attributes}</attributes>
      {content}
    </measure>"""


def _score(*measures: str) -> str:
    return f"""<score-partwise version="4.0"><part id="P1">
      {"".join(measures)}
    </part></score-partwise>"""


def _note(step: str, octave: int, staff: int = 1, voice: int = 1) -> str:
    return f"""<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>
      <duration>1</duration><voice>{voice}</voice><type>quarter</type>
      <staff>{staff}</staff></note>"""


def _shift(shift_type: str, staff: int = 1, size: int = 8) -> str:
    return f"""<direction><direction-type>
      <octave-shift type="{shift_type}" size="{size}"/>
      </direction-type><staff>{staff}</staff></direction>"""


TREBLE = (
    "<time><beats>2</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef>"
)


def _tokens(xml: str, sounding_pitch: bool = False) -> str:
    parts = music_xml_string_to_tokens(xml, sounding_pitch=sounding_pitch)
    return token_lines_to_str([s for part in parts for measure in part for s in measure])


def test_octave_shift_has_written_pitches_until_it_stops() -> None:
    xml = _score(
        _measure(_shift("down") + _note("C", 6) + _note("D", 6), TREBLE),
        _measure(_note("E", 6) + _shift("stop") + _note("F", 5)),
    )
    assert _tokens(xml) == """clef_G2 _ _ _ _ upper
timeSignature/4 . . . . .
octaveShift_8va _ _ _ _ upper
note_4 C5 _ _ _ upper
note_4 D5 _ _ _ upper
barline . . . . .
note_4 E5 _ _ _ upper
octaveShiftStop _ _ _ _ upper
note_4 F5 _ _ _ upper
barline . . . . ."""


def test_benchmarks_keep_sounding_pitches() -> None:
    xml = _score(_measure(_shift("down") + _note("C", 6) + _shift("stop") + _note("D", 5), TREBLE))
    assert _tokens(xml, sounding_pitch=True) == """clef_G2 _ _ _ _ upper
timeSignature/4 . . . . .
note_4 C6 _ _ _ upper
note_4 D5 _ _ _ upper
barline . . . . ."""


def test_octave_clef() -> None:
    clef = "<clef><sign>G</sign><line>2</line><clef-octave-change>-1</clef-octave-change></clef>"
    xml = _score(_measure(_note("C", 3) + _note("D", 3), clef))
    assert _tokens(xml) == """clef_G2_8vb _ _ _ _ upper
note_4 C4 _ _ _ upper
note_4 D4 _ _ _ upper
barline . . . . ."""


def test_shift_applies_by_time_not_by_document_order() -> None:
    # The stop is written after the first voice, the second voice starts earlier
    xml = _score(
        _measure(
            _shift("down")
            + _note("C", 6)
            + _note("D", 6)
            + _shift("stop")
            + "<backup><duration>2</duration></backup>"
            + _note("E", 6, voice=2)
            + _note("F", 6, voice=2),
            TREBLE,
        )
    )
    tokens = _tokens(xml)
    assert "note_4 E5 _ _ _ upper2" in tokens
    assert "note_4 F5 _ _ _ upper2" in tokens


def test_unsupported_octave_marks_are_rejected() -> None:
    with pytest.raises(ValueError):
        music_xml_string_to_tokens(
            _score(_measure(_shift("down", size=15) + _note("C", 7), TREBLE))
        )
    clef = "<clef><sign>C</sign><line>3</line><clef-octave-change>-1</clef-octave-change></clef>"
    with pytest.raises(ValueError):
        music_xml_string_to_tokens(_score(_measure(_note("C", 3), clef)))


def test_continued_shift_is_repeated_on_the_next_staff() -> None:
    xml = _score(
        _measure(_shift("down") + _note("C", 6) + _note("D", 6), TREBLE),
        _measure(_note("E", 6) + _shift("stop") + _note("F", 5)),
    )
    cutter = MeasureCutter(music_xml_string_to_tokens(xml)[0])
    cutter.extract_measures(1)
    assert token_lines_to_str(cutter.extract_measures(1)) == """clef_G2 _ _ _ _ upper
keySignature_0 . . . . .
octaveShift_8va _ _ _ _ upper
note_4 E5 _ _ _ upper
octaveShiftStop _ _ _ _ upper
note_4 F5 _ _ _ upper
barline . . . . ."""


def test_shift_which_stops_at_the_start_of_a_staff_isnt_repeated() -> None:
    xml = _score(
        _measure(_shift("down") + _note("C", 6) + _note("D", 6), TREBLE),
        _measure(_shift("stop") + _note("E", 5) + _note("F", 5)),
    )
    cutter = MeasureCutter(music_xml_string_to_tokens(xml)[0])
    cutter.extract_measures(1)
    assert token_lines_to_str(cutter.extract_measures(1)) == """clef_G2 _ _ _ _ upper
keySignature_0 . . . . .
note_4 E5 _ _ _ upper
note_4 F5 _ _ _ upper
barline . . . . ."""


def test_generator_writes_sounding_pitches() -> None:
    tokens = """clef_G2_8vb _ _ _ _ upper&clef_F4 _ _ _ _ lower
timeSignature/4 . . . . .
note_2 C4 _ _ _ upper&note_2 C3 _ _ _ lower
octaveShift_8va _ _ _ _ upper&octaveShift_8vb _ _ _ _ lower
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . ."""
    xml = generate_xml(XmlGeneratorArguments(), [read_token_lines(tokens.splitlines())], "")
    clef = xml.find(".//clef[@number='1']")
    assert clef is not None
    assert clef.findtext("clef-octave-change") == "-1"
    pitches = [
        (n.findtext("pitch/step", "") + n.findtext("pitch/octave", ""), n.findtext("staff"))
        for n in xml.iter("note")
    ]
    assert pitches == [("C3", "1"), ("C3", "2"), ("C5", "1"), ("C2", "2")]
    shifts = sorted((s.get("type", ""), s.get("number", "")) for s in xml.iter("octave-shift"))
    assert shifts == [("down", "1"), ("stop", "1"), ("stop", "2"), ("up", "2")]


def test_generator_ends_shift_with_its_staff() -> None:
    tokens = """clef_G2 _ _ _ _ upper
timeSignature/4 . . . . .
octaveShift_8va _ _ _ _ upper
note_1 C5 _ _ _ upper
barline . . . . .
newline . . . . .
clef_G2 _ _ _ _ upper
note_1 C5 _ _ _ upper
barline . . . . ."""
    xml = generate_xml(XmlGeneratorArguments(), [read_token_lines(tokens.splitlines())], "")
    octaves = [n.findtext("pitch/octave") for n in xml.iter("note")]
    assert octaves == ["6", "5"]
    assert [s.get("type") for s in xml.iter("octave-shift")] == ["down", "stop"]


def test_round_trip() -> None:
    tokens = """clef_G2_8vb _ _ _ _ upper
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper
octaveShift_8va _ _ _ _ upper
note_4 C5 _ _ _ upper
octaveShiftStop _ _ _ _ upper
note_4 D5 _ _ _ upper
note_4 E5 _ _ _ upper
barline . . . . ."""
    xml = generate_xml(XmlGeneratorArguments(), [read_token_lines(tokens.splitlines())], "")
    assert _tokens(ET.tostring(xml, encoding="unicode")) == tokens
