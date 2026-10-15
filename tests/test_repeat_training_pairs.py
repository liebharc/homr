"""Check repeat signs against the measures selected for each training image."""

# ruff: noqa: S101

import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import pytest

from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.convert_lieder import MeasureCutter
from training.omr_datasets.music_xml_parser import (
    Measure,
    music_xml_string_to_tokens,
    normalize_barlines_and_repeats,
)
from training.transformer.training_vocabulary import token_lines_to_str
from training.validate_music_xml_conversion import validate_conversion
from validation.ned_score import _xml_parts_from_text

ATTRIBUTES = """<attributes>
  <divisions>1</divisions><key><fifths>0</fifths></key>
  <time><beats>4</beats><beat-type>4</beat-type></time>
  <clef><sign>G</sign><line>2</line></clef>
</attributes>"""
START = '<barline location="left"><repeat direction="forward"/></barline>'
END = '<barline location="right"><repeat direction="backward"/></barline>'
DOUBLE = '<barline location="right"><bar-style>light-light</bar-style></barline>'
BREAKS = ["", '<print new-system="yes"/>', '<print new-page="yes"/>']
EMPTY_CONTENTS = [
    "",
    "<forward><duration>4</duration></forward>",
    '<note print-object="no"><rest/><duration>4</duration><type>whole</type></note>',
]


def _source(ending: str = DOUBLE, prefix: str = START) -> str:
    """Write two source measures with configurable boundary notation."""
    return f"""<score-partwise><part id="P1">
      <measure number="1">{ATTRIBUTES}
        <note><pitch><step>C</step><octave>4</octave></pitch>
          <duration>4</duration><type>whole</type></note>{ending}
      </measure>
      <measure number="2">{prefix}
        <note><pitch><step>D</step><octave>4</octave></pitch>
          <duration>4</duration><type>whole</type></note>
      </measure>
    </part></score-partwise>"""


def _measures(ending: str = DOUBLE, prefix: str = START) -> list[Measure]:
    return music_xml_string_to_tokens(_source(ending, prefix))[0]


def _rhythms(symbols: list[EncodedSymbol]) -> list[str]:
    return [symbol.rhythm for symbol in symbols]


@pytest.mark.parametrize("boundary", BREAKS)
def test_parser_preserves_repeat_measure(boundary: str) -> None:
    measures = _measures(prefix=boundary + START)
    assert len(measures) == 2
    assert _rhythms(measures[0])[-1] == "doublebarline"
    assert "repeatStart" not in _rhythms(measures[0])
    assert _rhythms(measures[1])[0] == "repeatStart"
    assert measures[1].new_page == ('new-page="yes"' in boundary)


def test_parser_completes_implicit_measure_boundaries() -> None:
    measures = _measures(ending="", prefix="")
    assert [_rhythms(measure) for measure in measures] == [
        ["clef_G2", "keySignature_0", "timeSignature/4", "note_1", "barline"],
        ["note_1", "barline"],
    ]


@pytest.mark.parametrize("boundary", BREAKS)
@pytest.mark.parametrize("content", EMPTY_CONTENTS)
def test_source_measure_without_visible_symbols_keeps_boundary(boundary: str, content: str) -> None:
    xml = f"""<score-partwise><part id="P1">
      <measure number="1">{boundary}{content}</measure>
    </part></score-partwise>"""
    measures = music_xml_string_to_tokens(xml)[0]
    assert [_rhythms(measure) for measure in measures] == [["barline"]]
    assert measures[0].new_page == ('new-page="yes"' in boundary)


@pytest.mark.parametrize("content", EMPTY_CONTENTS)
def test_empty_measure_with_opening_repeat_still_gets_ending_barline(content: str) -> None:
    xml = f"""<score-partwise><part id="P1">
      <measure number="1">{ATTRIBUTES}{START}{content}</measure>
      <measure number="2">
        <note><rest/><duration>4</duration><type>whole</type></note>
      </measure>
    </part></score-partwise>"""
    measures = music_xml_string_to_tokens(xml)[0]
    assert _rhythms(measures[0])[-2:] == ["repeatStart", "barline"]
    expected = [
        "clef_G2",
        "keySignature_0",
        "timeSignature/4",
        "repeatStart",
        "barline",
        "rest_1",
        "barline",
    ]
    assert _rhythms(MeasureCutter(measures).extract_measures(2)) == expected
    assert _rhythms(_xml_parts_from_text(xml, "native")[0]) == expected


def test_consecutive_empty_measures_keep_separate_boundaries_in_image() -> None:
    xml = """<score-partwise><part id="P1">
      <measure number="1"/><measure number="2"/>
    </part></score-partwise>"""
    measures = music_xml_string_to_tokens(xml)[0]
    assert [_rhythms(measure) for measure in measures] == [["barline"], ["barline"]]
    symbols = MeasureCutter(measures).extract_measures(2)
    assert _rhythms(symbols)[-2:] == ["barline", "barline"]
    assert _rhythms(symbols).count("barline") == 2


def test_part_without_measures_has_no_boundaries() -> None:
    xml = '<score-partwise><part id="P1"/></score-partwise>'
    assert music_xml_string_to_tokens(xml) == [[]]
    assert _xml_parts_from_text(xml, "native") == [[]]
    assert normalize_barlines_and_repeats([]) == []
    assert MeasureCutter([]).extract_measures(0) == []


def test_opposite_ends_of_one_measure_keep_separate_repeat_signs() -> None:
    xml = f"""<score-partwise><part id="P1"><measure number="1">
      {ATTRIBUTES}{START}
      <note print-object="no"><rest/><duration>4</duration><type>whole</type></note>
      {END}
    </measure></part></score-partwise>"""
    measures = music_xml_string_to_tokens(xml)[0]
    expected = [
        "clef_G2",
        "keySignature_0",
        "timeSignature/4",
        "repeatStart",
        "repeatEnd",
    ]
    assert _rhythms(measures[0]) == expected
    assert _rhythms(MeasureCutter(measures).extract_measures(1)) == expected
    assert _rhythms(_xml_parts_from_text(xml, "native")[0]) == expected


@pytest.mark.parametrize("content", EMPTY_CONTENTS)
@pytest.mark.parametrize("prefix", ["", START])
def test_empty_measure_after_repeat_end_keeps_its_boundary(
    tmp_path: Path, content: str, prefix: str
) -> None:
    xml = f"""<score-partwise><part id="P1">
      <measure number="1">{ATTRIBUTES}
        <note><pitch><step>C</step><octave>4</octave></pitch>
          <duration>4</duration><type>whole</type></note>{END}
      </measure>
      <measure number="2">{content}</measure>
      <measure number="3">{prefix}
        <note><pitch><step>D</step><octave>4</octave></pitch>
          <duration>4</duration><type>whole</type></note>
      </measure>
    </part></score-partwise>"""
    expected = [
        "clef_G2",
        "keySignature_0",
        "timeSignature/4",
        "note_1",
        "repeatEnd",
        "repeatStart" if prefix else "barline",
        "note_1",
        "barline",
    ]
    measures = music_xml_string_to_tokens(xml)[0]
    before = [token_lines_to_str(measure) for measure in measures]
    symbols = MeasureCutter(list(measures)).extract_measures(3)
    assert _rhythms(symbols) == expected
    assert _rhythms(_xml_parts_from_text(xml, "native")[0]) == expected
    assert [token_lines_to_str(measure) for measure in measures] == before

    answer = tmp_path / "image.tokens"
    answer.write_text(token_lines_to_str(symbols))
    source = ET.fromstring(xml)  # noqa: S314
    with patch("training.validate_music_xml_conversion.generate_xml", return_value=source):
        assert validate_conversion(str(answer))

    cutter = MeasureCutter(list(measures))
    assert _rhythms(cutter.extract_measures(1))[-1] == "repeatEnd"
    assert _rhythms(cutter.extract_measures(1)) == ["clef_G2", "keySignature_0", "barline"]
    assert _rhythms(cutter.extract_measures(1)) == ["clef_G2", "keySignature_0"] + (
        ["repeatStart"] if prefix else []
    ) + ["note_1", "barline"]


def test_combined_repeat_at_start_of_empty_measure_preserves_its_repeat_end() -> None:
    xml = f"""<score-partwise><part id="P1">
      <measure number="1">{ATTRIBUTES}
        <note><pitch><step>C</step><octave>4</octave></pitch>
          <duration>4</duration><type>whole</type></note>{END}
      </measure>
      <measure number="2">{START}
        <note print-object="no"><rest/><duration>4</duration><type>whole</type></note>{END}
      </measure>
    </part></score-partwise>"""
    measures = music_xml_string_to_tokens(xml)[0]
    expected = [
        "clef_G2",
        "keySignature_0",
        "timeSignature/4",
        "note_1",
        "repeatEndStart",
        "repeatEnd",
    ]
    assert _rhythms(MeasureCutter(list(measures)).extract_measures(2)) == expected
    assert _rhythms(_xml_parts_from_text(xml, "native")[0]) == expected

    cutter = MeasureCutter(list(measures))
    assert _rhythms(cutter.extract_measures(1))[-1] == "repeatEnd"
    assert _rhythms(cutter.extract_measures(1)) == [
        "clef_G2",
        "keySignature_0",
        "repeatStart",
        "repeatEnd",
    ]


@pytest.mark.parametrize("rhythms", [[], ["repeatStart", "note_1"]])
def test_normalization_does_not_add_measure_boundaries(rhythms: list[str]) -> None:
    symbols = [EncodedSymbol(rhythm) for rhythm in rhythms]
    assert _rhythms(normalize_barlines_and_repeats([symbols])) == rhythms


@pytest.mark.parametrize("boundary", BREAKS)
def test_separate_images_keep_repeat_in_second_answer(boundary: str) -> None:
    measures = _measures(prefix=boundary + START)
    before = [_rhythms(measure) for measure in measures]
    cutter = MeasureCutter(list(measures))
    previous = cutter.extract_measures(1)
    following = cutter.extract_measures(1)
    assert _rhythms(previous) == [
        "clef_G2",
        "keySignature_0",
        "timeSignature/4",
        "note_1",
        "doublebarline",
    ]
    assert _rhythms(following) == ["clef_G2", "keySignature_0", "repeatStart", "note_1", "barline"]
    assert [symbol.pitch for symbol in previous + following if symbol.rhythm == "note_1"] == [
        "C4",
        "D4",
    ]
    assert [_rhythms(measure) for measure in measures] == before


def test_repeat_inside_image_merges_adjacent_barline() -> None:
    symbols = MeasureCutter(_measures()).extract_measures(2)
    assert _rhythms(symbols) == [
        "clef_G2",
        "keySignature_0",
        "timeSignature/4",
        "note_1",
        "repeatStart",
        "note_1",
        "barline",
    ]


def test_window_start_keeps_repeat_without_printed_break() -> None:
    measures = _measures()
    cutter = MeasureCutter(measures[1:])
    # Window converters supply the inherited context from preceding measures.
    cutter.clefs = [symbol for symbol in measures[0] if symbol.rhythm.startswith("clef")]
    symbols = cutter.extract_measures(1, always_include_time=True)
    assert _rhythms(symbols) == [
        "clef_G2",
        "keySignature_0",
        "timeSignature/4",
        "repeatStart",
        "note_1",
        "barline",
    ]


def test_adjacent_end_and_start_stay_separate_across_images() -> None:
    cutter = MeasureCutter(_measures(ending=END))
    previous = cutter.extract_measures(1)
    following = cutter.extract_measures(1)
    assert _rhythms(previous)[-1] == "repeatEnd"
    assert "repeatStart" not in _rhythms(previous)
    assert "repeatEndStart" not in _rhythms(previous)
    assert _rhythms(following) == ["clef_G2", "keySignature_0", "repeatStart", "note_1", "barline"]


def test_adjacent_end_and_start_merge_inside_image() -> None:
    symbols = MeasureCutter(_measures(ending=END)).extract_measures(2)
    assert _rhythms(symbols) == [
        "clef_G2",
        "keySignature_0",
        "timeSignature/4",
        "note_1",
        "repeatEndStart",
        "note_1",
        "barline",
    ]


@pytest.mark.parametrize(
    "bar_style,expected",
    [
        ("regular", "barline"),
        ("light-light", "doublebarline"),
        ("light-heavy", "bolddoublebarline"),
    ],
)
def test_boundary_without_repeat_preserves_barline(bar_style: str, expected: str) -> None:
    ending = f'<barline location="right"><bar-style>{bar_style}</bar-style></barline>'
    symbols = MeasureCutter(_measures(ending=ending, prefix="")).extract_measures(2)
    assert _rhythms(symbols) == [
        "clef_G2",
        "keySignature_0",
        "timeSignature/4",
        "note_1",
        expected,
        "note_1",
        "barline",
    ]


def test_opening_repeat_does_not_duplicate_clef_key_or_time() -> None:
    xml = f"""<score-partwise><part id="P1"><measure number="1">
      {START}{ATTRIBUTES}
      <note><rest/><duration>4</duration><type>whole</type></note>
    </measure></part></score-partwise>"""
    symbols = MeasureCutter(music_xml_string_to_tokens(xml)[0]).extract_measures(1)
    assert _rhythms(symbols) == [
        "clef_G2",
        "keySignature_0",
        "timeSignature/4",
        "repeatStart",
        "rest_1",
        "barline",
    ]


def test_context_after_opening_repeat_replaces_inherited_context() -> None:
    attributes = """<attributes><key><fifths>2</fifths></key>
      <time><beats>6</beats><beat-type>8</beat-type></time>
      <clef><sign>F</sign><line>4</line></clef></attributes>"""
    cutter = MeasureCutter(_measures(prefix=START + attributes))
    cutter.extract_measures(1)
    symbols = cutter.extract_measures(1)
    assert _rhythms(symbols) == [
        "clef_F4",
        "keySignature_2",
        "timeSignature/8",
        "repeatStart",
        "note_1",
        "barline",
    ]


@pytest.mark.parametrize("ending", [DOUBLE, END])
def test_image_transcription_validation_preserves_repeats(tmp_path: Path, ending: str) -> None:
    symbols = MeasureCutter(_measures(ending=ending)).extract_measures(2)
    answer = tmp_path / "image.tokens"
    answer.write_text(token_lines_to_str(symbols))
    # Fix the validator's XML input to the independently authored source.
    source = ET.fromstring(_source(ending))  # noqa: S314
    with patch("training.validate_music_xml_conversion.generate_xml", return_value=source):
        assert validate_conversion(str(answer))


def test_validation_does_not_combine_repeats_from_different_parts(tmp_path: Path) -> None:
    xml = f"""<score-partwise>
      <part id="P1"><measure number="1">{ATTRIBUTES}
        <note><rest/><duration>4</duration><type>whole</type></note>{END}
      </measure></part>
      <part id="P2"><measure number="1">{START}
        <note><rest/><duration>4</duration><type>whole</type></note>
      </measure></part>
    </score-partwise>"""
    parts = music_xml_string_to_tokens(xml)
    expected = [symbol for part in parts for measure in part for symbol in measure]
    assert _rhythms(expected)[-4:] == ["repeatEnd", "repeatStart", "rest_1", "barline"]
    answer = tmp_path / "image.tokens"
    answer.write_text(token_lines_to_str(expected))
    source = ET.fromstring(xml)  # noqa: S314
    with patch("training.validate_music_xml_conversion.generate_xml", return_value=source):
        assert validate_conversion(str(answer))
    actual = _xml_parts_from_text(xml, "native")
    assert len(actual) == 2
    assert _rhythms(actual[0])[-1] == "repeatEnd"
    assert _rhythms(actual[1]) == ["repeatStart", "rest_1", "barline"]


@pytest.mark.parametrize("ending", [DOUBLE, END])
def test_whole_part_comparison_preserves_repeat_transcription(ending: str) -> None:
    symbols = MeasureCutter(_measures(ending=ending)).extract_measures(2)
    actual = _xml_parts_from_text(_source(ending), "native")
    assert len(actual) == 1
    assert token_lines_to_str(actual[0]) == token_lines_to_str(symbols)
