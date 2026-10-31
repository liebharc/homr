# ruff: noqa: S101
import xml.etree.ElementTree as ET

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import split_slur_and_beam
from training.omr_datasets.humdrum_kern_parser import convert_kern_to_tokens
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.transformer.training_vocabulary import (
    read_token_lines,
    token_lines_to_str,
)


def _xml_note(step: str, note_type: str, duration: int, beam: str = "", chord: bool = False) -> str:
    chord_tag = "<chord/>" if chord else ""
    beam_tag = f'<beam number="1">{beam}</beam>' if beam else ""
    return f"""<note>{chord_tag}<pitch><step>{step}</step><octave>5</octave></pitch>
      <duration>{duration}</duration><voice>1</voice><type>{note_type}</type>
      <staff>1</staff>{beam_tag}</note>"""


def _xml(*notes: str) -> str:
    return f"""<score-partwise version="4.0"><part id="P1"><measure>
      <attributes><divisions>2</divisions>
      <time><beats>2</beats><beat-type>4</beat-type></time>
      <clef><sign>G</sign><line>2</line></clef></attributes>
      {"".join(notes)}
    </measure></part></score-partwise>"""


def _tokens(xml: str) -> str:
    parts = music_xml_string_to_tokens(xml)
    return token_lines_to_str([s for part in parts for measure in part for s in measure])


def _generate(tokens: str) -> ET.Element:
    return generate_xml(XmlGeneratorArguments(), [read_token_lines(tokens.splitlines())], "")


def _beams(xml: ET.Element) -> list[list[str]]:
    return [
        [f"{b.get('number')}:{b.text}" for b in note.findall("beam")] for note in xml.iter("note")
    ]


def test_music_xml_beam_is_on_the_chord() -> None:
    xml = _xml(
        _xml_note("C", "eighth", 1, "begin"),
        _xml_note("E", "eighth", 1, chord=True),
        _xml_note("D", "eighth", 1, "end"),
        _xml_note("E", "quarter", 2),
    )
    assert _tokens(xml) == """clef_G2 _ _ _ _ upper
timeSignature/4 . . . . .
note_8 E5 _ _ beamStart upper&note_8 C5 _ _ _ upper
note_8 D5 _ _ beamStop upper
note_4 E5 _ _ _ upper
barline . . . . ."""


def test_kern_beams_count_all_levels() -> None:
    kern = """**kern\t**kern
*clefF4\t*clefG2
*M2/4\t*M2/4
4C\t16ccLL
.\t16ddJ
.\t8eeJ
4D\t8dd 8ffL
.\t8ccJ
=\t=
*-\t*-"""
    tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))
    upper = [line.split("&")[0] for line in tokens.splitlines() if "upper" in line]
    beams = [split_slur_and_beam(line.split()[4])[1] for line in upper if line.startswith("note")]
    assert beams == ["beamStart", "beamContinue", "beamStop", "beamStart", "beamStop"]


def test_generator_adds_secondary_beams_and_hooks() -> None:
    xml = _generate("""clef_G2 _ _ _ _ upper
timeSignature/4 . . . . .
note_8. C5 _ _ beamStart upper
note_16 D5 _ _ beamStop upper
note_16 E5 _ _ beamStart upper
note_16 F5 _ _ beamContinue upper
note_8 G5 _ _ beamStop upper
barline . . . . .""")
    assert _beams(xml) == [
        ["1:begin"],
        ["1:end", "2:backward hook"],
        ["1:begin", "2:begin"],
        ["1:continue", "2:end"],
        ["1:end"],
    ]


def test_generator_repairs_beams() -> None:
    # A beam which starts but never continues, a quarter in a beam and a group
    # without a start
    xml = _generate("""clef_G2 _ _ _ _ upper
timeSignature/4 . . . . .
note_8 C5 _ _ beamStart upper
note_4 D5 _ _ beamContinue upper
note_8 E5 _ _ beamContinue upper
note_8 F5 _ _ beamStop upper
note_8 G5 _ _ beamStart upper
barline . . . . .""")
    assert _beams(xml) == [[], [], ["1:begin"], ["1:end"], []]


def test_generator_beams_chords_on_the_first_note() -> None:
    xml = _generate("""clef_G2 _ _ _ _ upper
timeSignature/4 . . . . .
note_8 C5 _ _ beamStart upper&note_8 E5 _ _ _ upper
note_8 D5 _ _ beamStop upper&note_8 F5 _ _ _ upper
barline . . . . .""")
    assert _beams(xml) == [["1:begin"], [], ["1:end"], []]


def test_round_trip() -> None:
    tokens = """clef_G2 _ _ _ _ upper
timeSignature/4 . . . . .
note_8 C5 _ _ beamStart_slurStart upper
note_16 D5 _ _ beamContinue upper
note_16 E5 _ _ beamStop_slurStop upper
note_8 F5 _ _ _ upper
note_8 G5 _ _ _ upper
barline . . . . ."""
    assert _tokens(ET.tostring(_generate(tokens), encoding="unicode")) == tokens


def test_generator_joins_cross_staff_beams() -> None:
    xml = _generate("""clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
timeSignature/4 . . . . .
note_8 C5 _ _ beamStart upper
note_8 C3 _ _ beamContinue lower
note_8 D5 _ _ beamStop upper
note_8 D3 _ _ _ lower
barline . . . . .""")
    notes = list(xml.iter("note"))
    assert _beams(xml) == [["1:begin"], ["1:continue"], ["1:end"], []]
    # A beam is drawn within one voice, the lower note moves into it
    assert notes[1].findtext("staff") == "2"
    assert notes[1].findtext("voice") == notes[0].findtext("voice")
