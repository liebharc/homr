# ruff: noqa: S101, S314
import xml.etree.ElementTree as ET

from homr.music_xml.accidentals import add_accidentals
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol
from tests.music_xml_model import apply
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens


def note(
    alter: int = 0,
    *,
    step: str = "F",
    octave: int = 4,
    staff: int = 1,
    voice: int = 1,
    prefix: str = "",
    tie: str = "",
) -> str:
    return (
        f"<note>{prefix}<pitch><step>{step}</step><alter>{alter}</alter>"
        f"<octave>{octave}</octave></pitch><duration>4</duration>{tie}"
        f"<voice>{voice}</voice><type>quarter</type><staff>{staff}</staff></note>"
    )


def accidentals(*measures: str) -> list[str | None]:
    part = ET.fromstring(
        "<part>" + "".join(f"<measure>{m}</measure>" for m in measures) + "</part>"
    )
    part = apply(add_accidentals, part)
    return [n.findtext("accidental") for n in part.findall("measure/note")]


def test_key_signature_is_inherited_but_accidentals_reset() -> None:
    key = "<attributes><key><fifths>1</fifths></key></attributes>"
    assert accidentals(key + note(1) + note() + note(), note(1) + note()) == [
        None,
        "natural",
        None,
        None,
        "natural",
    ]


def test_accidentals_are_local_to_staff_and_octave() -> None:
    notes = note(1) + note(octave=5) + note(staff=2) + note(1) + note()
    assert accidentals(notes) == ["sharp", None, None, None, "natural"]


def test_simultaneous_unison_prints_once() -> None:
    backup = "<backup><duration>4</duration></backup>"
    assert accidentals(note(-1) + backup + note(-1, voice=2)) == ["flat", None]


def test_voices_are_read_in_musical_order() -> None:
    backup = "<backup><duration>8</duration></backup>"
    first_voice = note(step="G") + note(1)
    assert accidentals(first_voice + backup + note(1, voice=2)) == [None, None, "sharp"]


def test_tied_note_does_not_change_the_next_measure_accidental() -> None:
    start, stop = '<tie type="start"/>', '<tie type="stop"/>'
    assert accidentals(note(1, tie=start), note(1, tie=stop) + note(1)) == [
        "sharp",
        None,
        "sharp",
    ]


def test_chord_uses_the_same_onset_for_its_notes() -> None:
    backup = "<backup><duration>4</duration></backup>"
    chord = note(step="D") + note(1, prefix="<chord/>")
    assert accidentals(chord + backup + note(1, voice=2)) == [None, "sharp", None]


def test_exported_double_accidentals_survive_reimport() -> None:
    symbols = [
        EncodedSymbol("clef_G2"),
        EncodedSymbol("note_4", "C4", "##"),
        EncodedSymbol("note_4", "B4", "bb"),
    ]
    xml = generate_xml(XmlGeneratorArguments(), [symbols], "")
    assert [n.text for n in xml.findall("part/measure/note/accidental")] == [
        "double-sharp",
        "flat-flat",
    ]
    parts = music_xml_string_to_tokens(ET.tostring(xml, encoding="unicode"))
    assert [n.lift for m in parts[0] for n in m if n.rhythm.startswith("note_")] == ["##", "bb"]


def test_grace_note_sets_the_accidental_before_a_regular_note() -> None:
    grace = note(1, prefix="<grace/>").replace("<duration>4</duration>", "")
    backup = "<backup><duration>4</duration></backup>"
    assert accidentals(note() + backup + grace) == ["natural", "sharp"]


def test_export_writes_sign_after_dot_without_changing_pitch() -> None:
    symbols = [EncodedSymbol("clef_G2"), EncodedSymbol("note_4.", "F4", "#")]
    xml = generate_xml(XmlGeneratorArguments(), [symbols], "")
    exported = xml.find("part/measure/note")
    assert exported is not None
    assert exported.findtext("pitch/alter") == "1"
    assert exported.findtext("accidental") == "sharp"
    tags = [child.tag for child in exported]
    assert tags[tags.index("dot") + 1] == "accidental"
