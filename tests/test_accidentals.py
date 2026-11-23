# ruff: noqa: S101, S314
import xml.etree.ElementTree as ET

from homr.circle_of_fifths import strip_naturals
from homr.music_xml_generator import (
    XmlGeneratorArguments,
    add_accidentals,
    generate_xml,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.music_xml_parser import (
    music_xml_string_to_tokens,
    normalize_barlines_and_repeats,
)


def _acc_list(xml: ET.Element) -> list[tuple[str, str, str | None, str | None]]:
    out = []
    for note in xml.findall(".//part/measure/note"):
        pitch = note.find("pitch")
        if pitch is None:
            continue
        step = pitch.findtext("step", "")
        octv = pitch.findtext("octave", "")
        alter = pitch.findtext("alter")
        acc = note.findtext("accidental")
        out.append((f"{step}{octv}", note.findtext("staff", "1"), alter, acc))
    return out


def _part_acc_list(part: ET.Element) -> list[tuple[str, str, str, str | None]]:
    out = []
    for note in part.findall("./measure/note"):
        pitch = note.find("pitch")
        if pitch is None:
            continue
        step = pitch.findtext("step", "")
        octv = pitch.findtext("octave", "")
        out.append(
            (
                f"{step}{octv}",
                note.findtext("staff", "1"),
                note.findtext("voice", "1"),
                note.findtext("accidental"),
            )
        )
    return out


def _xml_note(
    step: str,
    octave: int,
    *,
    alter: float | None = None,
    dur: int = 4,
    voice: int = 1,
    staff: int = 1,
    chord: bool = False,
    grace: bool = False,
    ties: tuple[str, ...] = (),
) -> str:
    chord_tag = "<chord/>" if chord else ""
    grace_tag = "<grace/>" if grace else ""
    alter_tag = f"<alter>{alter}</alter>" if alter is not None else ""
    dur_tag = "" if grace else f"<duration>{dur}</duration>"
    tie_tags = "".join(f'<tie type="{t}"/>' for t in ties)
    note_type = "eighth" if grace else "quarter"
    return (
        f"<note>{chord_tag}{grace_tag}"
        f"<pitch><step>{step}</step>{alter_tag}<octave>{octave}</octave></pitch>"
        f"{dur_tag}{tie_tags}<voice>{voice}</voice><type>{note_type}</type>"
        f"<staff>{staff}</staff></note>"
    )


def test_key_signature_and_measure_reset() -> None:
    # Key of D major (2 sharps: F#, C#)
    # Bar 1: F#4 (in key -> None), F4 (natural -> "natural"), F4 again (persists -> None),
    #        F#4 (restored -> "sharp"), F#5 (other octave, in key -> None), G#4 (sharp -> "sharp")
    # Bar 2: F#4 (in key after reset -> None), G#4 (reset -> "sharp")
    syms = [
        EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
        EncodedSymbol("keySignature_2", ".", ".", ".", ".", "."),
        EncodedSymbol("timeSignature/4", ".", ".", ".", ".", "."),
        EncodedSymbol("note_8", "F4", "#", "_", "_", "upper"),
        EncodedSymbol("note_8", "F4", "_", "_", "_", "upper"),
        EncodedSymbol("note_8", "F4", "_", "_", "_", "upper"),
        EncodedSymbol("note_8", "F4", "#", "_", "_", "upper"),
        EncodedSymbol("note_8", "F5", "#", "_", "_", "upper"),
        EncodedSymbol("note_8", "G4", "#", "_", "_", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
        EncodedSymbol("note_4", "F4", "#", "_", "_", "upper"),
        EncodedSymbol("note_4", "G4", "#", "_", "_", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
    ]
    xml = generate_xml(XmlGeneratorArguments(), [syms], "")
    assert _acc_list(xml) == [
        ("F4", "1", "1", None),
        ("F4", "1", None, "natural"),
        ("F4", "1", None, None),
        ("F4", "1", "1", "sharp"),
        ("F5", "1", "1", None),
        ("G4", "1", "1", "sharp"),
        ("F4", "1", "1", None),
        ("G4", "1", "1", "sharp"),
    ]


def test_double_accidentals_and_round_trip() -> None:
    syms = [
        EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
        EncodedSymbol("timeSignature/4", ".", ".", ".", ".", "."),
        EncodedSymbol("note_4", "C4", "##", "_", "_", "upper"),
        EncodedSymbol("note_4", "B4", "bb", "_", "_", "upper"),
        EncodedSymbol("note_4", "C4", "#", "_", "_", "upper"),
        EncodedSymbol("note_4", "B4", "b", "_", "_", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
    ]
    xml = generate_xml(XmlGeneratorArguments(), [syms], "")
    assert _acc_list(xml) == [
        ("C4", "1", "2", "double-sharp"),
        ("B4", "1", "-2", "flat-flat"),
        ("C4", "1", "1", "sharp"),
        ("B4", "1", "-1", "flat"),
    ]
    parts = music_xml_string_to_tokens(ET.tostring(xml, encoding="unicode"), sounding_pitch=True)
    flat = strip_naturals(normalize_barlines_and_repeats(parts[0]))
    lifts = [s.lift for s in flat if s.rhythm.startswith("note_")]
    assert lifts == ["##", "bb", "#", "b"]


def test_parser_natural_sharp_and_natural_flat() -> None:
    xml_str = """<score-partwise version="4.0"><part id="P1"><measure number="1">
      <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>
      <clef><sign>G</sign><line>2</line></clef></attributes>
      <note><pitch><step>C</step><alter>2</alter><octave>4</octave></pitch>
        <duration>1</duration><voice>1</voice><type>quarter</type>
        <accidental>double-sharp</accidental><staff>1</staff></note>
      <note><pitch><step>C</step><alter>1</alter><octave>4</octave></pitch>
        <duration>1</duration><voice>1</voice><type>quarter</type>
        <accidental>natural-sharp</accidental><staff>1</staff></note>
      <note><pitch><step>B</step><alter>-2</alter><octave>4</octave></pitch>
        <duration>1</duration><voice>1</voice><type>quarter</type>
        <accidental>flat-flat</accidental><staff>1</staff></note>
      <note><pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch>
        <duration>1</duration><voice>1</voice><type>quarter</type>
        <accidental>natural-flat</accidental><staff>1</staff></note>
      <note><pitch><step>F</step><alter>2</alter><octave>4</octave></pitch>
        <duration>1</duration><voice>1</voice><type>quarter</type>
        <accidental>sharp-sharp</accidental><staff>1</staff></note>
    </measure></part></score-partwise>"""
    parts = music_xml_string_to_tokens(xml_str, sounding_pitch=True)
    flat = strip_naturals(normalize_barlines_and_repeats(parts[0]))
    lifts = [s.lift for s in flat if s.rhythm.startswith("note_")]
    assert lifts == ["##", "#", "bb", "b", "##"]


def test_ties_across_barline_and_within_measure() -> None:
    # C major: F#4 tied across barline (same system) -> tied stop gets None,
    # following untied F#4 gets "sharp"
    syms = [
        EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
        EncodedSymbol("timeSignature/4", ".", ".", ".", ".", "."),
        EncodedSymbol("note_2", "F4", "#", "_", "slurStart", "upper"),
        EncodedSymbol("note_2", "F4", "#", "_", "slurStart_slurStop", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
        EncodedSymbol("note_2", "F4", "#", "_", "slurStop", "upper"),
        EncodedSymbol("note_2", "F4", "#", "_", "_", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
    ]
    xml = generate_xml(XmlGeneratorArguments(), [syms], "")
    assert _acc_list(xml) == [
        ("F4", "1", "1", "sharp"),
        ("F4", "1", "1", None),
        ("F4", "1", "1", None),
        ("F4", "1", "1", "sharp"),
    ]


def test_tie_across_system_break_repeats_accidental() -> None:
    # C major: F#4 tied across newline (system break) repeats sharp on arrival,
    # then a second tied continuation on the same new system does NOT repeat it.
    syms = [
        EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
        EncodedSymbol("timeSignature/4", ".", ".", ".", ".", "."),
        EncodedSymbol("note_1", "F4", "#", "_", "slurStart", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
        EncodedSymbol("newline", ".", ".", ".", ".", "."),
        EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
        EncodedSymbol("note_2", "F4", "#", "_", "slurStart_slurStop", "upper"),
        EncodedSymbol("note_2", "F4", "#", "_", "slurStop", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
    ]
    xml = generate_xml(XmlGeneratorArguments(), [syms], "")
    assert _acc_list(xml) == [
        ("F4", "1", "1", "sharp"),
        ("F4", "1", "1", "sharp"),
        ("F4", "1", "1", None),
    ]

    # G major: F4 natural tied across system break repeats natural; in-key F#4 does not.
    syms_g = [
        EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
        EncodedSymbol("keySignature_1", ".", ".", ".", ".", "."),
        EncodedSymbol("timeSignature/4", ".", ".", ".", ".", "."),
        EncodedSymbol("note_2", "F4", "_", "_", "slurStart", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
        EncodedSymbol("newline", ".", ".", ".", ".", "."),
        EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
        EncodedSymbol("keySignature_1", ".", ".", ".", ".", "."),
        EncodedSymbol("note_2", "F4", "_", "_", "slurStop", "upper"),
        EncodedSymbol("note_2", "F5", "#", "_", "slurStart", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
        EncodedSymbol("newline", ".", ".", ".", ".", "."),
        EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
        EncodedSymbol("keySignature_1", ".", ".", ".", ".", "."),
        EncodedSymbol("note_2", "F5", "#", "_", "slurStop", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
    ]
    xml_g = generate_xml(XmlGeneratorArguments(), [syms_g], "")
    assert _acc_list(xml_g) == [
        ("F4", "1", None, "natural"),
        ("F4", "1", None, "natural"),
        ("F5", "1", "1", None),
        ("F5", "1", "1", None),
    ]


def test_tied_unison_across_system_break_prints_one_accidental() -> None:
    # Two voices tied over the same system break repeat the sharp once, not once per voice.
    part = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><key><fifths>0</fifths></key></attributes>"
        + _xml_note("F", 4, alter=1, dur=4, voice=1, ties=("start",))
        + "</measure><measure number='2'><print new-system='yes'/>"
        + _xml_note("F", 4, alter=1, dur=4, voice=1, ties=("stop",))
        + "<backup><duration>4</duration></backup>"
        + _xml_note("F", 4, alter=1, dur=4, voice=2, ties=("stop",))
        + "</measure></part>"
    )
    add_accidentals(part)
    assert _part_acc_list(part) == [
        ("F4", "1", "1", "sharp"),
        ("F4", "1", "1", "sharp"),
        ("F4", "1", "2", None),
    ]


def test_grand_staff_scopes_accidentals_per_staff() -> None:
    syms = [
        EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
        EncodedSymbol("chord", ".", ".", ".", ".", "."),
        EncodedSymbol("clef_F4", "_", "_", "_", "_", "lower"),
        EncodedSymbol("timeSignature/4", ".", ".", ".", ".", "."),
        EncodedSymbol("note_2", "C4", "#", "_", "_", "upper"),
        EncodedSymbol("chord", ".", ".", ".", ".", "."),
        EncodedSymbol("note_2", "C4", "#", "_", "_", "lower"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
    ]
    xml = generate_xml(XmlGeneratorArguments(), [syms], "")
    accs = _acc_list(xml)
    assert sorted(accs) == [
        ("C4", "1", "1", "sharp"),
        ("C4", "2", "1", "sharp"),
    ]


def test_explicit_natural_lift_emits_natural() -> None:
    syms = [
        EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
        EncodedSymbol("timeSignature/4", ".", ".", ".", ".", "."),
        EncodedSymbol("note_4", "F4", "N", "_", "_", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
    ]
    xml = generate_xml(XmlGeneratorArguments(), [syms], "")
    assert _acc_list(xml) == [("F4", "1", None, "natural")]


def test_explicit_natural_repeated_within_measure_is_kept() -> None:
    # Explicit naturals are printed accidentals: a second N after N is a real precaution,
    # not a duplicate of the implicit accidental state.
    syms = [
        EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
        EncodedSymbol("timeSignature/4", ".", ".", ".", ".", "."),
        EncodedSymbol("note_4", "F4", "N", "_", "_", "upper"),
        EncodedSymbol("note_4", "F4", "N", "_", "_", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
    ]
    xml = generate_xml(XmlGeneratorArguments(), [syms], "")
    assert _acc_list(xml) == [
        ("F4", "1", None, "natural"),
        ("F4", "1", None, "natural"),
    ]


def test_grace_chord_and_grace_before_main_note() -> None:
    # Previous main note F4 at t=0..4; then grace chord {C#5, F#4} at t=4 (second note has
    # <chord/><grace/> and must get onset=4, NOT last_start=0); then main note F4 at t=4
    # must follow the grace F#4 and emit "natural".
    part = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><key><fifths>0</fifths></key></attributes>"
        + _xml_note("F", 4, dur=4)
        + _xml_note("C", 5, alter=1, grace=True)
        + _xml_note("F", 4, alter=1, grace=True, chord=True)
        + _xml_note("F", 4, dur=4)
        + "</measure></part>"
    )
    add_accidentals(part)
    assert _part_acc_list(part) == [
        ("F4", "1", "1", None),
        ("C5", "1", "1", "sharp"),
        ("F4", "1", "1", "sharp"),
        ("F4", "1", "1", "natural"),
    ]

    # Cross-voice grace note (written after <backup>) is still processed before voice 1's main
    # note at the same onset; sequential repeated grace notes do not duplicate the accidental.
    part2 = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><key><fifths>0</fifths></key></attributes>"
        + _xml_note("F", 4, dur=4, voice=1)
        + "<backup><duration>4</duration></backup>"
        + _xml_note("F", 4, alter=1, grace=True, voice=2)
        + _xml_note("F", 4, alter=1, grace=True, voice=2)
        + _xml_note("E", 4, dur=4, voice=2)
        + "</measure></part>"
    )
    add_accidentals(part2)
    assert _part_acc_list(part2) == [
        ("F4", "1", "1", "natural"),
        ("F4", "1", "2", "sharp"),
        ("F4", "1", "2", None),
        ("E4", "1", "2", None),
    ]


def test_simultaneous_false_relation_is_order_independent() -> None:
    # Order A: natural first, sharp second at t=0; then F4 at t=4
    part_a = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><key><fifths>0</fifths></key></attributes>"
        + _xml_note("F", 4, dur=4, voice=1)
        + _xml_note("F", 4, dur=4, voice=1)
        + "<backup><duration>8</duration></backup>"
        + _xml_note("F", 4, alter=1, dur=4, voice=2)
        + "</measure></part>"
    )
    # Order B: sharp first, natural second at t=0; then F4 at t=4
    part_b = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><key><fifths>0</fifths></key></attributes>"
        + _xml_note("F", 4, alter=1, dur=4, voice=2)
        + "<backup><duration>4</duration></backup>"
        + _xml_note("F", 4, dur=4, voice=1)
        + _xml_note("F", 4, dur=4, voice=1)
        + "</measure></part>"
    )
    add_accidentals(part_a)
    add_accidentals(part_b)

    by_voice_a: dict[str, list[str | None]] = {"1": [], "2": []}
    for _, _, v, acc in _part_acc_list(part_a):
        by_voice_a[v].append(acc)
    by_voice_b: dict[str, list[str | None]] = {"1": [], "2": []}
    for _, _, v, acc in _part_acc_list(part_b):
        by_voice_b[v].append(acc)
    assert by_voice_a == {"1": ["natural", "natural"], "2": ["sharp"]}
    assert by_voice_b == by_voice_a

    # Simultaneous false relation between a barline-tied F#4 (same system) and an untied F4:
    # tied note keeps None, untied F4 gets "natural", independent of XML order.
    def _tied_clash(tied_first: bool) -> dict[str, list[str | None]]:
        v1 = _xml_note("F", 4, alter=1, dur=4, voice=1, ties=("stop",))
        v2 = _xml_note("F", 4, dur=4, voice=2)
        m2 = (
            f"{v1}<backup><duration>4</duration></backup>{v2}"
            if tied_first
            else (f"{v2}<backup><duration>4</duration></backup>{v1}")
        )
        p = ET.fromstring(
            "<part id='P1'>"
            "<measure number='1'>"
            "<attributes><divisions>4</divisions><key><fifths>0</fifths></key></attributes>"
            + _xml_note("F", 4, alter=1, dur=4, voice=1, ties=("start",))
            + "</measure>"
            f"<measure number='2'>{m2}</measure></part>"
        )
        add_accidentals(p)
        res: dict[str, list[str | None]] = {"1": [], "2": []}
        for _, _, v, acc in _part_acc_list(p):
            res[v].append(acc)
        return res

    assert _tied_clash(True) == {"1": ["sharp", None], "2": ["natural"]}
    assert _tied_clash(False) == _tied_clash(True)


def test_per_staff_key_signatures_and_attributes_after_backup() -> None:
    # Staff 1 in D major (+2: F#, C#), Staff 2 in F major (-1: Bb) via <key number="...">.
    # Staff 2's <attributes> comes AFTER <backup> and must not overwrite Staff 1's key.
    part = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><staves>2</staves>"
        "<key number='1'><fifths>2</fifths></key></attributes>"
        + _xml_note("F", 4, alter=1, dur=4, voice=1, staff=1)
        + _xml_note("B", 4, alter=-1, dur=4, voice=1, staff=1)
        + "<backup><duration>8</duration></backup>"
        + "<attributes><key number='2'><fifths>-1</fifths></key></attributes>"
        + _xml_note("B", 3, alter=-1, dur=4, voice=5, staff=2)
        + _xml_note("F", 3, alter=1, dur=4, voice=5, staff=2)
        + "</measure></part>"
    )
    add_accidentals(part)
    assert _part_acc_list(part) == [
        ("F4", "1", "1", None),
        ("B4", "1", "1", "flat"),
        ("B3", "2", "5", None),
        ("F3", "2", "5", "sharp"),
    ]

    # Mid-measure key change on staff 2 only does not clear staff 1's active accidental,
    # and oversized <backup> clamps time to 0 so a later <attributes> at t=0 isn't inverted.
    part2 = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><staves>2</staves>"
        "<key><fifths>0</fifths></key></attributes>"
        + _xml_note("G", 4, alter=1, dur=4, voice=1, staff=1)
        + _xml_note("G", 4, alter=1, dur=4, voice=1, staff=1)
        + "<backup><duration>8</duration></backup>"
        + _xml_note("B", 3, dur=4, voice=5, staff=2)
        + "<attributes><key number='2'><fifths>-1</fifths></key></attributes>"
        + _xml_note("B", 3, alter=-1, dur=4, voice=5, staff=2)
        + "</measure></part>"
    )
    add_accidentals(part2)
    assert _part_acc_list(part2) == [
        ("G4", "1", "1", "sharp"),
        ("G4", "1", "1", None),
        ("B3", "2", "5", None),
        ("B3", "2", "5", None),
    ]

    part3 = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><key><fifths>1</fifths></key></attributes>"
        + _xml_note("F", 4, alter=1, dur=2, voice=1, staff=1)
        + "<backup><duration>4</duration></backup>"
        + "<attributes><key><fifths>0</fifths></key></attributes>"
        + _xml_note("F", 4, alter=1, dur=4, voice=2, staff=1)
        + "</measure></part>"
    )
    add_accidentals(part3)
    # Oversized backup clamped to 0 -> both notes are at t=0 under fifths=0; the unison
    # shares the first voice's sharp, so the second voice doesn't print it again
    assert _part_acc_list(part3) == [
        ("F4", "1", "1", "sharp"),
        ("F4", "1", "2", None),
    ]

    # Mid-measure key change at t=4 in voice 1 + <backup> to t=0 for voice 2:
    # voice 2 at t=0 sees the old key (C major), voice 2 at t=4 sees the new key (G major).
    part4 = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><key><fifths>0</fifths></key></attributes>"
        + _xml_note("F", 4, alter=1, dur=4, voice=1, staff=1)
        + "<attributes><key><fifths>1</fifths></key></attributes>"
        + _xml_note("F", 4, alter=1, dur=4, voice=1, staff=1)
        + "<backup><duration>8</duration></backup>"
        + _xml_note("F", 5, alter=1, dur=4, voice=2, staff=1)
        + _xml_note("F", 5, alter=1, dur=4, voice=2, staff=1)
        + "</measure></part>"
    )
    add_accidentals(part4)
    assert _part_acc_list(part4) == [
        ("F4", "1", "1", "sharp"),
        ("F4", "1", "1", None),
        ("F5", "1", "2", "sharp"),
        ("F5", "1", "2", None),
    ]


def test_repeated_accidental_within_measure_is_not_reprinted() -> None:
    # C major: F#4 then F#4 -> only the first prints sharp; F#4 then F4 -> natural is printed
    syms = [
        EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
        EncodedSymbol("timeSignature/4", ".", ".", ".", ".", "."),
        EncodedSymbol("note_4", "F4", "#", "_", "_", "upper"),
        EncodedSymbol("note_4", "F4", "#", "_", "_", "upper"),
        EncodedSymbol("note_4", "F4", "_", "_", "_", "upper"),
        EncodedSymbol("barline", ".", ".", ".", ".", "."),
    ]
    xml = generate_xml(XmlGeneratorArguments(), [syms], "")
    assert _acc_list(xml) == [
        ("F4", "1", "1", "sharp"),
        ("F4", "1", "1", None),
        ("F4", "1", None, "natural"),
    ]


def test_simultaneous_unison_same_alter_prints_one_accidental() -> None:
    # Voice 1 carries F#4 with an explicit accidental; voice 2 sounds the same F#4 at the
    # same onset and must not get a second <accidental>sharp</accidental>.
    part = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><key><fifths>0</fifths></key></attributes>"
        "<note><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch>"
        "<duration>4</duration><voice>1</voice><type>quarter</type>"
        "<accidental>sharp</accidental><staff>1</staff></note>"
        "<backup><duration>4</duration></backup>"
        + _xml_note("F", 4, alter=1, dur=4, voice=2)
        + "</measure></part>"
    )
    add_accidentals(part)
    assert _part_acc_list(part) == [
        ("F4", "1", "1", "sharp"),
        ("F4", "1", "2", None),
    ]


def test_existing_accidental_in_unison_is_order_independent() -> None:
    # Whichever voice comes first in the XML, an <accidental> already present in the
    # unison is shared, not duplicated on the other voice.
    def _unison(existing_voice: int) -> ET.Element:
        voices = {}
        for voice in (1, 2):
            if voice == existing_voice:
                voices[voice] = (
                    f"<note><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch>"
                    f"<duration>4</duration><voice>{voice}</voice><type>quarter</type>"
                    "<accidental>sharp</accidental><staff>1</staff></note>"
                )
            else:
                voices[voice] = _xml_note("F", 4, alter=1, dur=4, voice=voice)
        return ET.fromstring(
            "<part id='P1'><measure number='1'>"
            "<attributes><divisions>4</divisions><key><fifths>0</fifths></key></attributes>"
            + voices[1]
            + "<backup><duration>4</duration></backup>"
            + voices[2]
            + "</measure></part>"
        )

    for existing_voice in (1, 2):
        part = _unison(existing_voice)
        add_accidentals(part)
        accidentals = [acc for *_, acc in _part_acc_list(part)]
        assert accidentals.count("sharp") == 1


def test_two_explicit_naturals_in_unison_print_one_accidental() -> None:
    # Both voices of a unison carry lift "N": the first inserts the natural, the second
    # must see that printed accidental even though neither note had one before the loop.
    part = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><key><fifths>0</fifths></key></attributes>"
        "<note homr-explicit-natural='1'>"
        "<pitch><step>F</step><octave>4</octave></pitch>"
        "<duration>4</duration><voice>1</voice><type>quarter</type><staff>1</staff></note>"
        "<backup><duration>4</duration></backup>"
        "<note homr-explicit-natural='1'>"
        "<pitch><step>F</step><octave>4</octave></pitch>"
        "<duration>4</duration><voice>2</voice><type>quarter</type><staff>1</staff></note>"
        "</measure></part>"
    )
    add_accidentals(part)
    assert _part_acc_list(part) == [
        ("F4", "1", "1", "natural"),
        ("F4", "1", "2", None),
    ]


def test_microtone_alter_is_left_intact() -> None:
    # G major expects F#; an F half-sharp is not a natural and must stay without accidental
    part = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><key><fifths>1</fifths></key></attributes>"
        + _xml_note("F", 4, alter=0.5, dur=4)
        + "</measure></part>"
    )
    add_accidentals(part)
    note = part.find("./measure/note")
    assert note is not None
    assert note.findtext("pitch/alter") == "0.5"
    assert note.find("accidental") is None


def test_grace_before_chord_updates_accidental_state() -> None:
    # Grace F# before a main chord {F natural, A}: the chord's F must print natural
    part = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><key><fifths>0</fifths></key></attributes>"
        + _xml_note("F", 4, alter=1, grace=True)
        + _xml_note("F", 4, dur=4)
        + _xml_note("A", 4, dur=4, chord=True)
        + "</measure></part>"
    )
    add_accidentals(part)
    assert _part_acc_list(part) == [
        ("F4", "1", "1", "sharp"),
        ("F4", "1", "1", "natural"),
        ("A4", "1", "1", None),
    ]
