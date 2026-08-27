# flake8: noqa: S101

import itertools
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from homr import constants
from homr.simple_logging import eprint
from homr.transformer.vocabulary import (
    EncodedSymbol,
    SymbolDuration,
    empty,
    nonote,
    sort_token_chords,
)


class ConversionState:
    def __init__(self, division: int, nominator: Fraction):
        self.beats = 4 * constants.duration_of_quarter
        self.division = division
        self.nominator = nominator
        self.tremolo_state = "stop"
        self.volta_number = 1
        self.last_volta_measure = -10

    def start_volta(self, measure_no: int) -> int:
        if measure_no == self.last_volta_measure + 1:
            self.volta_number += 1
        else:
            self.volta_number = 1
        return self.volta_number

    def stop_volta(self, measure_no: int) -> int:
        self.last_volta_measure = measure_no
        return self.volta_number

    def toggle_tremolo_state(self) -> str:
        if self.tremolo_state == "start":
            self.tremolo_state = "stop"
        else:
            self.tremolo_state = "start"
        return self.tremolo_state


class SymbolChord:
    def __init__(self, symbols: list[EncodedSymbol], tuplet_mark: str = "") -> None:
        self.symbols = symbols
        self.tuplet_mark = tuplet_mark

    def __str__(self) -> str:
        return str.join("&", [str(s) for s in self.symbols])

    def __repr__(self) -> str:
        return str(self)

    def is_barline(self) -> bool:
        if len(self.symbols) == 0:
            return False
        first_rhythm = self.symbols[0].rhythm
        return "barline" in first_rhythm or "repeat" in first_rhythm

    def get_duration(self) -> Fraction:
        notes_rests = [
            s.get_duration().fraction for s in self.symbols if s.rhythm.startswith(("note", "rest"))
        ]
        if len(notes_rests) == 0:
            return Fraction(0)
        return min(notes_rests)

    def into_positions(self) -> list["SymbolChord"]:
        upper = []
        lower = []
        lower_is_only_rest = True
        for symbol in self.symbols:
            if symbol.position == "upper":
                upper.append(symbol)
            else:
                lower.append(symbol)
                lower_is_only_rest = lower_is_only_rest and symbol.rhythm.startswith("rest")
        chords = (
            SymbolChord(upper, self.tuplet_mark),
            SymbolChord(lower, self.tuplet_mark),
        )
        if lower_is_only_rest:
            chords = (chords[1], chords[0])
        return [chord for chord in chords if len(chord.symbols) > 0]


class XmlGeneratorArguments:
    def __init__(
        self, large_page: bool | None = None, metronome: int | None = None, tempo: int | None = None
    ):
        self.large_page = large_page
        self.metronome = metronome
        self.tempo = tempo


def build_identification() -> ET.Element:
    ident = ET.Element("identification")
    enc = ET.SubElement(ident, "encoding")
    ET.SubElement(enc, "software").text = "homr"
    return ident


def generate_xml(
    args: XmlGeneratorArguments, staffs: list[list[EncodedSymbol]], title: str
) -> ET.Element:
    root = ET.Element("score-partwise", version="4.0")
    root.append(build_work(title))
    root.append(build_identification())
    root.append(build_defaults(args))
    has_two_staves_by_part = [_voice_has_two_staves(staff) for staff in staffs]
    root.append(build_part_list(has_two_staves_by_part))
    for index, staff in enumerate(staffs):
        root.append(build_part(args, staff, index, has_two_staves_by_part[index]))
    return root


def xml_to_string(element: ET.Element) -> str:
    return ET.tostring(element, encoding="unicode")


def _voice_has_two_staves(voice: list[EncodedSymbol]) -> bool:
    """True if any symbol uses the lower staff (e.g. piano left hand / bass clef)."""
    return any(s.position == "lower" for s in voice)


def build_part(
    args: XmlGeneratorArguments, voice: list[EncodedSymbol], index: int, has_two_staves: bool
) -> ET.Element:
    part = ET.Element("part", id=get_part_id(index))
    is_first_part = index == 0
    for measure in build_measures(args, voice, is_first_part, has_two_staves):
        part.append(measure)
    return part


def build_measures(
    args: XmlGeneratorArguments,
    voice: list[EncodedSymbol],
    is_first_part: bool,
    has_two_staves: bool = False,
) -> list[ET.Element]:
    measure_cursor = Fraction(0)
    measure_was_planned = False

    def close_current_measure() -> None:
        nonlocal measure_cursor, measure_was_planned
        if not measure_was_planned:
            # Planned measures carry their voicing from the plan itself; the
            # overlap-based rebalancing would tear held/moving lines apart.
            rebalance_measure_voices(current_measure)
        measures.append(current_measure)
        measure_cursor = Fraction(0)
        measure_was_planned = False

    measure_number = 1
    groups = add_tuplet_start_stop(group_into_chords(voice))
    division, nominator = find_division_and_time_signature_nominator(groups)
    state = ConversionState(division, nominator)
    voice_plans = _plan_voice_repairs(groups, nominator)
    measures: list[ET.Element] = []
    current_measure = ET.Element("measure", number=str(measure_number))
    first_attributes = build_or_get_attributes(current_measure, None)
    ET.SubElement(first_attributes, "divisions").text = str(division // 4)
    if has_two_staves:
        ET.SubElement(first_attributes, "staves").text = "2"
        ET.SubElement(first_attributes, "part-symbol").text = "brace"
    if is_first_part:
        direction = build_add_time_direction(args)
        if direction is not None:
            current_measure.append(direction)
    attributes: ET.Element | None = first_attributes
    for group_no, group in enumerate(groups):
        symbol = group.symbols[0]
        rhythm = symbol.rhythm
        last_attributes = attributes
        attributes = None
        if rhythm.startswith(("note", "rest")):
            if len(group.symbols) == 1 and rhythm.endswith("m"):
                attributes = build_or_get_attributes(current_measure, last_attributes)
                build_multi_measure_rest(symbol, attributes)
            elif group_no in voice_plans:
                measure_was_planned = True
                for planned in voice_plans[group_no]:
                    delta = planned.attack - measure_cursor
                    if delta > 0:
                        current_measure.append(build_forward(delta, state))
                    elif delta < 0:
                        current_measure.append(build_backup(-delta, state))
                    for note_xml in build_note_chord(planned.part, state, planned.advance):
                        if note_xml.tag == "note":
                            voice_el = note_xml.find("voice")
                            duration_el = note_xml.find("duration")
                            if voice_el is not None:
                                xml_voice = planned.voice
                                if duration_el is not None and duration_el.text:
                                    duration = Fraction(int(duration_el.text), state.division)
                                    xml_voice = planned.voice_by_duration.get(duration, xml_voice)
                                voice_el.text = str(xml_voice)
                        current_measure.append(note_xml)
                    measure_cursor = planned.attack + planned.advance
            else:
                staff_positions = group.into_positions()
                for pos_no, staff_pos in enumerate(staff_positions):
                    chord_duration = (
                        group.get_duration() if pos_no == len(staff_positions) - 1 else Fraction(0)
                    )
                    for note_xml in build_note_chord(staff_pos, state, chord_duration):
                        current_measure.append(note_xml)
            continue
        if rhythm == "newline":
            is_last_measure = group_no == len(groups) - 1
            if not is_last_measure:
                ET.SubElement(current_measure, "print", attrib={"new-system": "yes"})
        elif rhythm.startswith("clef"):
            attributes = build_or_get_attributes(current_measure, last_attributes, force_new=True)
            for should_be_clef in group.symbols:
                if should_be_clef.rhythm.startswith("clef"):
                    build_clef(should_be_clef, attributes)
        elif rhythm.startswith("keySignature"):
            attributes = build_or_get_attributes(current_measure, last_attributes)
            build_key(symbol, attributes)
        elif rhythm.startswith("timeSignature"):
            attributes = build_or_get_attributes(current_measure, last_attributes)
            build_time_signature(symbol, attributes, state)
        elif "barline" in rhythm:
            if rhythm != "barline":
                barline = build_or_get_barline(current_measure, "right")
                build_barline_style(symbol, barline)

            close_current_measure()
            measure_number += 1
            current_measure = ET.Element("measure", number=str(measure_number))
        elif rhythm == "repeatStart":
            close_current_measure()
            measure_number += 1
            current_measure = ET.Element("measure", number=str(measure_number))

            barline = build_or_get_barline(current_measure, "right")
            build_repeat(symbol, barline)
        elif rhythm == "repeatEnd":
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(symbol, barline)

            close_current_measure()
            measure_number += 1
            current_measure = ET.Element("measure", number=str(measure_number))
        elif rhythm == "repeatEndStart":
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(EncodedSymbol("repeatEnd"), barline)

            close_current_measure()
            measure_number += 1
            current_measure = ET.Element("measure", number=str(measure_number))
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(EncodedSymbol("repeatStart"), barline)
        elif rhythm.startswith("voltaStart"):
            volta_number = state.start_volta(measure_number)
            barline = build_or_get_barline(current_measure, "left")
            build_barline_ending(symbol, barline, volta_number)
        elif rhythm.startswith(("voltaStop", "voltaDiscontinue")):
            volta_number = state.stop_volta(measure_number)
            barline = build_or_get_barline(current_measure, "right")
            build_barline_ending(symbol, barline, volta_number)
        else:
            eprint("Symbol isn't supported yet ", symbol)

    if len(list(current_measure)) > 0:
        close_current_measure()
    if first_attributes.find("time") is None:
        time_el = ET.SubElement(first_attributes, "time")
        beats = max(int(state.nominator * 4), 1)
        ET.SubElement(time_el, "beats").text = str(beats)
        ET.SubElement(time_el, "beat-type").text = "4"
    return measures


def build_work(title_text: str) -> ET.Element:
    work = ET.Element("work")
    ET.SubElement(work, "work-title").text = title_text
    return work


def build_defaults(args: XmlGeneratorArguments) -> ET.Element:
    defaults = ET.Element("defaults")
    if args.large_page:
        page_layout = ET.SubElement(defaults, "page-layout")
        ET.SubElement(page_layout, "page-height").text = "300"
        ET.SubElement(page_layout, "page-width").text = "110"
    return defaults


def get_part_id(index: int) -> str:
    return "P" + str(index + 1)


def _part_metadata(has_two_staves: bool) -> tuple[str, str, str, int]:
    """Return (part_name, instrument_name, instrument_sound, midi_program) for a part.

    We classify by staff layout only:
    - single-staff parts -> Voice
    - two-staff parts -> Piano
    midi_program is 1-based (MusicXML: 1-128; 1 = Acoustic Grand Piano, 54 = Voice Oohs).
    """
    if has_two_staves:
        return ("Piano", "Piano", "keyboard.piano", 1)
    return ("Voice", "Voice", "voice", 54)


def build_part_list(has_two_staves_by_part: list[bool]) -> ET.Element:
    part_list = ET.Element("part-list")
    for part, has_two_staves in enumerate(has_two_staves_by_part):
        part_id = get_part_id(part)
        part_name_str, instrument_name_str, instrument_sound_str, midi_program = _part_metadata(
            has_two_staves
        )
        score_part = ET.SubElement(part_list, "score-part", id=part_id)
        ET.SubElement(score_part, "part-name").text = part_name_str
        score_instrument = ET.SubElement(score_part, "score-instrument", id=part_id + "-I1")
        ET.SubElement(score_instrument, "instrument-name").text = instrument_name_str
        ET.SubElement(score_instrument, "instrument-sound").text = instrument_sound_str
        midi_instrument = ET.SubElement(score_part, "midi-instrument", id=part_id + "-I1")
        ET.SubElement(midi_instrument, "midi-channel").text = str(part + 1)
        ET.SubElement(midi_instrument, "midi-program").text = str(midi_program)
        ET.SubElement(midi_instrument, "volume").text = "100"
        ET.SubElement(midi_instrument, "pan").text = "0"
    return part_list


def build_or_get_attributes(
    measure: ET.Element, last_attributes: ET.Element | None, force_new: bool = False
) -> ET.Element:
    if last_attributes is not None and not force_new:
        return last_attributes
    return ET.SubElement(measure, "attributes")


def build_or_get_barline(measure: ET.Element, location: str) -> ET.Element:
    for child in measure:
        if child.tag == "barline" and child.get("location") == location:
            return child
    return ET.SubElement(measure, "barline", location=location)


def build_key(model_key: EncodedSymbol, attributes: ET.Element) -> None:
    key = ET.SubElement(attributes, "key")
    ET.SubElement(key, "fifths").text = model_key.rhythm.split("_")[1]


def get_staff(symbol: EncodedSymbol) -> int:
    return 2 if symbol.position == "lower" else 1


def get_xml_voice(staff_num: int, rhythmic_layer: int) -> int:
    """Build a stable MusicXML voice number per staff and rhythmic layer.

    Voice numbers are part-global in MusicXML, so using only the staff number can merge
    independent layers on the same staff. Reserve 4 voices per staff:
    staff 1 -> voices 1-4, staff 2 -> voices 5-8.
    """
    return (staff_num - 1) * 4 + rhythmic_layer + 1


@dataclass
class TimedNoteEvent:
    staff_num: int
    start: int
    end: int
    notes: list[ET.Element]


def rebalance_measure_voices(measure: ET.Element) -> None:
    """Assign stable non-overlapping voices per staff for a whole measure."""
    timed_events: list[TimedNoteEvent] = []
    current_time = 0
    last_note_start = 0
    for child in measure:
        if child.tag == "backup":
            dur = child.find("duration")
            if dur is not None:
                current_time -= int(dur.text)  # type: ignore[arg-type]
            continue
        if child.tag == "forward":
            dur = child.find("duration")
            if dur is not None:
                current_time += int(dur.text)  # type: ignore[arg-type]
            continue
        if child.tag != "note":
            continue

        dur_el = child.find("duration")
        duration = int(dur_el.text) if dur_el is not None else 0  # type: ignore[arg-type]
        staff_el = child.find("staff")
        staff_num = int(staff_el.text) if staff_el is not None else 1  # type: ignore[arg-type]
        is_chord_tone = child.find("chord") is not None
        start = last_note_start if is_chord_tone else current_time
        end = start + duration
        if is_chord_tone and (
            len(timed_events) > 0
            and timed_events[-1].staff_num == staff_num
            and timed_events[-1].start == start
            and timed_events[-1].end == end
        ):
            timed_events[-1].notes.append(child)
        elif is_chord_tone:
            timed_events.append(TimedNoteEvent(staff_num, start, end, [child]))
        else:
            last_note_start = start
            current_time += duration
            timed_events.append(TimedNoteEvent(staff_num, start, end, [child]))

    by_staff: dict[int, list[TimedNoteEvent]] = defaultdict(list)
    for event in timed_events:
        by_staff[event.staff_num].append(event)

    for staff_num, events in by_staff.items():
        sorted_events = sorted(events, key=lambda e: (e.start, e.end))
        active: list[tuple[int, int]] = []
        for event in sorted_events:
            active = [
                (active_end, voice_no)
                for active_end, voice_no in active
                if active_end > event.start
            ]
            used_voices = {voice_no for _, voice_no in active}
            voice_no = 1
            while voice_no in used_voices:
                voice_no += 1
            active.append((event.end, voice_no))
            xml_voice = str(get_xml_voice(staff_num, voice_no - 1))
            for note in event.notes:
                voice_el = note.find("voice")
                if voice_el is not None:
                    voice_el.text = xml_voice


def build_clef(model_clef: EncodedSymbol, attributes: ET.Element) -> None:
    sign_and_line = model_clef.rhythm.split("_")[1]
    clef = ET.SubElement(attributes, "clef", number=str(get_staff(model_clef)))
    ET.SubElement(clef, "sign").text = sign_and_line[0]
    ET.SubElement(clef, "line").text = sign_and_line[1]


def build_time_signature(
    model_time_signature: EncodedSymbol, attributes: ET.Element, state: ConversionState
) -> None:
    time = ET.SubElement(attributes, "time")
    denominator = model_time_signature.rhythm.split("/")[1]
    beats = max(int(state.nominator * int(denominator)), 1)
    ET.SubElement(time, "beats").text = str(beats)
    ET.SubElement(time, "beat-type").text = denominator
    state.beats = beats


def build_barline_style(barline: EncodedSymbol, xml: ET.Element) -> None:
    style_value = "heavy-heavy" if barline.rhythm == "bolddoublebarline" else "light-light"
    ET.SubElement(xml, "bar-style").text = style_value


def build_barline_ending(volta: EncodedSymbol, xml: ET.Element, volta_number: int) -> None:
    if volta.rhythm.startswith("voltaStart"):
        type_ = "start"
    elif volta.rhythm.startswith("voltaStop"):
        type_ = "stop"
    elif volta.rhythm.startswith("voltaDiscontinue"):
        type_ = "discontinue"
    else:
        raise ValueError("Unknown ending " + str(volta))
    ET.SubElement(xml, "ending", type=type_, number=str(volta_number))


def build_repeat(barline: EncodedSymbol, xml: ET.Element) -> None:
    if xml.find("repeat") is not None:
        eprint("barline already has a repeat")
        return
    direction = "forward" if barline.rhythm == "repeatStart" else "backward"
    ET.SubElement(xml, "repeat", direction=direction)


LIFT_TO_ALTER = {
    "N": 0,
    "#": 1,
    "##": 2,
    "b": -1,
    "bb": -2,
}

DURATION_NAMES = {
    0: "breve",
    1: "whole",
    2: "half",
    4: "quarter",
    8: "eighth",
    16: "16th",
    32: "32nd",
    64: "64th",
    128: "128th",
}


def build_articulations(
    note: ET.Element, articualations: str, tuplet_mark: str, state: ConversionState
) -> None:
    notation = ET.SubElement(note, "notations")

    xml_articulations: list[ET.Element] = []
    xml_ornaments: list[ET.Element] = []

    for articulation in articualations.split("_"):
        if articulation == "":
            continue
        elif articulation == nonote:
            eprint("WARNING note without valid articulation", articualations)
        elif articulation == "fermata":
            ET.SubElement(notation, "fermata")
        elif articulation == "arpeggiate":
            ET.SubElement(notation, "arpeggiate")
        elif articulation == "accent":
            xml_articulations.append(ET.Element("accent"))
        elif articulation == "staccato":
            xml_articulations.append(ET.Element("staccato"))
        elif articulation == "staccatissimo":
            xml_articulations.append(ET.Element("staccatissimo"))
        elif articulation == "tenuto":
            xml_articulations.append(ET.Element("tenuto"))
        elif articulation == "tremolo":
            el = ET.Element("tremolo", type=state.toggle_tremolo_state())
            el.text = "3"
            xml_ornaments.append(el)
        elif articulation == "trill":
            xml_ornaments.append(ET.Element("trill-mark"))
        elif articulation == "breathMark":
            xml_articulations.append(ET.Element("breath-mark"))
        elif articulation == "turn":
            xml_ornaments.append(ET.Element("inverted-turn"))
        elif articulation == "caesura":
            xml_articulations.append(ET.Element("caesura"))
        elif articulation == "doit":
            xml_articulations.append(ET.Element("doit"))
        elif articulation == "slurStart":
            ET.SubElement(notation, "slur", type="start")
        elif articulation == "slurStop":
            ET.SubElement(notation, "slur", type="stop")
        elif articulation == "tieStart":
            ET.SubElement(notation, "tied", type="start")
        elif articulation == "tieStop":
            ET.SubElement(notation, "tied", type="stop")
        else:
            raise ValueError("Unsupported articulation " + articulation)

    if tuplet_mark != "":
        ET.SubElement(notation, "tuplet", type=tuplet_mark)

    if xml_articulations:
        parent = ET.SubElement(notation, "articulations")
        for child in xml_articulations:
            parent.append(child)

    if xml_ornaments:
        parent = ET.SubElement(notation, "ornaments")
        for child in xml_ornaments:
            parent.append(child)


def build_slurs(note: ET.Element, slurs: str, slur_number: int) -> None:
    notation = note.find("notations")
    if notation is None:
        notation = ET.SubElement(note, "notations")

    if slurs in {"_", ""}:
        pass
    elif slurs == nonote:
        eprint("WARNING note without valid articulation", slurs)
    elif slurs == "slurStart":
        ET.SubElement(notation, "slur", type="start", number=str(slur_number))
    elif slurs == "slurStop":
        ET.SubElement(notation, "slur", type="stop", number=str(slur_number))
    elif slurs == "slurStart_slurStop":
        ET.SubElement(notation, "slur", type="stop", number=str(slur_number))
        ET.SubElement(notation, "slur", type="start", number=str(slur_number))
    else:
        raise ValueError("Unsupported slur " + slurs)


def build_note_or_rest(
    model_note: EncodedSymbol,
    rhythmic_layer: int,
    is_chord: bool,
    state: ConversionState,
    tuplet_mark: str,
) -> ET.Element:
    note = ET.Element("note")
    if is_chord:
        ET.SubElement(note, "chord")
    model_pitch = model_note.pitch
    model_duration = model_note.get_duration()

    if "G" in model_note.rhythm:
        ET.SubElement(note, "grace")

    if model_pitch == empty:
        if model_duration.fraction.numerator == 0:
            ET.SubElement(note, "rest", measure="yes")
        else:
            ET.SubElement(note, "rest")
    elif model_pitch == nonote:
        eprint("WARNING note without pitch", model_note)
        ET.SubElement(note, "rest")
    else:
        pitch = ET.SubElement(note, "pitch")
        ET.SubElement(pitch, "step").text = model_pitch[0]
        ET.SubElement(pitch, "octave").text = model_pitch[1]
        if model_note.lift == nonote:
            eprint("WARNING note with invalid lift", model_note)
        elif model_note.lift != empty:
            ET.SubElement(pitch, "alter").text = str(LIFT_TO_ALTER[model_note.lift])

    if "G" in model_note.rhythm:
        base_duration = model_duration.kern
        ET.SubElement(note, "type").text = DURATION_NAMES[base_duration]
    elif model_duration.fraction.numerator > 0:
        base_duration = 1 if model_duration.kern == 0 else model_duration.kern
        ET.SubElement(note, "duration").text = str(int(model_duration.fraction * state.division))
        ET.SubElement(note, "type").text = DURATION_NAMES[base_duration]
    else:
        ET.SubElement(note, "duration").text = str(state.beats)
        ET.SubElement(note, "type").text = DURATION_NAMES[0]

    for _ in range(model_duration.dots):
        ET.SubElement(note, "dot")

    if model_duration.actual_notes != model_duration.normal_notes:
        time_mod = ET.SubElement(note, "time-modification")
        ET.SubElement(time_mod, "actual-notes").text = str(model_duration.actual_notes)
        ET.SubElement(time_mod, "normal-notes").text = str(model_duration.normal_notes)

    staff_num = get_staff(model_note)
    slur_number = staff_num
    ET.SubElement(note, "voice").text = str(get_xml_voice(staff_num, rhythmic_layer))
    ET.SubElement(note, "staff").text = str(staff_num)

    build_articulations(note, model_note.articulation, tuplet_mark, state)
    build_slurs(note, model_note.slur, slur_number)

    return note


def build_multi_measure_rest(symbol: EncodedSymbol, attributes: ET.Element) -> None:
    if attributes.find("measure-style") is not None:
        eprint("Measure already has a multi rest")
        return
    duration = int(symbol.rhythm.split("_")[1].replace("m", ""))
    style = ET.SubElement(attributes, "measure-style")
    ET.SubElement(style, "multiple-rest").text = str(duration)


def build_backup(duration: Fraction, state: ConversionState) -> ET.Element:
    assert duration > Fraction(0), "Backup duration must be positive"
    backup = ET.Element("backup")
    ET.SubElement(backup, "duration").text = str(int(duration * state.division))
    return backup


def build_forward(duration: Fraction, state: ConversionState) -> ET.Element:
    assert duration > Fraction(0), "Forward duration must be positive"
    forward = ET.Element("forward")
    ET.SubElement(forward, "duration").text = str(int(duration * state.division))
    return forward


class PlannedPart:
    """One staff's share of a token chord with an absolute attack time."""

    def __init__(self, part: "SymbolChord", attack: Fraction, advance: Fraction) -> None:
        self.part = part
        self.attack = attack
        self.advance = advance
        self.voice = 1
        self.voice_by_duration: dict[Fraction, int] = {}


def _assign_plan_voices(plan: dict[int, list[PlannedPart]]) -> None:
    """Voice numbers from the plan's own reading: notes that flow with the
    staff cursor are the moving line and share the staff's first voice; a
    note sounding past its part's cursor advance is a held voice layered
    into the next free voice for as long as it sounds. Rebalancing by
    overlap alone would instead splice the moving line's tail onto the held
    voice once it ends. Held notes are decided per duration within a part:
    a chord can carry a moving note and a held note in one token chord."""
    by_staff: dict[int, list[PlannedPart]] = defaultdict(list)
    for group_no in sorted(plan):
        for planned in plan[group_no]:
            by_staff[get_staff(planned.part.symbols[0])].append(planned)
    for staff, staff_parts in by_staff.items():
        held_active: list[tuple[Fraction, int]] = []  # (sounding end, layer)
        for planned in staff_parts:  # already in attack order
            held_active = [(end, layer) for end, layer in held_active if end > planned.attack]
            planned.voice = get_xml_voice(staff, 0)
            planned.voice_by_duration = {}
            if planned.advance <= 0:
                continue
            for duration in sorted(_group_notes(planned.part.symbols)):
                if duration <= planned.advance:
                    continue
                used = {layer for _, layer in held_active}
                layer = 1
                while layer in used:
                    layer += 1
                held_active.append((planned.attack + duration, layer))
                planned.voice_by_duration[duration] = get_xml_voice(staff, layer)


def _plan_measure_streams(
    measure_groups: list[tuple[int, "SymbolChord"]], expected: Fraction
) -> dict[int, list[PlannedPart]] | None:
    """
    Re-time a measure whose duration doesn't add up by giving each staff its
    own time cursor instead of one shared cursor.

    The token stream serializes simultaneous voices in raster order, so when
    the model fails to chord a second voice with its neighbors, the shared
    cursor treats concurrent events as sequential and the measure overflows.
    A plan is accepted only when every staff closes exactly on the expected
    duration with attack times non-decreasing in token order (the raster
    property).

    Candidates are ranked by how many notes they shrink. That ranking is a
    prior, not arithmetic — a held voice is usually a single long note, so
    when several interpretations close the checksum, the one that changes
    the least is taken. The best-ranked candidates must also agree on the
    sound (attacks, pitches, durations); when equally simple interpretations
    sound different, or the alternative space is too large to enumerate
    exhaustively, the arithmetic cannot decide and the caller keeps the
    shared-cursor interpretation.
    """
    candidates = _attempt_stream_plans(measure_groups, expected)
    if not candidates:
        return None
    fewest_shrunk = min(shrunk for _, shrunk in candidates)
    top = [plan for plan, shrunk in candidates if shrunk == fewest_shrunk]
    if len({_plan_sound(plan) for plan in top}) > 1:
        return None
    _assign_plan_voices(top[0])
    return top[0]


def _plan_sound(plan: dict[int, list[PlannedPart]]) -> tuple:
    """What a plan sounds like: attack, pitch and printed duration of every
    symbol. Staff assignment and cursor advances are voicing, not sound."""
    events = []
    for parts in plan.values():
        for planned in parts:
            for symbol in planned.part.symbols:
                events.append((planned.attack, symbol.rhythm, symbol.pitch, symbol.lift))
    return tuple(sorted(events))


def _shrink_alternatives(
    indices: list[int], base: list[Fraction], overshoot: Fraction
) -> list[tuple[dict[int, Fraction], int]] | None:
    """All ways to shed exactly `overshoot` from one staff by shrinking
    notes' cursor advances, as (advance reduction by index, number of shrunk
    notes). A note can give its advance away down to the shortest advance on
    the staff; long notes are the held-voice candidates.

    The enumeration is exhaustive on the duration grid the staff spans, so
    the caller's uniqueness check covers the whole alternative space. None
    means the space is too large to enumerate; the caller must then treat
    the measure as undecidable instead of judging from a sample."""
    positive = [i for i in indices if base[i] > 0]
    if not positive:
        return []
    floor = min(base[i] for i in positive)
    shrinkable = [i for i in positive if base[i] > floor]
    capacities = [base[i] - floor for i in shrinkable]
    if sum(capacities, start=Fraction(0)) < overshoot:
        return []

    grid = math.lcm(overshoot.denominator, *(c.denominator for c in capacities))
    units = int(overshoot * grid)
    cap_units = [int(c * grid) for c in capacities]

    max_alternatives = 256
    alternatives: list[tuple[dict[int, Fraction], int]] = []

    def distribute(note: int, remaining: int, gives: list[int]) -> bool:
        """False when the alternative space exceeds the enumeration cap."""
        if remaining == 0:
            reductions = {shrinkable[k]: Fraction(g, grid) for k, g in enumerate(gives) if g > 0}
            alternatives.append((reductions, len(reductions)))
            return len(alternatives) <= max_alternatives
        if note == len(cap_units) or remaining > sum(cap_units[note:]):
            return True
        for give in range(min(remaining, cap_units[note]) + 1):
            if not distribute(note + 1, remaining - give, [*gives, give]):
                return False
        return True

    if not distribute(0, units, []):
        return None
    return alternatives


def _attempt_stream_plans(
    measure_groups: list[tuple[int, "SymbolChord"]],
    expected: Fraction,
) -> list[tuple[dict[int, list[PlannedPart]], int]] | None:
    """Candidate interpretations of a measure: replay each staff on its own
    cursor, shrinking long (held-voice) notes' advances when a staff
    overruns, and validate the checksum and raster constraints. Each valid
    choice of shrunk notes is one candidate, paired with its shrink count
    for ranking. None means the alternative space is too large to enumerate
    exhaustively, so uniqueness cannot be certified."""
    parts_info: list[tuple[int, SymbolChord, int]] = []
    for group_no, group in measure_groups:
        for part in group.into_positions():
            staff = get_staff(part.symbols[0])
            parts_info.append((group_no, part, staff))

    base = [part.get_duration() for _, part, _ in parts_info]
    per_staff: list[list[tuple[dict[int, Fraction], int]]] = []
    for staff in sorted({staff for _, _, staff in parts_info}):
        indices = [i for i, (_, _, s) in enumerate(parts_info) if s == staff]
        total = sum((base[i] for i in indices), start=Fraction(0))
        if total == expected:
            per_staff.append([({}, 0)])
            continue
        if total < expected:
            # Notes are missing outright; re-timing can't reconstruct them
            return []
        alternatives = _shrink_alternatives(indices, base, total - expected)
        if alternatives is None:
            return None
        if not alternatives:
            return []
        per_staff.append(alternatives)

    max_combinations = 256
    if math.prod(len(alternatives) for alternatives in per_staff) > max_combinations:
        return None
    plans: list[tuple[dict[int, list[PlannedPart]], int]] = []
    for combination in itertools.product(*per_staff):
        gives: dict[int, Fraction] = {}
        shrunk = 0
        for staff_gives, staff_shrunk in combination:
            gives.update(staff_gives)
            shrunk += staff_shrunk
        advances = [base[i] - gives.get(i, Fraction(0)) for i in range(len(base))]
        plan = _validate_stream_plan(parts_info, advances, expected)
        if plan is not None:
            plans.append((plan, shrunk))
    return plans


def _validate_stream_plan(
    parts_info: list[tuple[int, "SymbolChord", int]],
    advances: list[Fraction],
    expected: Fraction,
) -> dict[int, list[PlannedPart]] | None:
    cursors: dict[int, Fraction] = defaultdict(Fraction)
    last_attack = Fraction(0)
    plan: dict[int, list[PlannedPart]] = defaultdict(list)
    for group_no, group_parts in itertools.groupby(enumerate(parts_info), key=lambda p: p[1][0]):
        parts = list(group_parts)
        # Chorded parts across staves attack together
        attack = max(cursors[staff] for _, (_, _, staff) in parts)
        if attack < last_attack:
            return None
        last_attack = attack
        for i, (_, part, staff) in parts:
            plan[group_no].append(PlannedPart(part, attack, advances[i]))
            cursors[staff] = attack + advances[i]

    if any(cursor != expected for cursor in cursors.values()):
        return None
    return dict(plan)


def _plan_voice_repairs(
    groups: list["SymbolChord"], expected: Fraction
) -> dict[int, list[PlannedPart]]:
    """Find measures whose duration doesn't match the expectation and try to
    repair them by re-timing their voices (see _plan_measure_streams)."""
    plans: dict[int, list[PlannedPart]] = {}
    if expected <= 0:
        return plans
    measure_no = 1
    measure_groups: list[tuple[int, SymbolChord]] = []
    has_multirest = False

    def plan_current_measure() -> None:
        if has_multirest or len(measure_groups) < 2:
            return
        duration = sum((group.get_duration() for _, group in measure_groups), start=Fraction(0))
        if duration == expected:
            return
        plan = _plan_measure_streams(measure_groups, expected)
        if plan is not None:
            eprint("Re-timed voices of measure #", measure_no, "to close on", expected)
            plans.update(plan)

    for group_no, group in enumerate(groups):
        first_rhythm = group.symbols[0].rhythm
        if group.is_barline() or first_rhythm.startswith("repeat"):
            plan_current_measure()
            measure_no += 1
            measure_groups = []
            has_multirest = False
        elif first_rhythm.startswith(("note", "rest")):
            if len(group.symbols) == 1 and first_rhythm.endswith("m"):
                has_multirest = True
            else:
                measure_groups.append((group_no, group))
    plan_current_measure()
    return plans


def build_note_chord(
    note_chord: SymbolChord, state: ConversionState, chord_duration: Fraction
) -> list[ET.Element]:
    by_duration = _group_notes(note_chord.symbols)
    result: list[ET.Element] = []
    for i, (group_duration, group_notes) in enumerate(by_duration.items()):
        notes = [n for n in group_notes if n.pitch not in (empty, nonote)]
        rests = [n for n in group_notes if n.pitch in (empty, nonote)]

        is_first = True
        for note in notes:
            result.append(build_note_or_rest(note, i, not is_first, state, note_chord.tuplet_mark))
            is_first = False

        if rests:
            assert group_duration > Fraction(0)
            if notes:
                # There are other notes, so to avoid rest being merged into chord, we emit a backup
                result.append(build_backup(group_duration, state))
            # Ideally we expect len(rests) == 1, but in dataset we see cases where
            # there are multiple rests. So here we just take the first rest
            result.append(build_note_or_rest(rests[0], i, False, state, note_chord.tuplet_mark))

        if i != len(by_duration) - 1 and group_duration > Fraction(0):
            result.append(build_backup(group_duration, state))

    if chord_duration < max(by_duration):
        result.append(build_backup(max(by_duration) - chord_duration, state))

    return result


def _group_notes(notes: list[EncodedSymbol]) -> dict[Fraction, list[EncodedSymbol]]:
    groups_by_duration = defaultdict(list)
    max_duration = max([n.get_duration().fraction for n in notes])
    for note in notes:
        duration = note.get_duration()
        is_grace = "G" in note.rhythm
        if is_grace:
            fraction = Fraction(0)
        elif duration.fraction.numerator == 0:
            fraction = max_duration
        else:
            fraction = duration.fraction
        groups_by_duration[fraction].append(note)
    return dict(sorted(groups_by_duration.items()))


def build_add_time_direction(args: XmlGeneratorArguments) -> ET.Element | None:
    if not args.metronome:
        return None
    direction = ET.Element("direction")
    direction_type = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(direction_type, "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = str(args.metronome)
    tempo = args.tempo if args.tempo else args.metronome
    ET.SubElement(direction, "sound", tempo=str(tempo))
    return direction


def find_common_division(durations: list[Fraction]) -> int:
    """
    Find the smallest division (denominator) so that all durations
    can be expressed as integer multiples.
    """

    def lcm(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b)

    denominators = [d.denominator for d in durations if d > 0]
    if not denominators:
        return 1
    common = denominators[0]
    for d in denominators[1:]:
        common = lcm(common, d)
    return common


def find_division_and_time_signature_nominator(voice: list[SymbolChord]) -> tuple[int, Fraction]:
    durations = [Fraction(1, 4)]
    duration_in_measure = Fraction(0)
    measure_duration = []
    for chord in voice:
        if chord.is_barline() and duration_in_measure > Fraction(0):
            measure_duration.append(duration_in_measure)
            duration_in_measure = Fraction(0)
        else:
            duration = chord.get_duration()
            if duration > Fraction(0):
                durations.append(duration)
                duration_in_measure += duration

    if duration_in_measure > Fraction(0):
        measure_duration.append(duration_in_measure)

    if len(measure_duration) == 0:
        return find_common_division(durations), Fraction(1)

    nominator: Fraction = np.median(measure_duration)  # type: ignore

    return find_common_division(durations), nominator


def group_into_chords(voice: list[EncodedSymbol]) -> list[SymbolChord]:
    return [SymbolChord(s) for s in sort_token_chords(voice)]


class TupletParser:
    @staticmethod
    def parse(groups: list[SymbolChord]) -> list[SymbolChord]:
        for measure_groups in TupletParser.split_into_measures(groups):
            saved_marks = [group.tuplet_mark for group in measure_groups]
            if TupletParser.add_tuplets(measure_groups):
                continue
            for group, mark in zip(measure_groups, saved_marks, strict=True):
                group.tuplet_mark = mark
        return groups

    @staticmethod
    def get_tuplet_duration(group: SymbolChord) -> SymbolDuration | None:
        for symbol in group.symbols:
            if symbol.rhythm.startswith(("note", "rest")):
                duration = symbol.get_duration()
                if duration.normal_notes != duration.actual_notes:
                    return duration
        return None

    @staticmethod
    def split_into_measures(groups: list[SymbolChord]) -> list[list[SymbolChord]]:
        measures: list[list[SymbolChord]] = []
        current_measure: list[SymbolChord] = []
        for group in groups:
            current_measure.append(group)
            if group.is_barline():
                measures.append(current_measure)
                current_measure = []
        if current_measure:
            measures.append(current_measure)
        return measures

    @staticmethod
    def add_tuplets(groups: list[SymbolChord]) -> bool:
        cursor = 0
        while cursor < len(groups):
            duration = TupletParser.get_tuplet_duration(groups[cursor])

            if duration is None:
                cursor += 1
                continue

            start = cursor
            tuplet_format = (duration.actual_notes, duration.normal_notes)
            tuplet_size = duration.actual_notes

            while cursor - start < tuplet_size:
                if cursor >= len(groups):
                    return False
                current_duration = TupletParser.get_tuplet_duration(groups[cursor])
                if current_duration is None:
                    return False
                current_format = (current_duration.actual_notes, current_duration.normal_notes)
                if current_format != tuplet_format:
                    return False
                cursor += 1

            groups[start].tuplet_mark = "start"
            groups[cursor - 1].tuplet_mark = "stop"

        return True


def add_tuplet_start_stop(groups: list[SymbolChord]) -> list[SymbolChord]:
    return TupletParser.parse(groups)


if __name__ == "__main__":
    import sys

    from training.transformer.training_vocabulary import read_tokens

    file = "tabi_measure.tokens"
    if len(sys.argv) > 1:
        file = sys.argv[1]
    tokens = read_tokens(file)
    xml = generate_xml(XmlGeneratorArguments(True), [tokens], "")
    ET.ElementTree(xml).write(
        file.replace(".tokens", ".musicxml"), encoding="unicode", xml_declaration=True
    )
