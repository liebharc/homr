import bisect
import math
from collections import defaultdict
from fractions import Fraction

import musicxml.xmlelement.xmlelement as mxl
import numpy as np

from homr import constants
from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote, sort_token_chords


def _find_child_of_type(node: mxl.XMLElement, child_type: type) -> mxl.XMLElement | None:
    for child in node.get_children(ordered=False):
        if isinstance(child, child_type):
            return child
    return None


def _is_beamable_note(symbol: EncodedSymbol) -> bool:
    if not symbol.rhythm.startswith("note"):
        return False
    if "G" in symbol.rhythm:
        return False
    duration = symbol.get_duration()
    if duration.fraction <= Fraction(0):
        return False
    return duration.fraction < Fraction(1, 4)


def _finalize_pending_beam(pending: EncodedSymbol | None, run_length: int) -> None:
    if pending is None or run_length < 2:
        return
    pending.beam = "end"


def apply_beaming(groups: list["SymbolChord"]) -> None:
    for group in groups:
        for symbol in group.symbols:
            if hasattr(symbol, "beam"):
                delattr(symbol, "beam")

    for staff_position in ("upper", "lower"):
        pending: EncodedSymbol | None = None
        run_length = 0
        for group in groups:
            if group.is_barline():
                _finalize_pending_beam(pending, run_length)
                pending = None
                run_length = 0
                continue

            staff_chord = None
            for chord in group.into_positions():
                if chord.symbols and chord.symbols[0].position == staff_position:
                    staff_chord = chord
                    break

            if staff_chord is None:
                _finalize_pending_beam(pending, run_length)
                pending = None
                run_length = 0
                continue

            candidate = None
            has_note_or_rest = False
            has_rest = False
            for symbol in staff_chord.symbols:
                if symbol.rhythm.startswith("rest"):
                    has_rest = True
                    has_note_or_rest = True
                elif symbol.rhythm.startswith("note"):
                    has_note_or_rest = True
                if _is_beamable_note(symbol):
                    candidate = symbol
                    break

            # A rest always breaks the beam, even if chord notes are present
            if has_rest:
                candidate = None

            if candidate is None:
                if has_note_or_rest:
                    _finalize_pending_beam(pending, run_length)
                    pending = None
                    run_length = 0
                continue

            if pending is None:
                pending = candidate
                run_length = 1
                continue

            pending.beam = "begin" if run_length == 1 else "continue"
            pending = candidate
            run_length += 1

        _finalize_pending_beam(pending, run_length)


class ConversionState:
    def __init__(self, division: int, nominator: Fraction):
        self.beats = 4 * constants.duration_of_quarter
        self.division = division
        self.nominator = nominator
        self.tremolo_state = "stop"
        self.volta_number = 1
        self.last_volta_measure = -10
        self._open_slurs: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._available_slur_numbers = list(range(1, 16))
        self._open_ties: dict[tuple[int, str, str], list[int]] = defaultdict(list)
        self._available_tie_numbers: dict[tuple[int, str, str], list[int]] = defaultdict(
            lambda: list(range(1, 16))
        )

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

    def start_slur(self, staff: int, voice: int) -> int:
        key = (staff, voice)
        stack = self._open_slurs[key]
        if len(self._available_slur_numbers) == 0:
            eprint("Exceeded slur numbering limits; slurs may overlap incorrectly")
            number = stack[-1] if len(stack) > 0 else 6
        else:
            number = self._available_slur_numbers.pop(0)
        stack.append(number)
        return number

    def stop_slur(self, staff: int, voice: int) -> int | None:
        key = (staff, voice)
        if key not in self._open_slurs or len(self._open_slurs[key]) == 0:
            return None
        number = self._open_slurs[key].pop()
        self._release_slur_number(key, number)
        return number

    def _release_slur_number(self, key: tuple[int, int], number: int) -> None:
        if 1 <= number <= 6 and number not in self._available_slur_numbers:
            bisect.insort(self._available_slur_numbers, number)

    def start_tie(self, staff: int, pitch: str, lift: str) -> int:
        key = (staff, pitch, lift)
        available = self._available_tie_numbers[key]
        stack = self._open_ties[key]
        if len(available) == 0:
            eprint("Exceeded tie numbering limits; ties may overlap incorrectly")
            number = stack[-1] if len(stack) > 0 else 6
        else:
            number = available.pop(0)
        stack.append(number)
        return number

    def stop_tie(self, staff: int, pitch: str, lift: str) -> int | None:
        key = (staff, pitch, lift)
        if key not in self._open_ties or len(self._open_ties[key]) == 0:
            return None
        number = self._open_ties[key].pop()
        self._release_tie_number(key, number)
        return number

    def _release_tie_number(self, key: tuple[int, str, str], number: int) -> None:
        available = self._available_tie_numbers[key]
        if 1 <= number <= 6 and number not in available:
            bisect.insort(available, number)


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

    def strip_slur_ties(self) -> tuple[list[list[str]], "SymbolChord"]:
        stripped_per_symbol = []
        result = []
        for symbol in self.symbols:
            stripped, result_symbol = symbol.strip_articulations(
                ["slurStart", "slurStop", "tieStart", "tieStop"]
            )
            stripped_per_symbol.append(stripped)
            result.append(result_symbol)

        return stripped_per_symbol, SymbolChord(result, tuplet_mark=self.tuplet_mark)


class XmlGeneratorArguments:
    def __init__(
        self, large_page: bool | None = None, metronome: int | None = None, tempo: int | None = None
    ):
        self.large_page = large_page
        self.metronome = metronome
        self.tempo = tempo


def generate_xml(
    args: XmlGeneratorArguments, staffs: list[list[EncodedSymbol]], title: str
) -> tuple[mxl.XMLElement, list[int]]:
    root = mxl.XMLScorePartwise()
    root.add_child(build_work(title))
    root.add_child(build_defaults(args))
    root.add_child(build_part_list(len(staffs)))
    measure_counts_per_staff_line: list[int] = []
    for index, staff in enumerate(staffs):
        part, measure_counts = build_part(args, staff, index)
        root.add_child(part)
        measure_counts_per_staff_line.extend(measure_counts)
    return root, measure_counts_per_staff_line


def count_measures_for_encoded_staff(staff: list[EncodedSymbol]) -> int:
    return len(build_measures(XmlGeneratorArguments(), staff, False))


def build_part(
    args: XmlGeneratorArguments,
    voice: list[EncodedSymbol],
    index: int,
) -> tuple[mxl.XMLPart, list[int]]:
    part = mxl.XMLPart(id=get_part_id(index))
    is_first_part = index == 0
    measures = build_measures(args, voice, is_first_part)
    for measure in measures:
        part.add_child(measure)
    staff_count = _max_staff_number(voice)
    return part, [len(measures)] * staff_count


def build_measures(
    args: XmlGeneratorArguments, voice: list[EncodedSymbol], is_first_part: bool
) -> list[mxl.XMLMeasure]:
    measure_number = 1
    groups = add_tuplet_start_stop(group_into_chords(voice))
    apply_beaming(groups)
    division, nominator = find_division_and_time_signature_nominator(groups)
    state = ConversionState(division, nominator)
    measures: list[mxl.XMLMeasure] = []
    current_measure = mxl.XMLMeasure(number=str(measure_number))
    first_attributes = build_or_get_attributes(current_measure, None)
    first_attributes.add_child(build_divisions(division))
    max_staff = _max_staff_number(voice)
    if max_staff > 1:
        first_attributes.add_child(mxl.XMLStaves(value_=max_staff))
    if is_first_part:
        direction = build_add_time_direction(args)
        if direction:
            current_measure.add_child(direction)
    attributes: mxl.XMLAttributes | None = first_attributes
    for group_no, group in enumerate(groups):
        symbol = group.symbols[0]
        rhythm = symbol.rhythm
        last_attributes = attributes
        attributes = None
        if rhythm.startswith(("note", "rest")):
            if len(group.symbols) == 1 and rhythm.endswith("m"):
                attributes = build_or_get_attributes(current_measure, last_attributes)
                build_multi_measure_rest(symbol, attributes)
            else:
                staff_positions = group.into_positions()
                for pos_no, staff_pos in enumerate(staff_positions):
                    chord_duration = (
                        group.get_duration() if pos_no == len(staff_positions) - 1 else Fraction(0)
                    )
                    for note_xml in build_note_chord(staff_pos, state, chord_duration):
                        current_measure.add_child(note_xml)
            continue
        if rhythm == "newline":
            is_last_measure = group_no == len(groups) - 1
            if not is_last_measure:
                current_measure.add_child(mxl.XMLPrint(new_system="yes"))
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
                # Standard barlines don't need extra handling
                barline = build_or_get_barline(current_measure, "right")
                build_barline_style(symbol, barline)

            measures.append(current_measure)
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))
        elif rhythm == "repeatStart":
            measures.append(current_measure)
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))

            barline = build_or_get_barline(current_measure, "right")
            build_repeat(symbol, barline)
        elif rhythm == "repeatEnd":
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(symbol, barline)

            measures.append(current_measure)
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))
        elif rhythm == "repeatEndStart":
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(EncodedSymbol("repeatEnd"), barline)

            measures.append(current_measure)
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))
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

    if len(current_measure.get_children()) > 0:
        measures.append(current_measure)
    return measures


def build_work(title_text: str) -> mxl.XMLWork:
    work = mxl.XMLWork()
    title = mxl.XMLWorkTitle()
    title._value = title_text
    work.add_child(title)
    return work


def build_defaults(args: XmlGeneratorArguments) -> mxl.XMLDefaults:
    if not args.large_page:
        return mxl.XMLDefaults()
    # These values are larger than a letter or A4 format so that
    # we only have to break staffs with every new detected staff
    # This works well for electronic formats, if the results are supposed
    # to get printed then they might need to be scaled down to fit the page
    page_width = 110  # Unit is in tenths: https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/page-height/
    page_height = 300
    defaults = mxl.XMLDefaults()
    page_layout = mxl.XMLPageLayout()
    page_height = mxl.XMLPageHeight(value_=page_height)
    page_width = mxl.XMLPageWidth(value_=page_width)
    page_layout.add_child(page_height)
    page_layout.add_child(page_width)
    defaults.add_child(page_layout)
    return defaults


def get_part_id(index: int) -> str:
    return "P" + str(index + 1)


def build_part_list(staffs: int) -> mxl.XMLPartList:
    part_list = mxl.XMLPartList()
    for part in range(staffs):
        part_id = get_part_id(part)
        score_part = mxl.XMLScorePart(id=part_id)
        part_name = mxl.XMLPartName(value_="")
        score_part.add_child(part_name)
        score_instrument = mxl.XMLScoreInstrument(id=part_id + "-I1")
        instrument_name = mxl.XMLInstrumentName(value_="Piano")
        score_instrument.add_child(instrument_name)
        instrument_sound = mxl.XMLInstrumentSound(value_="keyboard.piano")
        score_instrument.add_child(instrument_sound)
        score_part.add_child(score_instrument)
        midi_instrument = mxl.XMLMidiInstrument(id=part_id + "-I1")
        midi_instrument.add_child(mxl.XMLMidiChannel(value_=1))
        midi_instrument.add_child(mxl.XMLMidiProgram(value_=1))
        midi_instrument.add_child(mxl.XMLVolume(value_=100))
        midi_instrument.add_child(mxl.XMLPan(value_=0))
        score_part.add_child(midi_instrument)
        part_list.add_child(score_part)
    return part_list


def _measure_has_musical_content(measure: mxl.XMLMeasure) -> bool:
    for child in measure.get_children():
        if isinstance(
            child,
            mxl.XMLNote | mxl.XMLBackup | mxl.XMLForward,
        ):
            return True
    return False


def build_or_get_attributes(
    measure: mxl.XMLMeasure, last_attributes: mxl.XMLAttributes | None, force_new: bool = False
) -> mxl.XMLAttributes:
    if last_attributes is not None and (not force_new or not _measure_has_musical_content(measure)):
        return last_attributes

    attributes = mxl.XMLAttributes()
    measure.add_child(attributes)
    return attributes


def _max_staff_number(voice: list[EncodedSymbol]) -> int:
    max_staff = 1
    for symbol in voice:
        max_staff = max(max_staff, get_staff(symbol))
    return max_staff


def build_or_get_barline(measure: mxl.XMLMeasure, location: str) -> mxl.XMLBarline:
    children = measure.get_children_of_type(mxl.XMLBarline)
    for child in children:
        if child.location == location:
            return child

    barline = mxl.XMLBarline(location=location)
    measure.add_child(barline)
    return barline


def build_key(model_key: EncodedSymbol, attributes: mxl.XMLAttributes) -> None:
    key = mxl.XMLKey()
    circle_of_fifth = model_key.rhythm.split("_")[1]
    fifth = mxl.XMLFifths(value_=int(circle_of_fifth))
    attributes.add_child(key)
    key.add_child(fifth)


def get_staff(symbol: EncodedSymbol) -> int:
    return 2 if symbol.position == "lower" else 1


def _get_symbol_lyric_verses(symbol: EncodedSymbol) -> dict[int, str]:
    verses: dict[int, str] = {}
    dynamic = getattr(symbol, "lyric_verses", None)
    if isinstance(dynamic, dict):
        for verse, text in dynamic.items():
            if not isinstance(verse, int) or verse < 1 or not isinstance(text, str):
                continue
            normalized = text.strip()
            if normalized:
                verses[verse] = normalized
    if symbol.lyric:
        normalized = symbol.lyric.strip()
        if normalized:
            verses.setdefault(1, normalized)
    return dict(sorted(verses.items(), key=lambda pair: pair[0]))


def build_clef(model_clef: EncodedSymbol, attributes: mxl.XMLAttributes) -> None:
    sign_and_line = model_clef.rhythm.split("_")[1]
    sign = sign_and_line[0]
    line = sign_and_line[1]
    clef = mxl.XMLClef(number=get_staff(model_clef))
    attributes.add_child(clef)
    clef.add_child(mxl.XMLSign(value_=sign))
    clef.add_child(mxl.XMLLine(value_=int(line)))


def build_time_signature(
    model_time_signature: EncodedSymbol, attributes: mxl.XMLAttributes, state: ConversionState
) -> None:
    time = mxl.XMLTime()

    denominator = model_time_signature.rhythm.split("/")[1]
    attributes.add_child(time)
    beats = max(int(state.nominator * int(denominator)), 1)
    time.add_child(mxl.XMLBeats(value_=str(beats)))
    time.add_child(mxl.XMLBeatType(value_=denominator))
    state.beats = beats


def build_barline_style(barline: EncodedSymbol, xml: mxl.XMLBarline) -> None:
    style_value = "heavy-heavy" if barline.rhythm == "bolddoublebarline" else "light-light"
    style = mxl.XMLBarStyle(value_=style_value)
    xml.add_child(style)


def build_barline_ending(volta: EncodedSymbol, xml: mxl.XMLBarline, volta_number: int) -> None:
    if volta.rhythm.startswith("voltaStart"):
        ending = mxl.XMLEnding(type="start", number=str(volta_number))
    elif volta.rhythm.startswith("voltaStop"):
        ending = mxl.XMLEnding(type="stop", number=str(volta_number))
    elif volta.rhythm.startswith("voltaDiscontinue"):
        ending = mxl.XMLEnding(type="discontinue", number=str(volta_number))
    else:
        raise ValueError("Unknown ending " + str(volta))
    xml.add_child(ending)


def build_repeat(barline: EncodedSymbol, xml: mxl.XMLBarline) -> None:
    if len(xml.get_children_of_type(mxl.XMLRepeat)) > 0:
        eprint("barline already has a repeat")
        return

    repeat = mxl.XMLRepeat()
    direction = "forward" if barline.rhythm == "repeatStart" else "backward"
    repeat._set_attributes({"direction": direction})
    xml.add_child(repeat)


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
    note: mxl.XMLNote,
    articualations: str,
    tuplet_mark: str,
    state: ConversionState,
    staff: int | None = None,
    voice: int | None = None,
) -> None:
    notation = mxl.XMLNotations()
    note.add_child(notation)

    xml_articulations = []
    xml_ornaments = []

    for articulation in articualations.split("_"):
        if articulation == "":
            continue
        elif articulation == nonote:
            eprint("WARNING note without valid articulation", articualations)
        elif articulation == "fermata":
            notation.add_child(mxl.XMLFermata())
        elif articulation == "arpeggiate":
            notation.add_child(mxl.XMLArpeggiate())
        elif articulation == "accent":
            xml_articulations.append(mxl.XMLAccent())
        elif articulation == "arpeggiate":
            notation.add_child(mxl.XMLArpeggiate())
        elif articulation == "fermata":
            xml_articulations.append(mxl.XMLFermata())
        elif articulation == "staccato":
            xml_articulations.append(mxl.XMLStaccato())
        elif articulation == "staccatissimo":
            xml_articulations.append(mxl.XMLStaccatissimo())
        elif articulation == "tenuto":
            xml_articulations.append(mxl.XMLTenuto())
        elif articulation == "tremolo":
            xml_ornaments.append(mxl.XMLTremolo(value_=3, type=state.toggle_tremolo_state()))
        elif articulation == "trill":
            xml_ornaments.append(mxl.XMLTrillMark())
        elif articulation == "breathMark":
            xml_articulations.append(mxl.XMLBreathMark())
        elif articulation == "turn":
            xml_ornaments.append(mxl.XMLInvertedTurn())
        elif articulation == "caesura":
            xml_articulations.append(mxl.XMLCaesura())
        elif articulation == "doit":
            xml_articulations.append(mxl.XMLDoit())
        elif articulation == "slurStart":
            attributes = {"type": "start"}
            if staff is not None and voice is not None:
                number = state.start_slur(staff, voice)
                attributes["number"] = number
            notation.add_child(mxl.XMLSlur(**attributes))
        elif articulation == "slurStop":
            attributes = {"type": "stop"}
            if staff is not None and voice is not None:
                number = state.stop_slur(staff, voice)
                if number is not None:
                    attributes["number"] = number
            if "number" in attributes:
                notation.add_child(mxl.XMLSlur(**attributes))
        elif articulation == "tieStart":
            attributes = {"type": "start"}
            if staff is not None and voice is not None:
                pitch = _find_child_of_type(note, mxl.XMLPitch)
                if pitch:
                    step = _find_child_of_type(pitch, mxl.XMLStep)
                    octave = _find_child_of_type(pitch, mxl.XMLOctave)
                    if step and octave:
                        pitch_key = f"{step.value_}{octave.value_}"
                        alter = _find_child_of_type(pitch, mxl.XMLAlter)
                        lift = str(alter.value_) if alter else "0"
                        number = state.start_tie(staff, pitch_key, lift)
                        attributes["number"] = number
            notation.add_child(mxl.XMLTied(**attributes))
        elif articulation == "tieStop":
            attributes = {"type": "stop"}
            if staff is not None and voice is not None:
                pitch = _find_child_of_type(note, mxl.XMLPitch)
                if pitch:
                    step = _find_child_of_type(pitch, mxl.XMLStep)
                    octave = _find_child_of_type(pitch, mxl.XMLOctave)
                    if step and octave:
                        pitch_key = f"{step.value_}{octave.value_}"
                        alter = _find_child_of_type(pitch, mxl.XMLAlter)
                        lift = str(alter.value_) if alter else "0"
                        number = state.stop_tie(staff, pitch_key, lift)
                        if number is not None:
                            attributes["number"] = number
            if "number" in attributes:
                notation.add_child(mxl.XMLTied(**attributes))
        else:
            raise ValueError("Unsupported articulation " + articulation)

    if tuplet_mark != "":
        tuplet_xml = mxl.XMLTuplet(type=tuplet_mark)
        notation.add_child(tuplet_xml)

    if len(xml_articulations) > 0:
        parent = mxl.XMLArticulations()
        for child in xml_articulations:
            parent.add_child(child)
        notation.add_child(parent)

    if len(xml_ornaments) > 0:
        parent = mxl.XMLOrnaments()
        for child in xml_ornaments:
            parent.add_child(child)
        notation.add_child(parent)


def build_note_or_rest(
    model_note: EncodedSymbol, voice: int, is_chord: bool, state: ConversionState, tuplet_mark: str
) -> mxl.XMLNote:
    note = mxl.XMLNote()
    if is_chord:
        note.add_child(mxl.XMLChord())
    model_pitch = model_note.pitch
    model_duration = model_note.get_duration()
    if model_pitch == empty:
        if model_duration.fraction.numerator == 0:
            note.add_child(mxl.XMLRest(measure="yes"))
        else:
            note.add_child(mxl.XMLRest())
    elif model_pitch == nonote:
        eprint("WARNING note without pitch", model_note)
        note.add_child(mxl.XMLRest())
    else:
        pitch = mxl.XMLPitch()
        pitch.add_child(mxl.XMLStep(value_=model_pitch[0]))
        pitch.add_child(mxl.XMLOctave(value_=int(model_pitch[1])))
        if model_note.lift == nonote:
            eprint("WARNING note with invalid lift", model_note)
        elif model_note.lift != empty:
            pitch.add_child(mxl.XMLAlter(value_=LIFT_TO_ALTER[model_note.lift]))
        note.add_child(pitch)
    for verse, lyric_text in _get_symbol_lyric_verses(model_note).items():
        lyric = mxl.XMLLyric(number=str(verse))
        lyric.add_child(mxl.XMLText(value_=lyric_text))
        note.add_child(lyric)

    if "G" in model_note.rhythm:
        note.add_child(mxl.XMLGrace())
        base_duration = model_duration.kern
        duration_name = DURATION_NAMES[base_duration]
        note.add_child(mxl.XMLType(value_=duration_name))
    elif model_duration.fraction.numerator > 0:
        base_duration = 1 if model_duration.kern == 0 else model_duration.kern
        duration_name = DURATION_NAMES[base_duration]
        note.add_child(mxl.XMLType(value_=duration_name))
        note.add_child(mxl.XMLDuration(value_=int(model_duration.fraction * state.division)))
    else:
        duration_name = DURATION_NAMES[0]
        note.add_child(mxl.XMLType(value_=duration_name))
        note.add_child(mxl.XMLDuration(value_=state.beats))

    staff_value = get_staff(model_note)
    note.add_child(mxl.XMLStaff(value_=staff_value))
    xml_voice = voice + 1
    note.add_child(mxl.XMLVoice(value_=(str(xml_voice))))
    beam_value = getattr(model_note, "beam", None)
    if beam_value and not is_chord and _is_beamable_note(model_note):
        note.add_child(mxl.XMLBeam(value_=beam_value, number=1))
    for _ in range(model_duration.dots):
        note.add_child(mxl.XMLDot())
    if model_duration.actual_notes != model_duration.normal_notes:
        time_modification = mxl.XMLTimeModification()
        time_modification.add_child(mxl.XMLActualNotes(value_=model_duration.actual_notes))
        time_modification.add_child(mxl.XMLNormalNotes(value_=model_duration.normal_notes))
        note.add_child(time_modification)
        build_articulations(
            note, model_note.articulation, tuplet_mark, state, staff_value, xml_voice
        )
    else:
        build_articulations(note, model_note.articulation, "", state, staff_value, xml_voice)

    return note


def build_multi_measure_rest(
    symbol: EncodedSymbol, attributes: mxl.XMLAttributes
) -> mxl.XMLMeasureStyle:
    other_styles = attributes.get_children_of_type(mxl.XMLMeasureStyle)
    if len(other_styles) > 0:
        eprint("Measure already has a multi rest")
        return
    duration = int(symbol.rhythm.split("_")[1].replace("m", ""))
    style = mxl.XMLMeasureStyle()
    rest = mxl.XMLMultipleRest(value_=duration)
    style.add_child(rest)
    attributes.add_child(style)


def build_note_chord(
    note_chord: SymbolChord, state: ConversionState, chord_duration: Fraction
) -> list[mxl.XMLNote]:
    slur_tie_groups, note_chord = note_chord.strip_slur_ties()
    slur_tie_map = {
        id(symbol): articulations
        for symbol, articulations in zip(note_chord.symbols, slur_tie_groups, strict=False)
    }
    by_duration = _group_notes(note_chord.symbols)
    result = []
    final_duration = Fraction(0)
    # Use voice 2 for lower staff to separate beam groups between staves
    voice_offset = 0
    if note_chord.symbols and note_chord.symbols[0].position == "lower":
        voice_offset = 1
    for i, group_duration in enumerate(sorted(by_duration)):
        is_first = True
        for note_loop in by_duration[group_duration]:
            note = note_loop
            slur_ties = slur_tie_map.get(id(note_loop), [])
            if slur_ties:
                note = note.add_articulations(slur_ties)
            voice = i + voice_offset
            result.append(
                build_note_or_rest(note, voice, not is_first, state, note_chord.tuplet_mark)
            )
            is_first = False
        if i != len(by_duration) - 1 and group_duration > Fraction(0):
            backup = mxl.XMLBackup()
            backup.add_child(mxl.XMLDuration(value_=int(group_duration * state.division)))
            result.append(backup)

        final_duration = group_duration

    # Reset the position to match the chord position
    if chord_duration < final_duration:
        backup = mxl.XMLBackup()
        backup.add_child(
            mxl.XMLDuration(value_=int((final_duration - chord_duration) * state.division))
        )
        result.append(backup)
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
            # Whole measure rest
            fraction = max_duration
        else:
            fraction = duration.fraction
        groups_by_duration[fraction].append(note)
    return groups_by_duration


def build_add_time_direction(args: XmlGeneratorArguments) -> mxl.XMLDirection | None:
    if not args.metronome:
        return None
    direction = mxl.XMLDirection()
    direction_type = mxl.XMLDirectionType()
    direction.add_child(direction_type)
    metronome = mxl.XMLMetronome()
    direction_type.add_child(metronome)
    beat_unit = mxl.XMLBeatUnit(value_="quarter")
    metronome.add_child(beat_unit)
    per_minute = mxl.XMLPerMinute(value_=str(args.metronome))
    metronome.add_child(per_minute)
    if args.tempo:
        direction.add_child(mxl.XMLSound(tempo=args.tempo))
    else:
        direction.add_child(mxl.XMLSound(tempo=args.metronome))
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
        duration_in_measure = Fraction(0)

    if len(measure_duration) == 0:
        return find_common_division(durations), Fraction(1)

    nominator: Fraction = np.median(measure_duration)  # type: ignore

    return find_common_division(durations), nominator


def group_into_chords(voice: list[EncodedSymbol]) -> list[SymbolChord]:
    return [SymbolChord(s) for s in sort_token_chords(voice)]


def add_tuplet_start_stop(groups: list[SymbolChord]) -> list[SymbolChord]:
    has_tuplet_mark = False
    for group in groups:
        is_tuplet = False
        for symbol in group.symbols:
            if symbol.rhythm.startswith(("note", "rest")):
                duration = symbol.get_duration()
                is_tuplet = is_tuplet or duration.normal_notes != duration.actual_notes

        if is_tuplet != has_tuplet_mark:
            if has_tuplet_mark:
                group.tuplet_mark = "stop"
            else:
                group.tuplet_mark = "start"
            has_tuplet_mark = is_tuplet

    return groups


def build_divisions(division: int) -> mxl.XMLDivisions:
    # The divisions element indicates how many divisions per quarter(!) note are
    # used to indicate a note's duration
    # https://usermanuals.musicxml.com/MusicXML/Content/EL-MusicXML-divisions.htm
    return mxl.XMLDivisions(value_=division // 4)


if __name__ == "__main__":
    import sys

    from training.transformer.training_vocabulary import read_tokens

    file = "tabi_measure.tokens"
    if len(sys.argv) > 1:
        file = sys.argv[1]
    tokens = read_tokens(file)
    xml, _ = generate_xml(XmlGeneratorArguments(True), [tokens], "")
    xml.write(file.replace(".tokens", ".musicxml"))
