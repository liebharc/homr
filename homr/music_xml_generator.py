import math
from collections import defaultdict
from fractions import Fraction

import musicxml.xmlelement.xmlelement as mxl
import numpy as np

from homr import constants
from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, SymbolDuration, empty, nonote


class ConversionState:
    def __init__(self, division: int, nominator: Fraction):
        self.beats = 4 * constants.duration_of_quarter
        self.division = division
        self.nominator = nominator
        self.tremolo_state = "stop"

    def toggle_tremolo_state(self) -> str:
        if self.tremolo_state == "start":
            self.tremolo_state = "stop"
        else:
            self.tremolo_state = "start"
        return self.tremolo_state


class SymbolGroup:
    def __init__(self, symbols: list[EncodedSymbol]) -> None:
        self.symbols = symbols
        self.tuplet_mark = ""

    def __str__(self) -> str:
        return str.join("&", [str(s) for s in self.symbols])

    def __repr__(self) -> str:
        return str(self)


class XmlGeneratorArguments:
    def __init__(self, large_page: bool | None, metronome: int | None, tempo: int | None):
        self.large_page = large_page
        self.metronome = metronome
        self.tempo = tempo


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


def build_or_get_attributes(
    measure: mxl.XMLMeasure, last_attributes: mxl.XMLAttributes | None, force_new: bool = False
) -> mxl.XMLAttributes:
    if last_attributes is not None and not force_new:
        return last_attributes

    attributes = mxl.XMLAttributes()
    measure.add_child(attributes)
    return attributes


def build_or_get_barline(measure: mxl.XMLMeasure) -> mxl.XMLBarline:
    children = measure.get_children()
    if len(children) > 0 and isinstance(children[-1], mxl.XMLBarline):
        return children[-1]

    barline = mxl.XMLBarline()
    measure.add_child(barline)
    return barline


def build_key(model_key: EncodedSymbol, attributes: mxl.XMLAttributes) -> None:
    key = mxl.XMLKey()
    circle_of_fifth = model_key.rhythm.split("_")[1]
    fifth = mxl.XMLFifths(value_=int(circle_of_fifth))
    attributes.add_child(key)
    key.add_child(fifth)


def build_clef(model_clef: EncodedSymbol, attributes: mxl.XMLAttributes) -> None:
    sign_and_line = model_clef.rhythm.split("_")[1]
    sign = sign_and_line[0]
    line = sign_and_line[1]
    clef = mxl.XMLClef()
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
    note: mxl.XMLNote, articualations: str, tuplet_mark: str, state: ConversionState
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

    if "G" in model_note.rhythm:
        note.add_child(mxl.XMLGrace())
        base_duration = model_duration.kern
        duration_name = DURATION_NAMES[base_duration]
        note.add_child(mxl.XMLType(value_=duration_name))
    elif model_duration.fraction.numerator > 0:
        base_duration = model_duration.kern
        duration_name = DURATION_NAMES[base_duration]
        note.add_child(mxl.XMLType(value_=duration_name))
        note.add_child(mxl.XMLDuration(value_=int(model_duration.fraction * state.division)))
    else:
        note.add_child(mxl.XMLDuration(value_=state.beats))

    note.add_child(mxl.XMLStaff(value_=1))
    note.add_child(mxl.XMLVoice(value_=(str(voice + 1))))
    for _ in range(model_duration.dots):
        note.add_child(mxl.XMLDot())
    if model_duration.actual_notes != model_duration.normal_notes:
        time_modification = mxl.XMLTimeModification()
        time_modification.add_child(mxl.XMLActualNotes(value_=model_duration.actual_notes))
        time_modification.add_child(mxl.XMLNormalNotes(value_=model_duration.normal_notes))
        note.add_child(time_modification)
        build_articulations(note, model_note.articulation, tuplet_mark, state)
    else:
        build_articulations(note, model_note.articulation, "", state)

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


def build_note_group(note_group: SymbolGroup, state: ConversionState) -> list[mxl.XMLNote]:
    by_duration = _group_notes(note_group.symbols)
    result = []
    for i, group_duration in enumerate(sorted(by_duration)):
        is_first = True
        for note in by_duration[group_duration]:
            result.append(build_note_or_rest(note, i, not is_first, state, note_group.tuplet_mark))
            is_first = False
        if i != len(by_duration) - 1 and group_duration > Fraction(0):
            backup = mxl.XMLBackup()
            backup.add_child(mxl.XMLDuration(value_=int(group_duration * state.division)))
            result.append(backup)

    # The shorted symbol defines the chord length
    non_zero_durations = [d for d in by_duration if d > Fraction(0)]
    if len(non_zero_durations) > 0:
        shortest_duration = min(non_zero_durations)
        largest_duration = max(non_zero_durations)

        if shortest_duration < largest_duration:
            backup = mxl.XMLBackup()
            backup.add_child(
                mxl.XMLDuration(value_=int((largest_duration - shortest_duration) * state.division))
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


def find_common_division(durations: list[SymbolDuration]) -> int:
    """
    Find the smallest division (denominator) so that all durations
    can be expressed as integer multiples.
    """

    def lcm(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b)

    denominators = [d.fraction.denominator for d in durations if d.fraction > 0]
    if not denominators:
        return 1
    common = denominators[0]
    for d in denominators[1:]:
        common = lcm(common, d)
    return common


def find_division_and_time_signature_nominator(voice: list[EncodedSymbol]) -> tuple[int, Fraction]:
    durations = []
    duration_in_measure = Fraction(0)
    measure_duration = []
    for symbol in voice:
        if symbol.rhythm.startswith(("note", "rest")):
            duration = symbol.get_duration()
            durations.append(duration)
            duration_in_measure += duration.fraction
        if "barline" in symbol.rhythm and duration_in_measure > Fraction(0):
            measure_duration.append(duration_in_measure)
            duration_in_measure = Fraction(0)

    if duration_in_measure > Fraction(0):
        measure_duration.append(duration_in_measure)
        duration_in_measure = Fraction(0)

    if len(measure_duration) == 0:
        return find_common_division(durations), Fraction(1)

    nominator: Fraction = np.median(measure_duration)  # type: ignore

    return find_common_division(durations), nominator


def group_into_chords(voice: list[EncodedSymbol]) -> list[SymbolGroup]:
    chords: list[SymbolGroup] = []
    is_in_chord = False
    for symbol in voice:
        rhythm = symbol.rhythm
        if rhythm == "chord":
            is_in_chord = True
            continue
        if is_in_chord and symbol.rhythm.startswith(("rest", "note")):
            chords[-1].symbols.append(symbol)
        else:
            chords.append(SymbolGroup([symbol]))
        is_in_chord = False
    return chords


def add_tuplet_start_stop(groups: list[SymbolGroup]) -> list[SymbolGroup]:
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


def build_measures(
    args: XmlGeneratorArguments, voice: list[EncodedSymbol], is_first_part: bool
) -> list[mxl.XMLMeasure]:
    measure_number = 1
    division, nominator = find_division_and_time_signature_nominator(voice)
    state = ConversionState(division, nominator)
    measures: list[mxl.XMLMeasure] = []
    current_measure = mxl.XMLMeasure(number=str(measure_number))
    first_attributes = build_or_get_attributes(current_measure, None)
    first_attributes.add_child(mxl.XMLDivisions(value_=division))
    if is_first_part:
        direction = build_add_time_direction(args)
        if direction:
            current_measure.add_child(direction)
    groups = add_tuplet_start_stop(group_into_chords(voice))
    attributes: mxl.XMLAttributes | None = first_attributes
    for group in groups:
        symbol = group.symbols[0]
        rhythm = symbol.rhythm
        last_attributes = attributes
        attributes = None
        if rhythm.startswith(("note", "rest")):
            if len(group.symbols) == 1 and rhythm.endswith("m"):
                attributes = build_or_get_attributes(current_measure, last_attributes)
                build_multi_measure_rest(symbol, attributes)
            else:
                for note_xml in build_note_group(group, state):
                    current_measure.add_child(note_xml)
            continue
        if rhythm == "newline":
            measures[-1].add_child(mxl.XMLPrint(new_system="yes"))
        elif rhythm.startswith("clef"):
            attributes = build_or_get_attributes(current_measure, last_attributes, force_new=True)
            build_clef(symbol, attributes)
        elif rhythm.startswith("keySignature"):
            attributes = build_or_get_attributes(current_measure, last_attributes)
            build_key(symbol, attributes)
        elif rhythm.startswith("timeSignature"):
            attributes = build_or_get_attributes(current_measure, last_attributes)
            build_time_signature(symbol, attributes, state)
        elif "barline" in rhythm:
            if rhythm != "barline":
                # Standard barlines don't need extra handling
                barline = build_or_get_barline(current_measure)
                build_barline_style(symbol, barline)

            measures.append(current_measure)
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))
        elif rhythm.startswith("repeat"):
            barline = build_or_get_barline(current_measure)
            build_repeat(symbol, barline)
        else:
            eprint("Symbol isn't supported yet ", symbol)

    if len(current_measure.get_children()) > 0:
        measures.append(current_measure)
    return measures


def build_part(args: XmlGeneratorArguments, voice: list[EncodedSymbol], index: int) -> mxl.XMLPart:
    part = mxl.XMLPart(id=get_part_id(index))
    is_first_part = index == 0
    measures = build_measures(args, voice, is_first_part)
    for measure in measures:
        part.add_child(measure)
    return part


def generate_xml(
    args: XmlGeneratorArguments, staffs: list[list[EncodedSymbol]], title: str
) -> mxl.XMLElement:
    root = mxl.XMLScorePartwise()
    root.add_child(build_work(title))
    root.add_child(build_defaults(args))
    root.add_child(build_part_list(len(staffs)))
    for index, staff in enumerate(staffs):
        root.add_child(build_part(args, staff, index))
    return root
