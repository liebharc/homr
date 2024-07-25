import musicxml.xmlelement.xmlelement as mxl  # type: ignore

from homr import constants
from homr.results import (
    DurationModifier,
    ResultChord,
    ResultClef,
    ResultMeasure,
    ResultNote,
    ResultStaff,
    ResultTimeSignature,
)


class XmlGeneratorArguments:
    def __init__(self, large_page: bool | None, metronome: int | None, tempo: int | None):
        self.large_page = large_page
        self.metronome = metronome
        self.tempo = tempo


def build_work(title_text: str) -> mxl.XMLWork:  # type: ignore
    work = mxl.XMLWork()
    title = mxl.XMLWorkTitle()
    title._value = title_text
    work.add_child(title)
    return work


def build_defaults(args: XmlGeneratorArguments) -> mxl.XMLDefaults:  # type: ignore
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


def build_part_list(staffs: int) -> mxl.XMLPartList:  # type: ignore
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


def build_or_get_attributes(measure: mxl.XMLMeasure) -> mxl.XMLAttributes:  # type: ignore
    for child in measure.get_children_of_type(mxl.XMLAttributes):
        return child

    attributes = mxl.XMLAttributes()
    measure.add_child(attributes)
    return attributes


def build_clef(model_clef: ResultClef, attributes: mxl.XMLAttributes) -> None:  # type: ignore
    attributes.add_child(mxl.XMLDivisions(value_=constants.duration_of_quarter))
    key = mxl.XMLKey()
    fifth = mxl.XMLFifths(value_=model_clef.circle_of_fifth)
    attributes.add_child(key)
    key.add_child(fifth)
    clef = mxl.XMLClef()
    attributes.add_child(clef)
    clef.add_child(mxl.XMLSign(value_=model_clef.clef_type.sign))
    clef.add_child(mxl.XMLLine(value_=model_clef.clef_type.line))


def build_time_signature(  # type: ignore
    model_time_signature: ResultTimeSignature, attributes: mxl.XMLAttributes
) -> None:
    time = mxl.XMLTime()
    attributes.add_child(time)
    time.add_child(mxl.XMLBeats(value_=str(model_time_signature.numerator)))
    time.add_child(mxl.XMLBeatType(value_=str(model_time_signature.denominator)))


def build_rest(model_rest: ResultChord) -> mxl.XMLNote:  # type: ignore
    note = mxl.XMLNote()
    note.add_child(mxl.XMLRest(measure="yes"))
    note.add_child(mxl.XMLDuration(value_=model_rest.duration.duration))
    note.add_child(mxl.XMLType(value_=model_rest.duration.duration_name))
    note.add_child(mxl.XMLStaff(value_=1))
    return note


def build_note(model_note: ResultNote, is_chord=False) -> mxl.XMLNote:  # type: ignore
    note = mxl.XMLNote()
    if is_chord:
        note.add_child(mxl.XMLChord())
    pitch = mxl.XMLPitch()
    model_pitch = model_note.pitch
    pitch.add_child(mxl.XMLStep(value_=model_pitch.step))
    if model_pitch.alter is not None:
        pitch.add_child(mxl.XMLAlter(value_=model_pitch.alter))
    else:
        pitch.add_child(mxl.XMLAlter(value_=0))
    pitch.add_child(mxl.XMLOctave(value_=model_pitch.octave))
    note.add_child(pitch)
    model_duration = model_note.duration
    note.add_child(mxl.XMLType(value_=model_duration.duration_name))
    note.add_child(mxl.XMLDuration(value_=model_duration.duration))
    note.add_child(mxl.XMLStaff(value_=1))
    note.add_child(mxl.XMLVoice(value_="1"))
    if model_duration.modifier == DurationModifier.DOT:
        note.add_child(mxl.XMLDot())
    elif model_duration.modifier == DurationModifier.TRIPLET:
        time_modification = mxl.XMLTimeModification()
        time_modification.add_child(mxl.XMLActualNotes(value_=3))
        time_modification.add_child(mxl.XMLNormalNotes(value_=2))
        note.add_child(time_modification)
    return note


def build_note_group(note_group: ResultChord) -> list[mxl.XMLNote]:  # type: ignore
    result = []
    is_first = True
    for note in note_group.notes:
        result.append(build_note(note, not is_first))
        is_first = False
    return result


def build_chord(chord: ResultChord) -> list[mxl.XMLNote]:  # type: ignore
    if chord.is_rest:
        return [build_rest(chord)]
    return build_note_group(chord)


def build_add_time_direction(args: XmlGeneratorArguments) -> mxl.XMLDirection | None:  # type: ignore
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


def build_measure(  # type: ignore
    args: XmlGeneratorArguments, measure: ResultMeasure, is_first_part: bool, measure_number: int
) -> mxl.XMLMeasure:
    result = mxl.XMLMeasure(number=str(measure_number))
    is_first_measure = measure_number == 1
    if is_first_measure and is_first_part:
        direction = build_add_time_direction(args)
        if direction:
            result.add_child(direction)
    if measure.is_new_line and not is_first_measure:
        result.add_child(mxl.XMLPrint(new_system="yes"))
    for symbol in measure.symbols:
        if isinstance(symbol, ResultClef):
            attributes = build_or_get_attributes(result)
            build_clef(symbol, attributes)
        elif isinstance(symbol, ResultTimeSignature):
            attributes = build_or_get_attributes(result)
            build_time_signature(symbol, attributes)
        elif isinstance(symbol, ResultChord):
            for element in build_chord(symbol):
                result.add_child(element)
    return result


def build_part(  # type: ignore
    args: XmlGeneratorArguments, staff: ResultStaff, index: int
) -> mxl.XMLPart:
    part = mxl.XMLPart(id=get_part_id(index))
    measure_number = 1
    is_first_part = index == 0
    for measure in staff.measures:
        part.add_child(build_measure(args, measure, is_first_part, measure_number))
        measure_number += 1
    return part


def generate_xml(  # type: ignore
    args: XmlGeneratorArguments, staffs: list[ResultStaff], title: str
) -> mxl.XMLElement:
    root = mxl.XMLScorePartwise()
    root.add_child(build_work(title))
    root.add_child(build_defaults(args))
    root.add_child(build_part_list(len(staffs)))
    for index, staff in enumerate(staffs):
        root.add_child(build_part(args, staff, index))
    return root
