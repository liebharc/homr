import musicxml.xmlelement.xmlelement as mxl  # type: ignore

from . import constants
from .results import (
    ClefType,
    ResultClef,
    ResultMeasure,
    ResultNote,
    ResultNoteGroup,
    ResultRest,
    ResultStaff,
)


def build_work(f_name: str) -> mxl.XMLWork:  # type: ignore
    work = mxl.XMLWork()
    title = mxl.XMLWorkTitle()
    title._value = f_name
    work.add_child(title)
    return work


def build_defaults() -> mxl.XMLDefaults:  # type: ignore
    # These values are larger than a letter or A4 format so that
    # we only have to break staffs with every new detected staff
    # This works well for electronic formats, if the results are supposed
    # to get printed then they might need to be scaled down to fit the page
    page_width = 100  # Unit is in tenths: https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/page-height/
    page_height = 120
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


def build_clef(model_clef: ResultClef) -> mxl.XMLAttributes:  # type: ignore
    attributes = mxl.XMLAttributes()
    attributes.add_child(mxl.XMLDivisions(value_=constants.duration_of_quarter))
    key = mxl.XMLKey()
    fifth = mxl.XMLFifths(value_=model_clef.circle_of_fifth)
    attributes.add_child(key)
    key.add_child(fifth)
    clef = mxl.XMLClef()
    attributes.add_child(clef)
    if model_clef.clef_type == ClefType.TREBLE:
        clef.add_child(mxl.XMLSign(value_="G"))
        clef.add_child(mxl.XMLLine(value_=2))
    else:
        clef.add_child(mxl.XMLSign(value_="F"))
        clef.add_child(mxl.XMLLine(value_=4))
    return attributes


def build_rest(model_rest: ResultRest) -> mxl.XMLNote:  # type: ignore
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
    if model_duration.has_dot:
        note.add_child(mxl.XMLDot())
    return note


def build_note_group(note_group: ResultNoteGroup) -> mxl.XMLNote:  # type: ignore
    result = []
    is_first = True
    for note in note_group.notes:
        result.append(build_note(note, not is_first))
        is_first = False
    return result


def build_measure(measure: ResultMeasure, measure_number: int) -> mxl.XMLMeasure:  # type: ignore
    result = mxl.XMLMeasure(number=str(measure_number))
    if measure.is_new_line:
        result.add_child(mxl.XMLPrint(new_system="yes"))
    for symbol in measure.symbols:
        if isinstance(symbol, ResultClef):
            result.add_child(build_clef(symbol))
        elif isinstance(symbol, ResultRest):
            result.add_child(build_rest(symbol))
        elif isinstance(symbol, ResultNote):
            result.add_child(build_note(symbol))
        elif isinstance(symbol, ResultNoteGroup):
            for element in build_note_group(symbol):
                result.add_child(element)
    return result


def build_part(staff: ResultStaff, index: int) -> mxl.XMLPart:  # type: ignore
    part = mxl.XMLPart(id=get_part_id(index))
    measure_number = 1
    for measure in staff.measures:
        part.add_child(build_measure(measure, measure_number))
        measure_number += 1
    return part


def generate_xml(staffs: list[ResultStaff], title: str) -> mxl.XMLElement:  # type: ignore
    root = mxl.XMLScorePartwise()
    root.add_child(build_work(title))
    root.add_child(build_defaults())
    root.add_child(build_part_list(len(staffs)))
    for index, staff in enumerate(staffs):
        root.add_child(build_part(staff, index))
    return root
