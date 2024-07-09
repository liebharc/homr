import xml.etree.ElementTree as ET

import musicxml.xmlelement.xmlelement as mxl  # type: ignore
from musicxml.parser.parser import _parse_node  # type: ignore

from homr import constants
from homr.circle_of_fifths import KeyTransformation, circle_of_fifth_to_key_signature
from homr.simple_logging import eprint


class MusicXmlValidationError(Exception):
    pass


class SymbolWithPosition:
    def __init__(self, position: int, symbol: str) -> None:
        self.position = position
        self.symbol = symbol


class SemanticMeasure:
    def __init__(self, number_of_clefs: int) -> None:
        self.staffs: list[list[SymbolWithPosition]] = [[] for _ in range(number_of_clefs)]
        self.current_position = 0

    def append_symbol(self, symbol: str) -> None:
        if len(self.staffs) == 0:
            raise ValueError("Expected to get clefs as first symbol")
        if symbol.startswith("note"):
            raise ValueError("Call append_note for notes")
        else:
            for staff in self.staffs:
                staff.append(SymbolWithPosition(-1, symbol))

    def append_symbol_to_staff(self, staff: int, symbol: str) -> None:
        if len(self.staffs) == 0:
            raise ValueError("Expected to get clefs as first symbol")
        if symbol.startswith("note"):
            raise ValueError("Call append_note for notes")
        else:
            self.staffs[staff].append(SymbolWithPosition(-1, symbol))

    def append_position_change(self, duration: int) -> None:
        if len(self.staffs) == 0:
            raise ValueError("Expected to get clefs as first symbol")
        new_position = self.current_position + duration
        if new_position < 0:
            raise ValueError(
                "Backup duration is too long " + str(self.current_position) + " " + str(duration)
            )
        self.current_position = new_position

    def append_rest(self, staff: int, duration: int, symbol: str) -> None:
        self.append_note(staff, False, duration, symbol)

    def append_note(self, staff: int, is_chord: bool, duration: int, symbol: str) -> None:
        if len(self.staffs) == 0:
            raise ValueError("Expected to get clefs as first symbol")
        if is_chord:
            if len(self.staffs[staff]) == 0:
                raise ValueError("A chord requires a previous note")
            previous_symbol = self.staffs[staff][-1]
            self.staffs[staff].append(SymbolWithPosition(previous_symbol.position, symbol))
            self.current_position = previous_symbol.position + duration
        else:
            self.staffs[staff].append(SymbolWithPosition(self.current_position, symbol))
            self.current_position += duration

    def complete_measure(self) -> list[list[str]]:
        result: list[list[str]] = []
        for staff in self.staffs:
            result_staff: list[str] = []
            grouped_symbols: dict[int, list[str]] = {}
            for symbol in staff:
                if symbol.position < 0:
                    # Directly append clefs, keys and time signatures
                    result_staff.append(symbol.symbol)
                    continue
                if symbol.position not in grouped_symbols:
                    grouped_symbols[symbol.position] = []
                grouped_symbols[symbol.position].append(symbol.symbol)
            for position in sorted(grouped_symbols.keys()):
                result_staff.append(str.join("|", grouped_symbols[position]))
            result_staff.append("barline")
            result.append(result_staff)
        return result


class SemanticPart:
    def __init__(self) -> None:
        self.current_measure: SemanticMeasure | None = None
        self.staffs: list[list[str]] = []

    def append_clefs(self, clefs: list[str]) -> None:
        if self.current_measure is not None:
            if len(self.current_measure.staffs) != len(clefs):
                raise ValueError("Number of clefs changed")
            for staff, clef in enumerate(clefs):
                if not any(symbol.symbol == clef for symbol in self.current_measure.staffs[staff]):
                    raise MusicXmlValidationError("Clef changed")
            return
        self.staffs = [[] for _ in range(len(clefs))]
        measure = SemanticMeasure(len(clefs))
        for staff, clef in enumerate(clefs):
            measure.append_symbol_to_staff(staff, clef)
        self.current_measure = measure

    def append_symbol(self, symbol: str) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        self.current_measure.append_symbol(symbol)

    def append_rest(self, staff: int, duration: int, symbol: str) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        self.current_measure.append_rest(staff, duration, symbol)

    def append_note(self, staff: int, is_chord: bool, duration: int, symbol: str) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        self.current_measure.append_note(staff, is_chord, duration, symbol)

    def append_position_change(self, duration: int) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        self.current_measure.append_position_change(duration)

    def on_end_of_measure(self) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        if self.current_measure.current_position == 0:
            # Measure was reset to start, likely to add a another voice
            # to it
            return
        for staff, result in enumerate(self.current_measure.complete_measure()):
            self.staffs[staff].extend(result)
        self.current_measure = SemanticMeasure(len(self.current_measure.staffs))

    def get_staffs(self) -> list[list[str]]:
        return self.staffs


def _translate_duration(duration: str) -> str:
    definition = {
        "breve": "double_whole",
        "whole": "whole",
        "half": "half",
        "quarter": "quarter",
        "eighth": "eighth",
        "16th": "sixteenth",
        "32nd": "thirty_second",
        "64th": "sixty_fourth",
    }
    return definition[duration]


def _get_alter(note: mxl.XMLPitch) -> str:  # type: ignore
    alter = note.get_children_of_type(mxl.XMLAlter)
    if len(alter) == 0:
        return ""
    alter_value = int(alter[0].value_)
    if alter_value == 1:
        return "#"
    if alter_value == -1:
        return "b"
    if alter_value == 0:
        return "N"
    return ""


def _get_alter_from_courtesey(accidental: mxl.XMLAccidental) -> str:  # type: ignore
    value = accidental.value_
    if value == "sharp":
        return "#"
    if value == "flat":
        return "b"
    if value == "natural":
        return "N"
    return ""


def _count_dots(note: mxl.XMLNote) -> str:  # type: ignore
    dots = note.get_children_of_type(mxl.XMLDot)
    return "." * len(dots)


def _get_triplet_mark(note: mxl.XMLNote) -> str:  # type: ignore
    time_modification = note.get_children_of_type(mxl.XMLTimeModification)
    if len(time_modification) == 0:
        return ""
    actual_notes = time_modification[0].get_children_of_type(mxl.XMLActualNotes)
    if len(actual_notes) == 0:
        return ""
    normal_notes = time_modification[0].get_children_of_type(mxl.XMLNormalNotes)
    if len(normal_notes) == 0:
        return ""
    is_triplet = (
        int(actual_notes[0].value_) == 3 and int(normal_notes[0].value_) == 2  # noqa: PLR2004
    )
    is_sixtuplet = (
        int(actual_notes[0].value_) == 6 and int(normal_notes[0].value_) == 4  # noqa: PLR2004
    )
    if is_triplet or is_sixtuplet:
        return constants.triplet_symbol
    return ""


def _process_attributes(  # type: ignore
    semantic: SemanticPart, attribute: mxl.XMLAttributes, key: KeyTransformation
) -> KeyTransformation:
    clefs = attribute.get_children_of_type(mxl.XMLClef)
    if len(clefs) > 0:
        clefs_semantic = []
        for clef in clefs:
            sign = clef.get_children_of_type(mxl.XMLSign)[0].value_
            line = clef.get_children_of_type(mxl.XMLLine)[0].value_
            clefs_semantic.append("clef-" + sign + str(line))
        semantic.append_clefs(clefs_semantic)
    keys = attribute.get_children_of_type(mxl.XMLKey)
    if len(keys) > 0:
        fifths = keys[0].get_children_of_type(mxl.XMLFifths)[0].value_
        semantic.append_symbol("keySignature-" + circle_of_fifth_to_key_signature(int(fifths)))
        key = KeyTransformation(int(fifths))
    times = attribute.get_children_of_type(mxl.XMLTime)
    if len(times) > 0:
        beats = times[0].get_children_of_type(mxl.XMLBeats)[0].value_
        beat_type = times[0].get_children_of_type(mxl.XMLBeatType)[0].value_
        semantic.append_symbol("timeSignature-" + beats + "/" + beat_type)
    return key


def _process_note(  # type: ignore
    semantic: SemanticPart, note: mxl.XMLNote, key: KeyTransformation
) -> KeyTransformation:
    staff = 0
    staff_nodes = note.get_children_of_type(mxl.XMLStaff)
    if len(staff_nodes) > 0:
        staff = int(staff_nodes[0].value_) - 1
    is_chord = len(note.get_children_of_type(mxl.XMLChord)) > 0
    if len(note.get_children_of_type(mxl.XMLDuration)) == 0:
        is_grace_note = len(note.get_children_of_type(mxl.XMLGrace)) > 0
        if not is_grace_note:
            eprint("Note without duration", note.get_children())
        duration = 0
    else:
        duration = int(note.get_children_of_type(mxl.XMLDuration)[0].value_)
    rest = note.get_children_of_type(mxl.XMLRest)
    if len(rest) > 0:
        dot = _count_dots(note)
        if rest[0] and rest[0].attributes.get("measure", None):
            semantic.append_rest(staff, duration, "rest-whole" + dot)
        else:
            duration_type = note.get_children_of_type(mxl.XMLType)[0].value_
            semantic.append_rest(
                staff, duration, "rest-" + _translate_duration(duration_type) + dot
            )
    pitch = note.get_children_of_type(mxl.XMLPitch)
    if len(pitch) > 0:
        alter = _get_alter(pitch[0])
        step = pitch[0].get_children_of_type(mxl.XMLStep)[0].value_
        octave = pitch[0].get_children_of_type(mxl.XMLOctave)[0].value_
        duration_type = note.get_children_of_type(mxl.XMLType)[0].value_
        alter = key.add_accidental(
            step + str(octave),
            alter,
        )
        courtesey_accidental = note.get_children_of_type(mxl.XMLAccidental)
        if len(courtesey_accidental) > 0:
            alter = _get_alter_from_courtesey(courtesey_accidental[0])

        semantic.append_note(
            staff,
            is_chord,
            duration,
            "note-"
            + step
            + str(octave)
            + alter
            + "_"
            + _translate_duration(duration_type)
            + _count_dots(note)
            + _get_triplet_mark(note),
        )
    return key


def _process_backup(semantic: SemanticPart, backup: mxl.XMLBackup) -> None:  # type: ignore
    backup_value = int(backup.get_children_of_type(mxl.XMLDuration)[0].value_)
    semantic.append_position_change(-backup_value)


def _process_forward(semantic: SemanticPart, backup: mxl.XMLBackup) -> None:  # type: ignore
    forward_value = int(backup.get_children_of_type(mxl.XMLDuration)[0].value_)
    semantic.append_position_change(forward_value)


def _music_part_to_semantic(part: mxl.XMLPart) -> list[list[str]]:  # type: ignore
    semantic = SemanticPart()
    key = KeyTransformation(0)
    for measure in part.get_children_of_type(mxl.XMLMeasure):
        for child in measure.get_children():
            if isinstance(child, mxl.XMLAttributes):
                key = _process_attributes(semantic, child, key)
            if isinstance(child, mxl.XMLNote):
                key = _process_note(semantic, child, key)
            if isinstance(child, mxl.XMLBackup):
                _process_backup(semantic, child)
            if isinstance(child, mxl.XMLForward):
                _process_forward(semantic, child)
        semantic.on_end_of_measure()
        key = key.reset_at_end_of_measure()
    return semantic.get_staffs()


def _remove_dynamics_attribute_from_nodes_recursive(node: ET.Element) -> None:
    """
    We don't need the dynamics attribute in the XML, but XSD validation
    sometimes fails if its negative. So we remove it.
    """
    if "dynamics" in node.attrib:
        del node.attrib["dynamics"]
    for child in node:
        _remove_dynamics_attribute_from_nodes_recursive(child)


def _music_xml_content_to_semantic(element: ET.Element) -> list[list[str]]:
    _remove_dynamics_attribute_from_nodes_recursive(element)
    root = _parse_node(element)
    result = []
    for part in root.get_children_of_type(mxl.XMLPart):
        semantic = _music_part_to_semantic(part)
        result.extend(semantic)
    return result


def music_xml_string_to_semantic(content: str) -> list[list[str]]:
    xml = ET.fromstring(content)  # noqa: S314
    return _music_xml_content_to_semantic(xml)


def music_xml_to_semantic(file_path: str) -> list[list[str]]:
    with open(file_path) as file:
        xml = ET.parse(file)  # noqa: S314
    return _music_xml_content_to_semantic(xml.getroot())


def group_in_measures(semantic: list[str]) -> tuple[str, list[list[str]]]:
    result: list[list[str]] = []
    clef = ""
    key = ""
    current_measure: list[str] = []
    for symbol in semantic:
        if symbol == "barline":
            current_measure.append(symbol)
            result.append(current_measure)
            current_measure = []
        else:
            current_measure.append(symbol)
            if symbol.startswith("clef"):
                clef = symbol
            elif symbol.startswith("keySignature"):
                key = symbol
    if len(current_measure) > 0:
        result.append(current_measure)
    prelude = clef + "+" + key + "+"
    return prelude, result
