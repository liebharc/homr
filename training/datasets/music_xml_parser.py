import xml.etree.ElementTree as ET

import musicxml.xmlelement.xmlelement as mxl
from musicxml.parser.parser import _parse_node

from homr.music_xml_generator import DURATION_NAMES
from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, empty
from training.transformer.training_vocabulary import VocabularyStats, check_token_lines

DURATION_NUMBER = {v: k for k, v in DURATION_NAMES.items()}

ARTIC_MAPPING: dict[str, str] = {
    "strongAccent": "accent",
    "softAccent": "accent",
    "invertedTurn": "turn",
    "trillMark": "trill",
    # trill and vibrato according to https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/wavy-line/
    "wavyLine": "trill",
    "invertedMordent": "trill",
    "accidentalMark": "",  # ignore this value
    "scoop": "",
    "doit": "",
    "caesura": "",
}


class SymbolWithPosition:
    def __init__(self, position: int, symbol: EncodedSymbol, insert_before: bool = False) -> None:
        self.position = position
        self.symbol = symbol
        self.insert_before = insert_before

    def __str__(self) -> str:
        return str(self.position) + " " + str(self.symbol) + " " + str(self.insert_before)

    def __repr__(self) -> str:
        return str(self)


class TokensMeasure:
    """
    MusicXML allows directions such as forward/backwards. For training we need
    to have a well defined sequence. So we linearize the MusicXML instrunctions
    using this class.
    """

    def __init__(self, number_of_clefs: int) -> None:
        self.staffs: list[list[SymbolWithPosition]] = [[] for _ in range(number_of_clefs)]
        self.current_position = 0
        self.slurred_tied_positions: list[set[int]] = [set() for _ in range(number_of_clefs)]

    def append_symbol(self, symbol: EncodedSymbol) -> None:
        if symbol.rhythm.startswith("note"):
            raise ValueError("Call append_note for notes")
        else:
            offset = 1 if "barline" in symbol.rhythm else 0
            for staff in self.staffs:
                staff.append(SymbolWithPosition(self.current_position + offset, symbol, True))

    def append_symbol_to_staff(self, staff: int, symbol: EncodedSymbol) -> None:
        if symbol.rhythm.startswith("note"):
            raise ValueError("Call append_note for notes")
        else:
            self.staffs[staff].append(SymbolWithPosition(self.current_position, symbol, True))

    def append_position_change(self, duration: int) -> None:
        if len(self.staffs) == 0:
            raise ValueError("Expected to get clefs as first symbol")
        new_position = self.current_position + duration
        if new_position < 0:
            raise ValueError(
                "Backup duration is too long " + str(self.current_position) + " " + str(duration)
            )
        self.current_position = new_position

    def append_rest(
        self, staff: int, is_chord: bool, duration: int, invisible: bool, symbol: EncodedSymbol
    ) -> None:
        self.append_note(staff, is_chord, duration, invisible, symbol)

    def append_note(
        self, staff: int, is_chord: bool, duration: int, invisible: bool, symbol: EncodedSymbol
    ) -> None:
        if len(self.staffs) == 0:
            raise ValueError("Expected to get clefs as first symbol")
        if is_chord:
            if len(self.staffs[staff]) == 0:
                raise ValueError("A chord requires a previous note")
            previous_symbol = self.staffs[staff][-1]
            if not invisible:
                self.staffs[staff].append(SymbolWithPosition(previous_symbol.position, symbol))
            self.current_position = previous_symbol.position + duration
        else:
            if not invisible:
                self.staffs[staff].append(SymbolWithPosition(self.current_position, symbol))
            self.current_position += duration

    def mark_position_a_slurred_tied(self, staff: int) -> None:
        self.slurred_tied_positions[staff].add(self.current_position)

    def _fill_in_arpeggiate(self, staffs: list[list[SymbolWithPosition]]) -> None:
        """
        In the Lieder dataset we find that an arpeggiate always is rendered
        as it would affect all notes in a chord, even if the MusicXML doesn't
        reflect that.
        """
        arpegiatted_positions = set()
        for staff in staffs:
            for symbol in staff:
                if "arpeggiate" in symbol.symbol.articulation:
                    arpegiatted_positions.add(symbol.position)

        for staff in staffs:
            for symbol in staff:
                if (
                    symbol.position in arpegiatted_positions
                    and "arpeggiate" not in symbol.symbol.articulation
                    and symbol.symbol.rhythm.startswith(("note", "rest"))
                    and not symbol.symbol.rhythm.startswith("note_0")
                    and not symbol.symbol.rhythm.startswith("rest_0")
                ):
                    art = symbol.symbol.articulation
                    art_parts = []
                    if art != empty:
                        art_parts = art.split("_")
                    art_parts.append("arpeggiate")
                    art = str.join("_", sorted(art_parts))
                    symbol.symbol.articulation = art

    def complete_measure(self) -> list[list[EncodedSymbol]]:  # noqa: C901
        self._fill_in_arpeggiate(self.staffs)
        result: list[list[EncodedSymbol]] = []
        for _staff_no, staff in enumerate(self.staffs):
            has_barline = False
            result_staff: list[EncodedSymbol] = []
            grouped_symbols: dict[int, list[SymbolWithPosition]] = {}
            for symbol in staff:
                if "barline" in symbol.symbol.rhythm:
                    has_barline = True
                if symbol.position not in grouped_symbols:
                    grouped_symbols[symbol.position] = []
                grouped_symbols[symbol.position].append(symbol)
            for position in sorted(grouped_symbols.keys()):
                group = grouped_symbols[position]
                group_pos = []
                for symbol in group:
                    if symbol.insert_before:
                        result_staff.append(symbol.symbol)
                    elif symbol.symbol.rhythm.endswith("G"):
                        # Grace notes come before a group
                        result_staff.append(symbol.symbol)
                    else:
                        group_pos.append(symbol.symbol)
                for i, symbol_in_group in enumerate(group_pos):
                    result_staff.append(symbol_in_group)
                    if i < len(group_pos) - 1:
                        result_staff.append(EncodedSymbol("chord"))
                # if position in self.slurred_tied_positions[staff_no]:
                #    result_staff.append(EncodedSymbol("tieSlur"))
            if not has_barline:
                result_staff.append(EncodedSymbol("barline"))
            result.append(result_staff)
        return result


class SlurTieState:
    def __init__(self) -> None:
        self.tie_count = 0
        self.slur_count = 0

    def handle_tie_type(self, slur_tie_type: str) -> None:
        if self._type_to_bool(slur_tie_type):
            self.tie_count += 1
        else:
            self.tie_count = max(0, self.tie_count - 1)

    def handle_slur_type(self, slur_tie_type: str) -> None:
        if self._type_to_bool(slur_tie_type):
            self.slur_count += 1
        else:
            self.slur_count -= 1

    def _type_to_bool(self, slur_tie_type: str) -> bool:
        if slur_tie_type == "start":
            return True
        elif slur_tie_type == "stop":
            return False
        else:
            raise ValueError("Unknown slur/tie type " + slur_tie_type)

    def is_in_slur_or_tie(self) -> bool:
        return self.slur_count > 0 or self.tie_count > 0


class TripletState:

    def __init__(self) -> None:
        self.started = False

    def get_triplet_factor(self, note: mxl.XMLNote) -> float:
        notations = note.get_children_of_type(mxl.XMLNotations)
        was_started = self.started
        if len(notations) > 0:
            tuplets = notations[0].get_children_of_type(mxl.XMLTuplet)
            print_object = notations[0].attributes.get("print-object", None)
            for t in tuplets:
                t_type = t.attributes.get("type", None)
                show_number = t.attributes.get("show-number", None)
                if t_type == "start" and show_number != "none" and print_object != "no":
                    self.started = True
                if t_type == "stop":
                    self.started = False

        if not self.started and not was_started:
            return 1.0
        time_modification = note.get_children_of_type(mxl.XMLTimeModification)
        if len(time_modification) == 0:
            return 1.0
        actual_notes = time_modification[0].get_children_of_type(mxl.XMLActualNotes)
        if len(actual_notes) == 0:
            return 1.0
        normal_notes = time_modification[0].get_children_of_type(mxl.XMLNormalNotes)
        if len(normal_notes) == 0:
            return 1.0
        actual = int(actual_notes[0].value_)
        normal = int(normal_notes[0].value_)
        return float(actual) / float(normal)


class TokensPart:
    def __init__(self) -> None:
        self.current_measure: TokensMeasure | None = None
        self.staffs: list[list[list[EncodedSymbol]]] = []
        self.triplets = TripletState()
        self.slur_tie: list[SlurTieState] = []

    def append_clefs(self, clefs: list[tuple[EncodedSymbol, int]]) -> None:
        current_measure = self.current_measure
        if current_measure is not None:
            for staff, clef in enumerate(clefs):
                if clef[1] >= 0:
                    current_measure.append_symbol_to_staff(clef[1], clef[0])
                else:
                    current_measure.append_symbol_to_staff(staff, clef[0])
        else:
            self.staffs = [[] for _ in range(len(clefs))]
            self.slur_tie = [SlurTieState() for _ in range(len(clefs))]
            measure = TokensMeasure(len(clefs))
            for staff, clef in enumerate(clefs):
                measure.append_symbol_to_staff(staff, clef[0])
            self.current_measure = measure

    def append_symbol(self, symbol: EncodedSymbol) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        self.current_measure.append_symbol(symbol)

    def append_rest(
        self, staff: int, is_chord: bool, duration: int, invisible: bool, symbol: EncodedSymbol
    ) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        self.current_measure.append_rest(staff, is_chord, duration, invisible, symbol)

    def append_note(
        self, staff: int, is_chord: bool, duration: int, invisible: bool, symbol: EncodedSymbol
    ) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        self.current_measure.append_note(staff, is_chord, duration, invisible, symbol)

    def mark_position_a_slurred_tied(self, staff: int) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        self.current_measure.mark_position_a_slurred_tied(staff)

    def append_position_change(self, duration: int) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        self.current_measure.append_position_change(duration)

    def is_in_slur_or_tie(self, staff: int) -> bool:
        return self.slur_tie[staff].is_in_slur_or_tie()

    def handle_tie_type(self, staff: int, slur_tie_type: str) -> None:
        self.slur_tie[staff].handle_tie_type(slur_tie_type)

    def handle_slur_type(self, staff: int, slur_tie_type: str) -> None:
        self.slur_tie[staff].handle_slur_type(slur_tie_type)

    def on_end_of_measure(self) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        for staff, result in enumerate(self.current_measure.complete_measure()):
            self.staffs[staff].append(result)
        self.current_measure = TokensMeasure(len(self.current_measure.staffs))

    def get_staffs(self) -> list[list[list[EncodedSymbol]]]:
        return self.staffs


def _lift_from_pitch_or_accidental(pitch: mxl.XMLPitch, note: mxl.XMLNote) -> str:
    # explicit courtesy accidental overrides calculated
    accs = note.get_children_of_type(mxl.XMLAccidental)
    if accs:
        v = accs[0].value_
        return {"sharp": "#", "flat": "b", "natural": "N"}.get(v, empty)
    alter = pitch.get_children_of_type(mxl.XMLAlter)
    if not alter:
        return empty
    val = int(alter[0].value_)
    return _alter_to_lifts(val)


def _count_dots(note: mxl.XMLNote) -> int:
    dots = note.get_children_of_type(mxl.XMLDot)
    return len(dots)


def _process_attributes(part: TokensPart, attribute: mxl.XMLAttributes) -> None:
    clefs = attribute.get_children_of_type(mxl.XMLClef)
    if len(clefs) > 0:
        clefs_tokens: list[tuple[EncodedSymbol, int]] = []
        for clef in clefs:
            sign = clef.get_children_of_type(mxl.XMLSign)[0].value_
            line = clef.get_children_of_type(mxl.XMLLine)[0].value_
            has_octave_change = len(clef.get_children_of_type(mxl.XMLClefOctaveChange)) > 0
            if has_octave_change:
                raise ValueError("Octave change isn't supported")
            clef_number = int(clef.attributes.get("number", "0")) - 1
            clefs_tokens.append((EncodedSymbol(f"clef_{sign}{line}"), clef_number))
        part.append_clefs(clefs_tokens)
    keys = attribute.get_children_of_type(mxl.XMLKey)
    if len(keys) > 0:
        fifths = keys[0].get_children_of_type(mxl.XMLFifths)[0].value_
        part.append_symbol(EncodedSymbol(f"keySignature_{int(fifths)}"))
    times = attribute.get_children_of_type(mxl.XMLTime)
    if len(times) > 0:
        beat_type = times[0].get_children_of_type(mxl.XMLBeatType)[0].value_
        part.append_symbol(EncodedSymbol(f"timeSignature/{beat_type}"))

    style = attribute.get_children_of_type(mxl.XMLMeasureStyle)
    if len(style) > 0:
        _process_multi_rests(part, style[0])


def _alter_to_lifts(alter: int) -> str:
    if alter < -1:
        return "bb"
    if alter == -1:
        return "b"
    if alter == 1:
        return "#"
    if alter > 1:
        return "##"
    return "N"


def _pitch_name(pitch: mxl.XMLPitch) -> str:
    step = pitch.get_children_of_type(mxl.XMLStep)[0].value_
    octave = pitch.get_children_of_type(mxl.XMLOctave)[0].value_
    return f"{step}{octave}"


def _rhythm_token(base: str, number: int, dots: int, is_grace: bool) -> str:
    dot_str = "." * min(dots, 2)
    grace = "G" if is_grace else ""
    return f"{base}_{number}{grace}{dot_str}"


def _collect_articulation(note: mxl.XMLNote, part: TokensPart, staff: int) -> str:
    notations = note.get_children_of_type(mxl.XMLNotations)
    if not notations:
        return empty
    articulations = []
    # pick the first articulation we support if multiple present
    for child in notations[0].get_children():
        print_object = child.attributes.get("print-object", None)
        invisible = print_object == "no"
        if isinstance(child, mxl.XMLArticulations) and not invisible:
            for a in child.get_children():
                name = a.__class__.__name__[3:]  # strip XML prefix, e.g., XMLStaccato -> Staccato
                name = name[0].lower() + name[1:]
                if name in ARTIC_MAPPING:
                    if ARTIC_MAPPING[name]:
                        articulations.append(ARTIC_MAPPING[name])
                else:
                    articulations.append(name)
        if isinstance(child, mxl.XMLFermata) and not invisible:
            articulations.append("fermata")
        if isinstance(child, mxl.XMLOrnaments) and not invisible:
            for o in child.get_children():
                nm = o.__class__.__name__[3:]
                nm = nm[0].lower() + nm[1:]
                if nm in ARTIC_MAPPING:
                    if ARTIC_MAPPING[nm]:
                        articulations.append(ARTIC_MAPPING[nm])
                else:
                    articulations.append(nm)
        if isinstance(child, mxl.XMLTuplet):
            # Tuplets are handled by TripletState
            pass
        if isinstance(child, mxl.XMLArpeggiate):
            articulations.append("arpeggiate")
        if isinstance(child, mxl.XMLTied):
            tie_type = child.attributes.get("type", "")
            part.handle_tie_type(staff, tie_type)
        if isinstance(child, mxl.XMLSlur):
            slur_type = child.attributes.get("type", "")
            part.handle_slur_type(staff, slur_type)
    articulations = list(set(articulations))
    if len(articulations) == 0:
        return empty
    return str.join("_", sorted(articulations))


def _process_note(part: TokensPart, note: mxl.XMLNote) -> None:
    staff = 0
    note_heads = note.get_children_of_type(mxl.XMLNotehead)
    for note_head in note_heads:
        if note_head.value_ == "none":
            # Notehead is not printed
            return
    staff_nodes = note.get_children_of_type(mxl.XMLStaff)
    print_object = note.attributes.get("print-object", None)
    invisible = print_object == "no"
    if len(staff_nodes) > 0:
        staff = int(staff_nodes[0].value_) - 1
    is_grace = len(note.get_children_of_type(mxl.XMLGrace)) > 0
    is_chord = len(note.get_children_of_type(mxl.XMLChord)) > 0
    triplet_factor = part.triplets.get_triplet_factor(note)
    if len(note.get_children_of_type(mxl.XMLDuration)) == 0:
        is_grace_note = len(note.get_children_of_type(mxl.XMLGrace)) > 0
        if not is_grace_note:
            eprint("Note without duration", note.get_children())
        duration = 0
    else:
        # Note: The duration in MusicXML is in the unit "divisions"
        # divisions are specified in the parts attributes
        duration = int(note.get_children_of_type(mxl.XMLDuration)[0].value_)
    rest = note.get_children_of_type(mxl.XMLRest)
    dots = _count_dots(note)
    dur_nodes = note.get_children_of_type(mxl.XMLType)
    duration_type = dur_nodes[0].value_ if dur_nodes else "eighth"
    base_duration = round(DURATION_NUMBER[duration_type] * triplet_factor)
    art = _collect_articulation(note, part, staff)
    if part.is_in_slur_or_tie(staff):
        part.mark_position_a_slurred_tied(staff)
    if len(rest) > 0:
        if rest[0] and rest[0].attributes.get("measure", None):
            part.append_rest(
                staff, is_chord, duration, invisible, EncodedSymbol("rest_0", empty, empty, empty)
            )
        else:
            rhythm = _rhythm_token("rest", base_duration, dots, is_grace)
            sym = EncodedSymbol(rhythm, empty, empty, art)
            part.append_rest(staff, is_chord, duration, invisible, sym)
    pitch = note.get_children_of_type(mxl.XMLPitch)
    if len(pitch) > 0:
        pitch_name = _pitch_name(pitch[0])
        lift = _lift_from_pitch_or_accidental(pitch[0], note)
        rhythm = _rhythm_token("note", base_duration, dots, is_grace)
        sym = EncodedSymbol(rhythm, pitch_name, lift, art)

        part.append_note(staff, is_chord, max(duration, 1), invisible, sym)


def _process_backup(part: TokensPart, backup: mxl.XMLBackup) -> None:
    backup_value = int(backup.get_children_of_type(mxl.XMLDuration)[0].value_)
    part.append_position_change(-backup_value)


def _process_forward(part: TokensPart, backup: mxl.XMLBackup) -> None:
    forward_value = int(backup.get_children_of_type(mxl.XMLDuration)[0].value_)
    part.append_position_change(forward_value)


def _process_barline(part: TokensPart, barline: mxl.XMLBarline) -> None:
    bar_style = ""  # style: light-heavy, light-light, "heave-heavy"
    bar_style_nodes = barline.get_children_of_type(mxl.XMLBarStyle)
    if len(bar_style_nodes) > 0:
        bar_style = bar_style_nodes[0].value_

    direction = ""
    repeat_nodes = barline.get_children_of_type(mxl.XMLRepeat)
    if len(repeat_nodes) > 0:
        direction = repeat_nodes[0].attributes.get("direction", "")

    if direction == "forward":
        part.append_symbol(EncodedSymbol("repeatStart"))
    elif direction == "backward":
        part.append_symbol(EncodedSymbol("repeatEnd"))
    elif "heavy" in bar_style:
        part.append_symbol(EncodedSymbol("bolddoublebarline"))
    elif "light" in bar_style:
        part.append_symbol(EncodedSymbol("doublebarline"))
    else:
        # "barline" elments without style or repeat are automatically added with measures
        pass


def _process_multi_rests(part: TokensPart, measure_style: mxl.XMLMeasureStyle) -> None:
    rests = measure_style.get_children_of_type(mxl.XMLMultipleRest)
    if len(rests) == 0:
        return
    rest = rests[0]
    rest_duration = min(rest.value_, 10)
    part.append_symbol(EncodedSymbol(f"rest_{rest_duration}m", empty, empty, empty))


def _music_part_to_tokens(part: mxl.XMLPart) -> list[list[list[EncodedSymbol]]]:
    tokens = TokensPart()
    for measure in part.get_children_of_type(mxl.XMLMeasure):
        for child in measure.get_children():
            if isinstance(child, mxl.XMLAttributes):
                _process_attributes(tokens, child)
            if isinstance(child, mxl.XMLNote):
                _process_note(tokens, child)
            if isinstance(child, mxl.XMLBackup):
                _process_backup(tokens, child)
            if isinstance(child, mxl.XMLForward):
                _process_forward(tokens, child)
            if isinstance(child, mxl.XMLBarline):
                _process_barline(tokens, child)
        tokens.on_end_of_measure()
    return tokens.get_staffs()


def music_xml_element_to_symbols(
    root: ET.Element,
) -> list[list[list[EncodedSymbol]]]:
    _remove_dynamics_attribute_from_nodes_recursive(root)
    parsed = _parse_node(root)
    result: list[list[list[EncodedSymbol]]] = []
    for part in parsed.get_children_of_type(mxl.XMLPart):
        tokens = _music_part_to_tokens(part)
        result.extend(tokens)
    return result


def music_xml_string_to_tokens(content: str) -> list[list[list[EncodedSymbol]]]:
    xml = ET.fromstring(content)  # noqa: S314
    return music_xml_element_to_symbols(xml)


def music_xml_file_to_tokens(file_path: str) -> list[list[list[EncodedSymbol]]]:
    with open(file_path, "rb") as f:
        xml = ET.parse(f)  # noqa: S314
    return music_xml_element_to_symbols(xml.getroot())


def _remove_dynamics_attribute_from_nodes_recursive(node: ET.Element) -> None:
    """
    We don't need the dynamics attribute in the XML, but XSD validation
    sometimes fails if its negative. So we remove it.
    """
    if "dynamics" in node.attrib:
        del node.attrib["dynamics"]

    for child in list(node):
        # If the node is a <metronome> tag, remove it from its parent
        if child.tag == "metronome":
            node.remove(child)
        _remove_dynamics_attribute_from_nodes_recursive(child)


if __name__ == "__main__":
    import glob
    import multiprocessing
    import os
    import sys

    from homr.simple_logging import eprint

    if len(sys.argv) > 1:
        result = music_xml_file_to_tokens(sys.argv[1])
        for line in result:
            eprint(line)
        sys.exit(0)

    stats = VocabularyStats()
    files = glob.glob(
        os.path.join("datasets", "Lieder-main", "**", "**.musicxml"),
        recursive=True,
    )

    def process_file(file: str) -> tuple[str, list[list[EncodedSymbol]], Exception | None]:
        try:
            voices = music_xml_file_to_tokens(file)
            tokens_list = []
            for staffs in voices:
                for tokens in staffs:
                    tokens_list.append(tokens)
            return (file, tokens_list, None)
        except Exception as e:
            return (file, [], e)

    errors = set()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for file, tokens_list, err in pool.imap_unordered(process_file, sorted(files)):
            if err:
                eprint("Error while processing", file, err)
            else:
                eprint("Finished", file)
                for tokens in tokens_list:
                    stats.add_lines(tokens)
                    try:
                        check_token_lines(tokens)
                    except Exception as e:
                        errors.add(str(e))
                        eprint(e)
    eprint("Stats", stats)
    if len(errors) > 0:
        eprint("There have been errors", errors)
        sys.exit(1)
