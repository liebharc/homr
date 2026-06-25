import xml.etree.ElementTree as ET
from typing import Iterable, SupportsIndex, TypeVar, overload

from homr.music_xml_generator import DURATION_NAMES
from homr.simple_logging import eprint
from homr.transformer.vocabulary import (
    EncodedSymbol,
    empty,
    has_rhythm_symbol_a_position,
)
from training.omr_datasets.staff_merging import (
    EncodedSymbolWithPos,
    merge_upper_and_lower_staff,
)
from training.transformer.training_vocabulary import VocabularyStats, check_token_lines

_T = TypeVar("_T")


def _children(el: ET.Element, tag: str) -> list[ET.Element]:
    return el.findall(tag)


def _child(el: ET.Element, tag: str) -> ET.Element | None:
    return el.find(tag)


def _text(el: ET.Element | None, default: str = "") -> str:
    if el is None or el.text is None:
        return default
    return el.text.strip()


def _int_text(el: ET.Element | None, default: int = 0) -> int:
    t = _text(el)
    if t == "":
        return default
    return int(t)


class Measure(list[EncodedSymbol]):
    """A list-like container that stores EncodedSymbol objects."""

    def __init__(self, iterable: Iterable[EncodedSymbol] = ()) -> None:
        """Initialize Measure from an iterable of EncodedSymbol objects."""
        super().__init__(iterable)
        self.new_page = False

    @overload
    def __setitem__(self, index: SupportsIndex, item: EncodedSymbol, /) -> None: ...

    @overload
    def __setitem__(self, index: slice, item: Iterable[EncodedSymbol], /) -> None: ...

    def __setitem__(
        self, index: SupportsIndex | slice, item: EncodedSymbol | Iterable[EncodedSymbol], /
    ) -> None:
        """Set item(s) at index."""
        if isinstance(index, slice):
            # Type narrowing: if index is slice, item must be Iterable
            super().__setitem__(index, list(item) if not isinstance(item, list) else item)  # type: ignore[arg-type]
        else:
            # Type narrowing: if index is not slice, item is EncodedSymbol
            super().__setitem__(index, item)  # type: ignore

    @overload
    def __add__(self, other: list[EncodedSymbol], /) -> "Measure": ...

    @overload
    def __add__(self, other: list[_T], /) -> list[EncodedSymbol | _T]: ...

    def __add__(self, other: list, /) -> "Measure | list":
        """Return a new Measure combining this one with another list."""
        result = Measure(self)
        result.extend(other)
        return result

    def __mul__(self, other: SupportsIndex, /) -> "Measure":
        """Return a new Measure with items repeated."""
        return Measure(super().__mul__(other))

    __rmul__ = __mul__

    def __repr__(self) -> str:
        """Return a string representation of the Measure."""
        return f"Measure({list(self)!r})"

    def copy(self) -> "Measure":
        """Return a shallow copy of the Measure."""
        return Measure(self)


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
    "otherOrnament": "",
    "otherArticulation": "",
}

# Tags inside <articulations> that we don't otherwise special-case.
_ARTICULATIONS_TAG = "articulations"
_ORNAMENTS_TAG = "ornaments"


def _xml_name_to_camel(tag: str) -> str:
    """Convert a hyphenated MusicXML tag (e.g. 'strong-accent') to camelCase
    (e.g. 'strongAccent'), matching what the old musicxml package's class
    names produced after stripping the XML prefix."""
    parts = tag.split("-")
    if len(parts) == 1:
        return parts[0]
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:] if p)


class TokensMeasure:
    """
    MusicXML allows directions such as forward/backwards. For training we need
    to have a well defined sequence. So we linearize the MusicXML instrunctions
    using this class.
    """

    def __init__(self) -> None:
        self.symbols: list[EncodedSymbolWithPos] = []
        self.current_position = 0
        self.new_page = False

    def append_symbol(self, symbol: EncodedSymbol) -> None:
        if symbol.rhythm.startswith("note"):
            raise ValueError("Call append_note for notes")
        else:
            self.current_position += 1
            offset = 1 if "barline" in symbol.rhythm else 0
            self.symbols.append(EncodedSymbolWithPos(self.current_position + offset, symbol, True))

    def mark_new_page(self) -> None:
        self.new_page = True

    def append_symbol_to_staff(self, staff: int, symbol: EncodedSymbol) -> None:
        if has_rhythm_symbol_a_position(symbol.rhythm):
            symbol.position = self._get_staff_position(staff)
        if symbol.rhythm.startswith("note"):
            raise ValueError("Call append_note for notes")
        else:
            self.symbols.append(EncodedSymbolWithPos(self.current_position, symbol, True))

    def append_position_change(self, duration: int) -> None:
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
        is_grace = "G" in symbol.rhythm
        symbol.position = self._get_staff_position(staff)
        if is_chord:
            previous_symbol = self.symbols[-1]
            if not invisible:
                self.symbols.append(
                    EncodedSymbolWithPos(previous_symbol.position, symbol, insert_before=is_grace)
                )
            self.current_position = previous_symbol.position + duration
        else:
            if not invisible:
                self.symbols.append(
                    EncodedSymbolWithPos(self.current_position, symbol, insert_before=is_grace)
                )
            self.current_position += duration

    def _get_staff_no(self, symbol: EncodedSymbolWithPos) -> int:
        if symbol.symbol.position == "lower":
            return 1
        return 0

    def _get_staff_position(self, staff: int) -> str:
        if staff == 0:
            return "upper"
        return "lower"

    def _fill_in_arpeggiate(self, symbols_with_pos: list[EncodedSymbolWithPos]) -> None:
        """
        In the Lieder dataset we find that an arpeggiate always is rendered
        as it would affect all notes in a chord, even if the MusicXML doesn't
        reflect that.
        """
        arpegiatted_positions = set()
        for entry in symbols_with_pos:
            pos, sym = entry.position, entry.symbol
            if "arpeggiate" in sym.articulation:
                # `pos` is the position in the measure, `sym.position` is the upper/lower staff.
                arpegiatted_positions.add((pos, sym.position))

        for entry in symbols_with_pos:
            pos, sym = entry.position, entry.symbol
            if (
                (pos, sym.position) in arpegiatted_positions
                and "arpeggiate" not in sym.articulation
                and sym.rhythm.startswith(("note", "rest"))
                and not sym.rhythm.startswith("note_0")
                and not sym.rhythm.startswith("rest_0")
            ):
                art = sym.articulation
                art_parts = []
                if art != empty:
                    art_parts = art.split("_")
                art_parts.append("arpeggiate")
                art = str.join("_", sorted(art_parts))
                sym.articulation = art

    def complete_measure(self) -> Measure:  # noqa: C901
        self._fill_in_arpeggiate(self.symbols)
        result_staff: list[list[EncodedSymbolWithPos]] = [[], []]
        grouped_symbols: dict[int, list[EncodedSymbolWithPos]] = {}
        for symbol in self.symbols:
            sort_order = symbol.sort_order()
            if sort_order not in grouped_symbols:
                grouped_symbols[sort_order] = []
            grouped_symbols[sort_order].append(symbol)
        for sort_order in sorted(grouped_symbols.keys()):
            group = grouped_symbols[sort_order]
            group_pos: list[EncodedSymbolWithPos] = []
            for symbol in group:
                if symbol.insert_before:
                    result_staff[self._get_staff_no(symbol)].append(symbol)
                elif symbol.symbol.rhythm.endswith("G"):
                    # Grace notes come before a group
                    result_staff[self._get_staff_no(symbol)].append(symbol)
                else:
                    group_pos.append(symbol)
            for symbol_in_group in group_pos:
                result_staff[self._get_staff_no(symbol_in_group)].append(symbol_in_group)

        result_measure = Measure(merge_upper_and_lower_staff(result_staff))
        result_measure.new_page = self.new_page
        return result_measure


class TupletState:

    def __init__(self) -> None:
        self.started = False
        self.last_stop_position = -1

    def get_tuplet_factor(self, note: ET.Element, position: int) -> float:
        notations = _children(note, "notations")
        was_started = self.started or self.last_stop_position == position
        if len(notations) > 0:
            tuplets = _children(notations[0], "tuplet")
            print_object = notations[0].get("print-object", None)
            for t in tuplets:
                t_type = t.get("type", None)
                show_number = t.get("show-number", None)
                if t_type == "start" and show_number != "none" and print_object != "no":
                    self.started = True
                if t_type == "stop":
                    self.started = False
                    self.last_stop_position = position

        if not self.started and not was_started:
            return 1.0
        time_modification = _children(note, "time-modification")
        if len(time_modification) == 0:
            return 1.0
        actual_notes = _children(time_modification[0], "actual-notes")
        if len(actual_notes) == 0:
            return 1.0
        normal_notes = _children(time_modification[0], "normal-notes")
        if len(normal_notes) == 0:
            return 1.0
        actual = _int_text(actual_notes[0])
        normal = _int_text(normal_notes[0])
        return float(actual) / float(normal)

    def on_end_of_measure(self) -> None:
        self.last_stop_position = -1


def _measure_rest_rhythm(duration: int, divisions: int) -> str:
    if divisions <= 0:
        return "rest_1"
    quarters = duration / divisions
    lookup = [
        (4.0, "rest_1"),
        (3.0, "rest_2."),
        (2.0, "rest_2"),
        (1.5, "rest_4."),
        (1.0, "rest_4"),
        (0.75, "rest_8."),
        (0.5, "rest_8"),
    ]
    for q, token in lookup:
        if abs(quarters - q) < 0.01:
            return token
    return "rest_1"


class TokensPart:
    def __init__(self) -> None:
        self.current_measure: TokensMeasure | None = None
        self.measures: list[Measure] = []
        self.tuplets = TupletState()
        self.tremolo = False
        self.divisions: int = 1

    def append_clefs(self, clefs: list[tuple[EncodedSymbol, int]]) -> None:
        current_measure = self.current_measure
        if current_measure is not None:
            for staff, clef in enumerate(clefs):
                if clef[1] >= 0:
                    current_measure.append_symbol_to_staff(clef[1], clef[0])
                else:
                    current_measure.append_symbol_to_staff(staff, clef[0])
        else:
            measure = TokensMeasure()
            for staff, clef in enumerate(clefs):
                measure.append_symbol_to_staff(staff, clef[0])
            self.current_measure = measure

    def append_symbol(self, symbol: EncodedSymbol) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        self.current_measure.append_symbol(symbol)

    def mark_new_page(self) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        self.current_measure.mark_new_page()

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

    def append_position_change(self, duration: int) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")
        self.current_measure.append_position_change(duration)

    def on_end_of_measure(self) -> None:
        if self.current_measure is None:
            raise ValueError("Expected to get clefs as first symbol")

        self.measures.append(self.current_measure.complete_measure())
        self.current_measure = TokensMeasure()
        self.tuplets.on_end_of_measure()

    def get_measures(self) -> list[Measure]:
        return self.measures

    def _get_current_position(self) -> int:
        if self.current_measure is None:
            return 0
        return self.current_measure.current_position

    def get_tuplet_factor(self, note: ET.Element, is_chord: bool, duration: int) -> float:
        position = self._get_current_position()
        if is_chord:
            position -= duration
        return self.tuplets.get_tuplet_factor(note, position)


def _lift_from_pitch_or_accidental(pitch: ET.Element, note: ET.Element) -> str:
    # explicit courtesy accidental overrides calculated
    accs = _children(note, "accidental")
    if accs:
        v = _text(accs[0])
        return {"sharp": "#", "flat": "b", "natural": "N"}.get(v, empty)
    alter = _child(pitch, "alter")
    if alter is None:
        return empty
    val = int(float(_text(alter, "0")))
    return _alter_to_lifts(val)


def _count_dots(note: ET.Element) -> int:
    dots = _children(note, "dot")
    return len(dots)


def _process_attributes(part: TokensPart, attribute: ET.Element) -> None:
    divs = _children(attribute, "divisions")
    if len(divs) > 0:
        part.divisions = _int_text(divs[0], 1) or 1
    clefs = _children(attribute, "clef")
    if len(clefs) > 0:
        clefs_tokens: list[tuple[EncodedSymbol, int]] = []
        for clef in clefs:
            sign = _text(_child(clef, "sign"))
            line = _text(_child(clef, "line"))
            has_octave_change = _child(clef, "clef-octave-change") is not None
            if has_octave_change:
                raise ValueError("Octave change isn't supported")
            clef_number = int(clef.get("number", "1")) - 1
            clefs_tokens.append(
                (EncodedSymbol(f"clef_{sign}{line}", empty, empty, empty, empty), clef_number)
            )
        part.append_clefs(clefs_tokens)
    keys = _children(attribute, "key")
    times = _children(attribute, "time")
    if len(keys) > 0:
        fifths = _text(_child(keys[0], "fifths"), "0")
        part.append_symbol(EncodedSymbol(f"keySignature_{int(fifths)}"))
    if len(times) > 0:
        beat_type = _text(_child(times[0], "beat-type"))
        part.append_symbol(EncodedSymbol(f"timeSignature/{beat_type}"))

    style = _children(attribute, "measure-style")
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


def _pitch_name(pitch: ET.Element) -> str:
    step = _text(_child(pitch, "step"))
    octave = _text(_child(pitch, "octave"))
    return f"{step}{octave}"


def _rhythm_token(base: str, number: int, dots: int, is_grace: bool) -> str:
    dot_str = "." * min(dots, 2)
    grace = "G" if is_grace else ""
    return f"{base}_{number}{grace}{dot_str}"


def _collect_articulation(note: ET.Element, part: TokensPart, staff: int) -> tuple[str, str]:
    notations_list = _children(note, "notations")
    if not notations_list:
        return empty, empty
    notations = notations_list[0]
    if notations.get("print-object", None) == "no":
        return empty, empty
    articulations = []
    slurs = []
    # pick the first articulation we support if multiple present
    for child in list(notations):
        print_object = child.get("print-object", None)
        invisible = print_object == "no"
        if child.tag == _ARTICULATIONS_TAG and not invisible:
            for a in list(child):
                a_tag = _xml_name_to_camel(a.tag)
                if a.tag == "tremolo" and not invisible:
                    tremolo_type = str(a.get("type", ""))
                    part.tremolo = tremolo_type == "start"
                else:
                    name = a_tag[0].lower() + a_tag[1:]
                    if name in ARTIC_MAPPING:
                        if ARTIC_MAPPING[name]:
                            articulations.append(ARTIC_MAPPING[name])
                    else:
                        articulations.append(name)
        if child.tag == "fermata" and not invisible:
            articulations.append("fermata")
        if child.tag == _ORNAMENTS_TAG and not invisible:
            for o in list(child):
                o_tag = _xml_name_to_camel(o.tag)
                nm = o_tag[0].lower() + o_tag[1:]
                if nm in ARTIC_MAPPING:
                    if ARTIC_MAPPING[nm]:
                        articulations.append(ARTIC_MAPPING[nm])
                else:
                    articulations.append(nm)
        if child.tag == "tuplet":
            # Tuplets are handled by TupletState
            pass
        if child.tag == "arpeggiate":
            articulations.append("arpeggiate")
        if child.tag == "tied":
            tie_type = str(child.get("type", ""))
            slurs.append("slur" + tie_type.capitalize())
        if child.tag == "slur":
            slur_type = str(child.get("type", ""))
            slurs.append("slur" + slur_type.capitalize())

    if part.tremolo:
        articulations.append("tremolo")

    articulations = list(set(articulations))
    if len(articulations) == 0 and len(slurs) == 0:
        return empty, empty

    if len(articulations) == 0:
        return empty, str.join("_", sorted(slurs))

    if len(slurs) == 0:
        return str.join("_", sorted(articulations)), empty

    return str.join("_", sorted(articulations)), str.join("_", sorted(slurs))


def _process_note(part: TokensPart, note: ET.Element) -> None:
    staff = 0
    note_heads = _children(note, "notehead")
    for note_head in note_heads:
        if _text(note_head) == "none":
            # Notehead is not printed
            return
    staff_nodes = _children(note, "staff")
    print_object = note.get("print-object", None)
    invisible = print_object == "no"
    if len(staff_nodes) > 0:
        staff = _int_text(staff_nodes[0], 1) - 1
    is_grace = _child(note, "grace") is not None
    is_chord = _child(note, "chord") is not None
    duration_node = _child(note, "duration")
    if duration_node is None:
        is_grace_note = _child(note, "grace") is not None
        if not is_grace_note:
            eprint("Note without duration", list(note))
        duration = 0
    else:
        # Note: The duration in MusicXML is in the unit "divisions"
        # divisions are specified in the parts attributes
        duration = _int_text(duration_node)
    triplet_factor = part.get_tuplet_factor(note, is_chord, duration)
    rest = _children(note, "rest")
    dots = _count_dots(note)
    dur_nodes = _children(note, "type")
    duration_type = _text(dur_nodes[0]) if dur_nodes else "eighth"
    base_duration = round(DURATION_NUMBER[duration_type] * triplet_factor)
    art, slur = _collect_articulation(note, part, staff)
    if len(rest) > 0:
        if rest[0] is not None and rest[0].get("measure", None):
            rhythm = _measure_rest_rhythm(duration, part.divisions)
            part.append_rest(
                staff,
                is_chord,
                duration,
                invisible,
                EncodedSymbol(rhythm, empty, empty, empty, empty),
            )
        else:
            rhythm = _rhythm_token("rest", base_duration, dots, is_grace)
            sym = EncodedSymbol(rhythm, empty, empty, art, slur)
            part.append_rest(staff, is_chord, duration, invisible, sym)
    pitch = _children(note, "pitch")
    if len(pitch) > 0:
        pitch_name = _pitch_name(pitch[0])
        lift = _lift_from_pitch_or_accidental(pitch[0], note)
        rhythm = _rhythm_token("note", base_duration, dots, is_grace)
        sym = EncodedSymbol(rhythm, pitch_name, lift, art, slur)

        part.append_note(staff, is_chord, max(duration, 1), invisible, sym)


def _process_backup(part: TokensPart, backup: ET.Element) -> None:
    backup_value = _int_text(_child(backup, "duration"))
    part.append_position_change(-backup_value)


def _process_forward(part: TokensPart, forward: ET.Element) -> None:
    forward_value = _int_text(_child(forward, "duration"))
    part.append_position_change(forward_value)


def _process_barline(part: TokensPart, barline: ET.Element) -> None:
    bar_style = ""  # style: light-heavy, light-light, "heave-heavy"
    bar_style_nodes = _children(barline, "bar-style")
    if len(bar_style_nodes) > 0:
        bar_style = _text(bar_style_nodes[0])

    direction = ""
    repeat_nodes = _children(barline, "repeat")
    if len(repeat_nodes) > 0:
        direction = repeat_nodes[0].get("direction", "")

    ending = ""
    ending_nodes = _children(barline, "ending")
    if len(ending_nodes) > 0:
        ending = ending_nodes[0].get("type", "")

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

    if ending == "stop":
        part.append_symbol(EncodedSymbol("voltaStop"))
    elif ending == "discontinue":
        part.append_symbol(EncodedSymbol("voltaDiscontinue"))
    elif ending == "start":
        part.append_symbol(EncodedSymbol("voltaStart"))


def _process_print(part: TokensPart, xmlprint: ET.Element) -> None:
    new_page = xmlprint.get("new-page", "")
    if new_page == "yes":
        part.mark_new_page()


def _process_direction(part: TokensPart, xmldirection: ET.Element) -> None:
    for direction_type in _children(xmldirection, "direction-type"):
        has_octave_shift = _child(direction_type, "octave-shift") is not None
        if has_octave_shift:
            raise ValueError("Octave shift isn't supported")


def _process_multi_rests(part: TokensPart, measure_style: ET.Element) -> None:
    rests = _children(measure_style, "multiple-rest")
    if len(rests) == 0:
        return
    rest = rests[0]
    rest_duration = min(_int_text(rest), 10)
    part.append_symbol(EncodedSymbol(f"rest_{rest_duration}m", empty, empty, empty, empty, "upper"))


def _music_part_to_tokens(part: ET.Element) -> list[Measure]:
    tokens = TokensPart()
    for measure in _children(part, "measure"):
        for child in list(measure):
            if child.tag == "attributes":
                _process_attributes(tokens, child)
            if child.tag == "note":
                _process_note(tokens, child)
            if child.tag == "backup":
                _process_backup(tokens, child)
            if child.tag == "forward":
                _process_forward(tokens, child)
            if child.tag == "barline":
                _process_barline(tokens, child)
            if child.tag == "print":
                _process_print(tokens, child)
            if child.tag == "direction":
                _process_direction(tokens, child)
        tokens.on_end_of_measure()
    return _cleanup_barlines_and_repeats(tokens.get_measures())


def _cleanup_barlines_and_repeats(measures: list[Measure]) -> list[Measure]:
    """
    Normalize measure-ending barlines and adjacent repeat symbols.
    """

    def is_barline_or_repeat(symbol: EncodedSymbol) -> bool:
        return "barline" in symbol.rhythm or "repeat" in symbol.rhythm

    def is_barline_repeat_or_other(symbol: EncodedSymbol) -> str:
        if symbol.rhythm.startswith("repeat"):
            return "repeat"
        if "barline" in symbol.rhythm:
            return "barline"
        return "other"

    def can_merge(a: EncodedSymbol, b: EncodedSymbol) -> bool:
        a_cat = is_barline_repeat_or_other(a)
        b_cat = is_barline_repeat_or_other(b)
        if a_cat == "other" or b_cat == "other":
            return False
        if a_cat == "barline" and b_cat == "barline":
            return False
        return True

    def merge_barlines_and_repeats(a: EncodedSymbol, b: EncodedSymbol) -> EncodedSymbol:
        if (a.rhythm == "repeatStart" and b.rhythm == "repeatEnd") or (
            a.rhythm == "repeatEnd" and b.rhythm == "repeatStart"
        ):
            return EncodedSymbol("repeatEndStart")
        if "repeat" in a.rhythm:
            return a
        return b

    last_symbol = EncodedSymbol("")
    result: list[Measure] = []
    for measure in measures:
        measure_result: Measure = Measure()
        measure_result.new_page = measure.new_page
        for symbol in measure:
            if can_merge(symbol, last_symbol):
                merged = merge_barlines_and_repeats(symbol, last_symbol)
                if len(measure_result) == 0:
                    result[-1][-1] = merged
                else:
                    measure_result[-1] = merged
                last_symbol = merged
            else:
                measure_result.append(symbol)
                last_symbol = symbol
        if len(measure_result) == 0 or not is_barline_or_repeat(measure_result[-1]):
            measure_result.append(EncodedSymbol("barline"))
        result.append(measure_result)
    return result


def _music_xml_element_to_symbols(
    root: ET.Element,
) -> list[list[Measure]]:
    # Support both <score-partwise> as the root and as a wrapped document.
    score_root = root
    result: list[list[Measure]] = []
    for part in _children(score_root, "part"):
        tokens = _music_part_to_tokens(part)
        result.append(tokens)
    return result


def music_xml_string_to_tokens(content: str) -> list[list[Measure]]:
    """
    Returns a list of voices.
    Each voice is a list of measures.
    """
    xml = ET.fromstring(content)  # noqa: S314
    return _music_xml_element_to_symbols(xml)


def music_xml_file_to_tokens(file_path: str) -> list[list[Measure]]:
    with open(file_path, "rb") as f:
        xml = ET.parse(f)  # noqa: S314
    return _music_xml_element_to_symbols(xml.getroot())


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

    def process_file(file: str) -> tuple[str, list[Measure], Exception | None]:
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
