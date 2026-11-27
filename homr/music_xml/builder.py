"""Tokens -> model, one handler per kind of token group."""

from collections import defaultdict
from collections.abc import Callable
from fractions import Fraction

from homr import constants
from homr.music_xml import mapping
from homr.music_xml import model as m
from homr.music_xml.tokens import (
    SymbolChord,
    TimeSignature,
    division_and_measure_lengths,
    group_into_chords,
    starts_with_time_signature,
    time_signature_length,
    time_signature_of,
)
from homr.simple_logging import eprint
from homr.transformer.vocabulary import (
    OCTAVE_SHIFT_STOP,
    EncodedSymbol,
    empty,
    is_lower_position,
    nonote,
    octaves_of_clef,
    octaves_of_shift,
    split_slur_and_beam,
    transpose_octaves,
)


def get_staff(symbol: EncodedSymbol) -> int:
    return 2 if is_lower_position(symbol.position) else 1


def get_xml_voice(staff_num: int, rhythmic_layer: int) -> int:
    """Voice numbers are part-global: staff 1 has voices 1-4, staff 2 voices 5-8."""
    return (staff_num - 1) * 4 + rhythmic_layer + 1


def kind_of(group: SymbolChord) -> str:
    """The handler for a group, by its first symbol."""
    symbol = group.symbols[0]
    rhythm = symbol.rhythm
    if rhythm.startswith(("note", "rest")):
        return "multi_rest" if len(group.symbols) == 1 and rhythm.endswith("m") else "notes"
    if rhythm in ("measureRest", "newline"):
        return rhythm
    for prefix in ("clef", "keySignature", "timeSignature"):
        if rhythm.startswith(prefix):
            return prefix
    if group.is_barline():
        return "barline"
    if rhythm.startswith(("voltaStart", "voltaStop", "voltaDiscontinue")):
        return "volta"
    if rhythm.startswith(("dynamic", "octaveShift")) or symbol.articulation.startswith("dynamic"):
        return "marking"
    return "unsupported"


class PartBuilder:
    """
    Lays out the token groups of one part into measures. Each kind of group has
    a handler, the state between groups lives on the builder.
    """

    def __init__(
        self,
        voice: list[EncodedSymbol],
        part: m.Part,
        metronome: m.Metronome | None,
        inherited_time: TimeSignature | None,
    ) -> None:
        self.part = part
        self.groups = group_into_chords(voice)
        self.division, self.nominators, self.leading_length = division_and_measure_lengths(
            self.groups
        )
        self.nominator = self.leading_length
        # Without a printed time signature at its start, a page continues in the previous
        # page's one: the model reads only the beat type, so an estimate from the measure
        # lengths can't tell 2/2 from 4/4, and would bring a time change at every page.
        inherited_length = time_signature_length(inherited_time)
        if inherited_length is not None and not starts_with_time_signature(self.groups):
            self.nominator = inherited_length
        else:
            inherited_time = None
        self.inherited_time = inherited_time
        self.beats = 4 * constants.duration_of_quarter
        self.time_signature_no = 0
        self.volta_number = 1
        self.last_volta_measure = -10
        self.clef_octaves: dict[int, int] = {}
        self.shift_octaves: dict[int, int] = {}
        self.time = 0  # of the current group, in divisions
        self.rest_measures = 0  # measures a multi-measure rest still covers
        self.measure = m.Measure(1)
        part.divisions = self.division // 4
        if metronome is not None:
            self.measure.items.append(metronome)
        self.handlers: dict[str, Callable[[SymbolChord, int], None]] = {
            "notes": self.on_notes,
            "multi_rest": self.on_multi_rest,
            "measureRest": self.on_measure_rest,
            "newline": self.on_newline,
            "clef": self.on_clef,
            "keySignature": self.on_key,
            "timeSignature": self.on_time,
            "barline": self.on_barline,
            "volta": self.on_volta,
            "marking": self.on_marking,
            "unsupported": lambda group, _: eprint("Symbol isn't supported yet ", group.symbols[0]),
        }

    def build(self) -> list[m.Measure]:
        for index, group in enumerate(self.groups):
            self.time = self.divisions(group.start, minimum=0)
            self.handlers[kind_of(group)](group, index)
        measures = self.part.measures
        if self.measure.has_notes() or not measures:
            self.stop_octave_shifts(self.measure)
            self.close_measure()
        else:
            self.stop_octave_shifts(measures[-1])
        if not any(isinstance(item, m.Time) for item in measures[0].items):
            if self.inherited_time is not None:
                measures[0].items.append(m.Time(*self.inherited_time))
            else:
                beats, beat_type = time_signature_of("timeSignature/4", self.leading_length)
                measures[0].items.append(m.Time(str(beats), str(beat_type)))
        return measures

    def close_measure(self) -> None:
        self.mark_whole_measure_rests()
        self.part.measures.append(self.measure)
        # MusicXML writes a multi-measure rest as that many measures
        number = self.measure.number
        for _ in range(self.rest_measures):
            number += 1
            filler = m.Measure(number, is_rest_filler=True)
            self.add_measure_rests(filler)
            self.part.measures.append(filler)
        self.rest_measures = 0
        self.measure = m.Measure(number + 1)

    def close_or_reuse_measure(self) -> m.Measure | None:
        """The measure which a barline ends: the current one, or the last if this has no notes."""
        if self.measure.has_notes():
            return self.measure
        return self.part.measures[-1] if self.part.measures else None

    def on_notes(self, group: SymbolChord, _: int) -> None:
        for chord in group.into_positions():
            self.measure.items.extend(self.note_chord(chord))

    def on_multi_rest(self, group: SymbolChord, _: int) -> None:
        if any(isinstance(item, m.MultipleRest) for item in self.measure.items):
            eprint("Measure already has a multi rest")
            return
        duration = int(group.symbols[0].rhythm.split("_")[1].replace("m", ""))
        self.measure.items.append(m.MultipleRest(duration, self.time))
        if duration > 0:
            self.add_measure_rests(self.measure)
            self.rest_measures = duration - 1

    def on_measure_rest(self, group: SymbolChord, _: int) -> None:
        for rest in sorted(group.symbols, key=get_staff):
            self.measure.items.append(self.measure_rest(get_staff(rest)))

    def on_newline(self, _: SymbolChord, index: int) -> None:
        # An octave shift ends with its staff, the next staff repeats it if it continues
        self.stop_octave_shifts(self.measure)
        if index != len(self.groups) - 1:
            self.measure.items.append(m.NewSystem(self.time))

    def on_clef(self, group: SymbolChord, _: int) -> None:
        for clef in group.symbols:
            if clef.rhythm.startswith("clef"):
                sign_and_line = clef.rhythm.split("_")[1]
                octaves = octaves_of_clef(clef.rhythm)
                staff = get_staff(clef)
                clef_item = m.Clef(staff, sign_and_line[0], sign_and_line[1], octaves, self.time)
                self.measure.items.append(clef_item)
                self.clef_octaves[staff] = octaves

    def on_key(self, group: SymbolChord, _: int) -> None:
        fifths = int(group.symbols[0].rhythm.split("_")[1])
        self.measure.items.append(m.Key(fifths, time=self.time))

    def on_time(self, group: SymbolChord, _: int) -> None:
        self.nominator = self.nominators[min(self.time_signature_no, len(self.nominators) - 1)]
        self.time_signature_no += 1
        self.beats, denominator = time_signature_of(group.symbols[0].rhythm, self.nominator)
        self.nominator = Fraction(self.beats, denominator)
        self.measure.items.append(m.Time(str(self.beats), str(denominator), self.time))

    def on_barline(self, group: SymbolChord, _: int) -> None:
        """
        Barlines and repeats. A sign which ends a measure goes to its right barline and
        closes it. If the measure has no notes yet (the sign is at the start of a line, or
        right after another one) it belongs to the measure before. The first measure is
        kept even without notes: it's a pickup in which this part has nothing, and
        dropping it would shift the part.
        """
        rhythm = group.rhythm
        right, left = _REPEATS.get(rhythm, (_bar_style(rhythm), []))
        if right is not None:
            is_empty_pickup = "barline" in rhythm and not self.part.measures
            ending = self.measure if is_empty_pickup else self.close_or_reuse_measure()
            if ending is not None and right:
                _add_signs(ending.barline("right"), right)
            if ending is self.measure:
                self.close_measure()
        elif self.measure.has_notes():
            self.close_measure()  # the repeated section starts with the next measure
        if left:
            _add_signs(self.measure.barline("left"), left)

    def on_volta(self, group: SymbolChord, _: int) -> None:
        if group.rhythm.startswith("voltaStart"):
            if self.measure.number == self.last_volta_measure + 1:
                self.volta_number += 1
            else:
                self.volta_number = 1
            self.measure.barline("left").children.append(m.Ending("start", self.volta_number))
            return
        # A stop comes after the barline which ended the volta's last measure
        stopped = self.close_or_reuse_measure()
        if stopped is not None:
            self.last_volta_measure = stopped.number
            kind = "stop" if group.rhythm.startswith("voltaStop") else "discontinue"
            stopped.barline("right").children.append(m.Ending(kind, self.volta_number))

    def on_marking(self, group: SymbolChord, _: int) -> None:
        for marking in group.symbols:
            if marking.rhythm.startswith("octaveShift"):
                self.octave_shift(marking)
            else:
                self.dynamic(marking)

    def divisions(self, duration: Fraction, minimum: int = 1) -> int:
        return max(minimum, int(duration * self.division))

    def measure_rest(self, staff: int) -> m.Note:
        """A rest which fills the measure, e.g. in a staff which is hidden in a system."""
        return m.Note(
            staff=staff,
            voice=1,
            is_measure_rest=True,
            duration=self.divisions(self.nominator),
        )

    def mark_whole_measure_rests(self) -> None:
        """A whole rest alone in its staff fills the measure in any meter, e.g. 9/8."""
        for staff in {n.staff for n in self.measure.notes()}:
            notes = [n for n in self.measure.notes() if n.staff == staff and not n.is_grace]
            if len(notes) != 1:
                continue
            rest = notes[0]
            is_whole_rest = rest.is_rest and rest.type == "whole" and rest.dots == 0
            if is_whole_rest and rest.time_modification is None:
                rest.is_measure_rest = True
                rest.start = 0
                rest.duration = self.divisions(self.nominator)

    def add_measure_rests(self, measure: m.Measure) -> None:
        for staff in (1, 2) if self.part.has_two_staves else (1,):
            measure.items.append(self.measure_rest(staff))

    def dynamic(self, symbol: EncodedSymbol) -> None:
        marks: list[tuple[str, str | None]] = []
        if symbol.rhythm.startswith("dynamic"):
            name = symbol.rhythm.split("_", 1)[1]
            # dynamic_empty is a modifier-only marking (e.g. "più"), not a MusicXML element
            if name != "empty":
                marks.append((name, None))
        if symbol.articulation.startswith("dynamic"):
            marks.append(
                ("other-dynamics", symbol.articulation.split("_", 1)[1].replace("piu", "più"))
            )
        if marks:
            self.measure.items.append(m.Dynamics(get_staff(symbol), marks, self.time))

    def octave_shift(self, symbol: EncodedSymbol) -> None:
        staff = get_staff(symbol)
        if self.shift_octaves.get(staff, 0) != 0:
            self.measure.items.append(m.OctaveShift(staff, "stop", "above", self.time))
            self.shift_octaves[staff] = 0
        if symbol.rhythm == OCTAVE_SHIFT_STOP:
            return
        octaves = octaves_of_shift(symbol.rhythm)
        if octaves == 0:
            eprint("Unknown octave shift", symbol)
            return
        # MusicXML names the direction in which the notes are printed: 8va is down
        shift_type, placement = ("down", "above") if octaves > 0 else ("up", "below")
        self.measure.items.append(m.OctaveShift(staff, shift_type, placement, self.time))
        self.shift_octaves[staff] = octaves

    def stop_octave_shifts(self, measure: m.Measure) -> None:
        """A shift stops after the last note of its measure."""
        for staff, octaves in self.shift_octaves.items():
            if octaves != 0:
                measure.items.append(m.OctaveShift(staff, "stop", "above", measure.end()))
        self.shift_octaves = {}

    def note_chord(self, chord: SymbolChord) -> list[m.Note]:
        """The notes of one position: a chord per duration, rests after the notes."""
        start = self.divisions(chord.start, minimum=0)
        result: list[m.Note] = []
        for layer, (layer_duration, symbols) in enumerate(_group_notes(chord.symbols).items()):
            notes = [n for n in symbols if n.pitch not in (empty, nonote)]
            rests = [n for n in symbols if n.pitch in (empty, nonote)]
            for i, note in enumerate(notes):
                result.append(self.note(note, layer, i > 0, start))
            # A grace note without pitch would be a rest which takes no time.
            # The dataset has a few groups with several rests, take the first.
            if rests and layer_duration > 0:
                result.append(self.note(rests[0], layer, False, start))
        return result

    def note(self, symbol: EncodedSymbol, layer: int, is_chord: bool, start: int) -> m.Note:
        staff = get_staff(symbol)
        duration = symbol.get_duration()
        voice = get_xml_voice(staff, layer)
        note = m.Note(staff=staff, voice=voice, is_chord=is_chord, start=start)
        note.is_grace = "G" in symbol.rhythm
        note.position = symbol.position
        if symbol.pitch == empty:
            note.is_measure_rest = duration.fraction.numerator == 0
        elif symbol.pitch == nonote:
            eprint("WARNING note without pitch", symbol)
        else:
            octaves = self.clef_octaves.get(staff, 0) + self.shift_octaves.get(staff, 0)
            sounding = transpose_octaves(symbol.pitch, octaves)
            note.pitch = m.Pitch(sounding[0], sounding[1:])
            if symbol.lift == nonote:
                eprint("WARNING note with invalid lift", symbol)
            elif symbol.lift != empty:
                note.pitch.alter = mapping.LIFT_TO_ALTER[symbol.lift]
        if note.is_grace:
            note.type = mapping.DURATION_NAMES[duration.kern]
        elif duration.fraction.numerator > 0:
            note.duration = max(1, int(duration.fraction * self.division))
            note.type = mapping.DURATION_NAMES[1 if duration.kern == 0 else duration.kern]
        else:
            note.duration = self.beats
            note.type = mapping.DURATION_NAMES[0]
        note.dots = duration.dots
        if duration.actual_notes != duration.normal_notes:
            note.time_modification = (duration.actual_notes, duration.normal_notes)
        slurs, beam = split_slur_and_beam(symbol.slur)
        if beam in mapping.BEAMS and note.pitch is not None:
            note.beam_state = mapping.BEAMS[beam]  # completed by passes.convert_beams
        self.add_articulations(note, symbol.articulation)
        self.add_slurs(note, slurs, staff)
        if symbol.image_coordinates is not None:
            x, y = symbol.image_coordinates
            note.image_position = (round(x), round(y))
        note.confidence = symbol.confidence
        return note

    def add_articulations(self, note: m.Note, articulations: str) -> None:
        if not articulations.startswith("dynamic"):
            for articulation in articulations.split("_"):
                if articulation == "":
                    continue
                if articulation == nonote:
                    eprint("WARNING note without valid articulation", articulations)
                    continue
                if articulation not in mapping.ARTICULATIONS:
                    raise ValueError("Unsupported articulation " + articulation)
                container, tag = mapping.ARTICULATIONS[articulation]
                if container == "":
                    note.direct.append(tag)
                elif container == "articulations":
                    note.articulations.append(tag)
                elif articulation == "tremolo":
                    note.has_tremolo = True
                else:
                    note.ornaments.append((tag, {}, None))

    @staticmethod
    def add_slurs(note: m.Note, slurs: str, number: int) -> None:
        if slurs == nonote:
            eprint("WARNING note without valid articulation", slurs)
        elif slurs not in {"_", ""}:
            types = {
                "slurStart": ["start"],
                "slurStop": ["stop"],
                "slurStart_slurStop": ["stop", "start"],
            }
            if slurs not in types:
                raise ValueError("Unsupported slur " + slurs)
            note.slurs.extend(m.Slur(t, number) for t in types[slurs])


# Signs for the right barline of the measure a repeat ends (None: it doesn't end one)
# and for the left barline of the next measure
_REPEATS: dict[str, tuple[list[m.Repeat] | None, list[m.Repeat]]] = {
    "repeatStart": (None, [m.Repeat("forward")]),
    "repeatEnd": ([m.Repeat("backward")], []),
    "repeatEndStart": ([m.Repeat("backward")], [m.Repeat("forward")]),
}


def _bar_style(rhythm: str) -> list[m.BarStyle]:
    """No style for a plain barline, it comes with every measure."""
    style = mapping.BAR_STYLES.get(rhythm, "light-light")
    return [] if rhythm == "barline" else [m.BarStyle(style)]


def _add_signs(barline: m.Barline, signs: list[m.Repeat] | list[m.BarStyle]) -> None:
    for sign in signs:
        if isinstance(sign, m.Repeat) and any(isinstance(c, m.Repeat) for c in barline.children):
            eprint("barline already has a repeat")
            continue
        barline.children.append(m.Repeat(sign.direction) if isinstance(sign, m.Repeat) else sign)


def _group_notes(notes: list[EncodedSymbol]) -> dict[Fraction, list[EncodedSymbol]]:
    groups: dict[Fraction, list[EncodedSymbol]] = defaultdict(list)
    max_duration = max(n.get_duration().fraction for n in notes)
    for note in notes:
        duration = note.get_duration().fraction
        if "G" in note.rhythm:
            duration = Fraction(0)
        elif duration.numerator == 0:
            duration = max_duration
        groups[duration].append(note)
    return dict(sorted(groups.items()))
