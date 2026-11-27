"""Convert MusicXML scores into musical symbols for training and evaluation.

music_xml_string_to_tokens() and music_xml_file_to_tokens() return a list of
parts, each containing a list of Measure objects. A part usually represents
one instrument or singing role and may contain multiple staves. Each Measure
contains an ordered list of EncodedSymbol objects and a page-break flag.

homr/music_xml/reader.py reads the MusicXML into the model which the generator
writes. This module turns the model into tokens: PartTokenizer goes through a part
and carries state between measures, such as clef changes queued for the next
measure. TokensMeasure orders the symbols of one measure across staves and supplies
its ending barline when needed. Even a measure with no emitted symbols retains its
ending barline.

Repeat signs stay in their source measures during parsing. Callers use
normalize_barlines_and_repeats() after selecting the measures for one training
image or a complete part for comparison. It combines signs at shared measure
boundaries, then returns a flat list of symbols.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterator
from typing import Iterable, Sequence

from homr.music_xml import mapping
from homr.music_xml import model as m
from homr.music_xml.mapping import DURATION_NUMBER
from homr.music_xml.reader import read_parts
from homr.simple_logging import eprint
from homr.transformer.vocabulary import (
    DYNAMICS,
    DYNAMICS_MODIFIER,
    MULTIREST,
    OCTAVE_CLEFS,
    OCTAVE_SHIFT_STOP,
    VALID_TIME_SIGNATURE_DENOMINATORS,
    EncodedSymbol,
    empty,
    has_rhythm_symbol_a_position,
    is_lower_position,
    octaves_of_clef,
    octaves_of_shift,
    transpose_octaves,
)
from training.omr_datasets.staff_merging import (
    EncodedSymbolWithPos,
    merge_upper_and_lower_staff,
)
from training.transformer.training_vocabulary import VocabularyStats, check_token_lines


class Measure(list[EncodedSymbol]):
    """The symbols of a measure, and whether it starts a new page."""

    def __init__(self, symbols: Iterable[EncodedSymbol] = (), new_page: bool = False) -> None:
        super().__init__(symbols)
        self.new_page = new_page


class TokensMeasure:
    """
    MusicXML allows directions such as forward/backwards. For training we need
    to have a well defined sequence. So we linearize the MusicXML instrunctions
    using this class.
    """

    def __init__(self, octaves: dict[tuple[int, str], int] | None = None) -> None:
        self.symbols: list[EncodedSymbolWithPos] = []
        self.main_voice: dict[int, int] = {}  # map staff id to its main voice id
        self.graces_at: dict[tuple[int, int], int] = {}  # (staff, position) -> grace notes so far
        # Time in divisions, from the shared timeline, and the symbols before it.
        # Symbols at one time keep their order and come before the notes at it.
        self.time = 0
        self.symbols_at: dict[int, int] = defaultdict(int)
        self.barline_times: set[int] = set()
        self.last_grace_index = 0
        self.new_page = False
        # Octaves between written and sounding pitch per (staff, "clef" or "shift")
        # at the start of the measure, and the changes within the measure.
        self.start_octaves: dict[tuple[int, str], int] = dict(octaves or {})
        self.octave_events: list[tuple[int, tuple[int, str], int]] = []

    def append_symbol(self, symbol: EncodedSymbol) -> None:
        if symbol.rhythm.startswith("note"):
            raise ValueError("Call append_note for notes")
        else:
            self.symbols_at[self.time] += 1
            # A barline comes after the notes which start at its time, and so do the
            # symbols after it, e.g. the end of a volta
            if "barline" in symbol.rhythm:
                self.barline_times.add(self.time)
            if self.time in self.barline_times:
                position = self._note_position() + 1
            else:
                position = self._symbol_position()
            self.symbols.append(EncodedSymbolWithPos(position, symbol, True))

    _SLOTS = 10000  # positions per division

    def set_time(self, time: int) -> None:
        self.time = time

    def _symbol_position(self) -> int:
        return self.time * self._SLOTS + self.symbols_at[self.time]

    def _note_position(self) -> int:
        return self.time * self._SLOTS + self._SLOTS - 2

    def mark_new_page(self) -> None:
        self.new_page = True

    def append_symbol_to_staff(self, staff: int, symbol: EncodedSymbol) -> None:
        if has_rhythm_symbol_a_position(symbol.rhythm):
            symbol.position = self._get_staff_position(staff)
        if symbol.rhythm.startswith("note"):
            raise ValueError("Call append_note for notes")
        staff_no = 1 if staff > 0 else 0  # the same as _get_staff_no
        if symbol.rhythm.startswith("clef"):
            self.octave_events.append(
                (self._symbol_position(), (staff_no, "clef"), octaves_of_clef(symbol.rhythm))
            )
        elif symbol.rhythm.startswith("octaveShift"):
            self.octave_events.append(
                (self._symbol_position(), (staff_no, "shift"), octaves_of_shift(symbol.rhythm))
            )
        self.symbols.append(EncodedSymbolWithPos(self._symbol_position(), symbol, True))

    def _octaves_at(self, position: float) -> dict[tuple[int, str], int]:
        """
        Octave clefs and shifts in effect at a position. A change at the position
        of a note applies to it: a shift starts before its first note and stops
        after its last one.
        """
        octaves = dict(self.start_octaves)
        for event_position, key, value in sorted(self.octave_events, key=lambda e: e[0]):
            if event_position <= position:
                octaves[key] = value
        return octaves

    def end_octaves(self) -> dict[tuple[int, str], int]:
        return self._octaves_at(float("inf"))

    def _to_written_pitches(self) -> None:
        """MusicXML has sounding pitches, the tokens have written pitches."""
        if not self.octave_events and not any(self.start_octaves.values()):
            return
        for entry in self.symbols:
            symbol = entry.symbol
            if not symbol.rhythm.startswith("note"):
                continue
            staff = self._get_staff_no(entry)
            octaves = self._octaves_at(entry.position)
            change = octaves.get((staff, "clef"), 0) + octaves.get((staff, "shift"), 0)
            symbol.pitch = transpose_octaves(symbol.pitch, -change)

    def append_note(
        self,
        staff: int,
        voice: int,
        is_chord: bool,
        invisible: bool,
        symbol: EncodedSymbol,
    ) -> None:
        is_grace = "G" in symbol.rhythm
        symbol.position = self._get_staff_position(staff)
        if not invisible:
            if voice != self.main_voice[staff]:
                symbol.position += "2"
        if is_chord:
            grace_index = self.last_grace_index
        else:
            grace_index = 0
            if is_grace:
                key = (staff, self.time)
                grace_index = self.graces_at.get(key, 0)
                self.graces_at[key] = grace_index + 1
            self.last_grace_index = grace_index
        # The chord's first note can be invisible, so it isn't always the last symbol
        if not invisible:
            self.symbols.append(
                EncodedSymbolWithPos(
                    # Grace notes share their group with the symbols at their time
                    self._symbol_position() if is_grace else self._note_position(),
                    symbol,
                    insert_before=is_grace,
                    grace_index=grace_index,
                )
            )

    def _get_staff_no(self, symbol: EncodedSymbolWithPos) -> int:
        if is_lower_position(symbol.symbol.position):
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
        arpegiatted_positions: set[tuple[tuple[int, int], str]] = set()
        for entry in symbols_with_pos:
            # `pos` is the place in the measure (grace notes come before the chord at the same
            # position), `sym.position` is the upper/lower staff.
            pos, sym = entry.sort_order(), entry.symbol
            if "arpeggiate" in sym.articulation:
                arpegiatted_positions.add((pos, sym.position))

        for entry in symbols_with_pos:
            pos, sym = entry.sort_order(), entry.symbol
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

    def complete_measure(self) -> Measure:
        """Build a Measure from the collected symbols.

        Order the symbols, combine the staff sequences, and add an ordinary
        ending barline when needed. Preserve the page-break flag.
        Empty source measures produce a single barline.
        """
        self._fill_in_arpeggiate(self.symbols)
        self._to_written_pitches()
        # merge_upper_and_lower_staff orders them: by position, the upper staff first
        staves: list[list[EncodedSymbolWithPos]] = [[], []]
        for symbol in self.symbols:
            staves[self._get_staff_no(symbol)].append(symbol)

        # Staff merging supplies ending barlines for nonempty symbol lists,
        # but treats any repeat as an ending. An empty measure still needs an
        # ending barline when it has no symbols or only an opening repeat.
        merged_symbols = merge_upper_and_lower_staff(staves)
        if not merged_symbols or merged_symbols[-1].rhythm == "repeatStart":
            merged_symbols.append(EncodedSymbol("barline"))
        return Measure(merged_symbols, self.new_page)


class TupletState:
    """
    Counts the open tuplets, as tuplets of different voices can overlap. Counting
    rather than tracking voices keeps it independent of how voices are numbered.
    """

    def __init__(self) -> None:
        self.open_tuplets = 0
        self.last_stop_time = -1

    def factor(self, note: m.Note) -> float:
        # A tuplet counts whether or not its number is printed: many editions only print the
        # "3" on the first groups. Notes with a <time-modification> outside of a <tuplet>
        # (e.g. a two-note tremolo) keep their written length.
        in_tuplet = self.open_tuplets > 0 or self.last_stop_time == note.start
        for tuplet in note.tuplets:
            if tuplet == "start":
                self.open_tuplets += 1
                in_tuplet = True
            if tuplet == "stop":
                self.open_tuplets = max(self.open_tuplets - 1, 0)
                self.last_stop_time = note.start
        if not in_tuplet or note.time_modification is None:
            return 1.0
        actual, normal = note.time_modification
        return float(actual) / float(normal)


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


class PartTokenizer:
    """The tokens of a part, measure by measure."""

    def __init__(self, sounding_pitch: bool = False) -> None:
        # Training uses the written pitches together with octave clefs and shifts.
        # The benchmarks compare sounding pitches and ignore octave clefs and shifts.
        self.sounding_pitch = sounding_pitch
        self.current = TokensMeasure()
        self.measures: list[Measure] = []
        self.tuplets = TupletState()
        self.divisions = 1
        # Clefs queued by an after-barline declaration in the previous measure,
        # not yet eligible to be flushed into the current one.
        self.queued_clefs: dict[int, EncodedSymbol] = {}
        # Queued clefs that became active once the measure they target started;
        # still cancellable by an explicit clef in that same measure.
        self.pending_clefs: dict[int, EncodedSymbol] = {}

    @property
    def measure(self) -> TokensMeasure:
        """The measure to add symbols to, with the clefs which were queued for it."""
        for staff, symbol in self.pending_clefs.items():
            self.current.append_symbol_to_staff(staff, symbol)
        self.pending_clefs = {}
        return self.current

    def tokenize(self, part: m.Part) -> list[Measure]:
        measures_to_skip = 0
        for measure in part.measures:
            if measures_to_skip > 0:
                # MusicXML writes a multi-measure rest as that many measures, MuseScore
                # renders one. The last hidden one can carry the rendered right barline.
                if measures_to_skip == 1:
                    for barline in (i for i in measure.items if isinstance(i, m.Barline)):
                        self.barline(barline)
                measures_to_skip -= 1
                continue
            measures_to_skip = max(self.read_measure(measure) - 1, 0)
        return self.measures

    def read_measure(self, measure: m.Measure) -> int:
        """Returns the length of a multi-measure rest which starts here, 0 if there's none."""
        if measure.divisions:
            self.divisions = measure.divisions
        self.current.main_voice = _find_main_voices(measure.notes())
        multiple_rest = 0
        for items in _batches(measure.items):
            item = items[0]
            self.current.set_time(item.start if isinstance(item, m.Note) else item.time)
            if isinstance(item, m.Note):
                # The first collapsed measure has a placeholder whole rest in MusicXML,
                # the multi-rest token stands for it
                if multiple_rest <= 1:
                    self.note(item)
            elif isinstance(item, m.Clef):
                self.clefs(items)  # type: ignore[arg-type]
            elif isinstance(item, m.Key):
                self.measure.append_symbol(EncodedSymbol(f"keySignature_{item.fifths}"))
            elif isinstance(item, m.Time):
                beat_type = item.beat_type
                if (
                    not beat_type.isdigit()
                    or int(beat_type) not in VALID_TIME_SIGNATURE_DENOMINATORS
                ):
                    raise ValueError(f"Unsupported time signature denominator: {beat_type}")
                self.measure.append_symbol(EncodedSymbol(f"timeSignature/{beat_type}"))
            elif isinstance(item, m.MultipleRest):
                multiple_rest = item.measures
                # All multirests share one token, the digit reader reads the number
                if item.measures > 1:
                    self.measure.append_symbol(
                        EncodedSymbol(MULTIREST, empty, empty, empty, empty, "upper")
                    )
            elif isinstance(item, m.Barline):
                self.barline(item)
            elif isinstance(item, m.Dynamics):
                self.dynamics(item)
            elif isinstance(item, m.OctaveShift):
                self.octave_shift(item)
            elif isinstance(item, m.NewPage):
                self.measure.mark_new_page()
        self.measures.append(self.current.complete_measure())
        self.current = TokensMeasure(self.current.end_octaves())
        self.pending_clefs.update(self.queued_clefs)
        self.queued_clefs = {}
        self.tuplets = TupletState()
        return multiple_rest

    def note(self, note: m.Note) -> None:
        if note.is_unpitched:
            return
        if note.duration is None and not note.is_grace:
            eprint("Note without duration")
        duration = note.duration or 0
        factor = self.tuplets.factor(note)
        note_type = note.type or "eighth"
        exact = DURATION_NUMBER[note_type] * factor
        value = round(exact)
        if abs(exact - value) > 1e-6 and not self.sounding_pitch:
            # E.g. a duplet in 6/8: rounded it reads as a different tuplet. Training skips it,
            # the benchmarks (sounding pitch) keep the rounded value.
            raise ValueError(f"Tuplet can't be encoded: {note_type} * {factor:.3f}")
        articulation, slur = self.articulation(note)
        grace = "G" if note.is_grace else ""
        rhythm = f"{value}{grace}{'.' * min(note.dots, 2)}"
        if note.pitch is None and note.is_measure_rest:
            rhythm = _measure_rest_rhythm(duration, self.divisions)
            symbol = EncodedSymbol(rhythm, empty, empty, empty, empty)
        elif note.pitch is None:
            symbol = EncodedSymbol(f"rest_{rhythm}", empty, empty, articulation, slur)
        else:
            pitch = f"{note.pitch.step}{note.pitch.octave}"
            symbol = EncodedSymbol(
                f"note_{rhythm}", pitch, _lift(note), articulation, _add_beam(slur, note)
            )
        self.measure.append_note(note.staff - 1, note.voice, note.is_chord, note.is_hidden, symbol)

    def articulation(self, note: m.Note) -> tuple[str, str]:
        if not note.has_notations:
            return empty, empty
        slurs = ["slur" + t.capitalize() for t in note.tied]
        slurs += ["slur" + s.type.capitalize() for s in note.slurs]
        articulations = list(note.direct)
        for tag in [*note.articulations, *(o[0] for o in note.ornaments)]:
            if token := mapping.read_articulation(tag):
                articulations.append(token)
        return _join(set(articulations)), _join(slurs)

    def clefs(self, clefs: list[m.Clef]) -> None:
        """The clefs of one <attributes>."""
        now = [c for c in clefs if not c.after_barline]
        for clef in clefs:
            if clef.after_barline:
                # Printed at the end of this measure, takes effect in the next one
                self.queued_clefs[clef.staff - 1] = self.clef_symbol(clef)
        # An explicit clef for a staff in this measure overrides any clef
        # that was queued for it via after-barline in the previous measure.
        for clef in now:
            self.pending_clefs.pop(clef.staff - 1, None)
        if now:
            measure = self.measure
            for clef in now:
                measure.append_symbol_to_staff(clef.staff - 1, self.clef_symbol(clef))

    def clef_symbol(self, clef: m.Clef) -> EncodedSymbol:
        name = f"clef_{clef.sign}{clef.line}"
        if clef.octave_change != 0 and not self.sounding_pitch:
            name += {1: "_8va", -1: "_8vb"}.get(clef.octave_change, f"_{clef.octave_change}")
            if name not in OCTAVE_CLEFS:
                raise ValueError(f"Unsupported octave clef {name}")
        return EncodedSymbol(name, empty, empty, empty, empty)

    def barline(self, barline: m.Barline) -> None:
        children = {type(c): c for c in reversed(barline.children)}  # the first of each kind
        repeat, style, ending = (children.get(k) for k in (m.Repeat, m.BarStyle, m.Ending))
        token = mapping.REPEATS_READ.get(repeat.direction) if isinstance(repeat, m.Repeat) else None
        if token is None:
            # A barline without style or repeat is added with every measure
            token = mapping.read_bar_style(style.style if isinstance(style, m.BarStyle) else "")
        if token is not None:
            self.measure.append_symbol(EncodedSymbol(token))
        if isinstance(ending, m.Ending) and ending.type in mapping.ENDINGS_READ:
            self.measure.append_symbol(EncodedSymbol(mapping.ENDINGS_READ[ending.type]))

    def dynamics(self, dynamics: m.Dynamics) -> None:
        texts = [(text or "").strip() for tag, text in dynamics.marks if tag == "other-dynamics"]
        modifier = "_".join(t.replace(" ", "") for t in texts).replace(".", "").replace("ù", "u")
        modifier_symbol = f"dynamic_{modifier}" if modifier in DYNAMICS_MODIFIER else empty
        has_normal_tag = any(tag in DYNAMICS for tag, _ in dynamics.marks)
        for tag, _ in dynamics.marks:
            if tag in DYNAMICS:
                name = f"dynamic_{tag}"
            elif not has_normal_tag and modifier in DYNAMICS_MODIFIER:
                name = "dynamic_empty"
            else:
                continue
            measure = self.measure
            symbol = EncodedSymbol(name, empty, empty, modifier_symbol, empty)
            symbol.position = measure._get_staff_position(dynamics.staff - 1)
            measure.append_symbol_to_staff(dynamics.staff - 1, symbol)

    def octave_shift(self, shift: m.OctaveShift) -> None:
        if self.sounding_pitch:
            return
        if shift.type == "stop":
            name = OCTAVE_SHIFT_STOP
        elif shift.type in ("down", "up"):
            if shift.size != 8:
                raise ValueError("Only octave shifts by one octave are supported")
            # MusicXML names the direction in which the notes are printed: down is 8va
            name = "octaveShift_8va" if shift.type == "down" else "octaveShift_8vb"
        else:
            return
        symbol = EncodedSymbol(name, empty, empty, empty, empty)
        self.measure.append_symbol_to_staff(shift.staff - 1, symbol)


def _batches(items: list[m.Item]) -> Iterator[list[m.Item]]:
    """
    The items one by one, except the clefs, keys or time signatures of one <attributes>:
    they come as one list. Of several keys or time signatures only the first counts.
    """
    index = 0
    while index < len(items):
        item = items[index]
        end = index + 1
        if isinstance(item, m.Clef | m.Key | m.Time):
            while end < len(items) and type(items[end]) is type(item):
                if items[end].time != item.time:  # type: ignore[union-attr]
                    break
                end += 1
        yield items[index:end]
        index = end


def _lift(note: m.Note) -> str:
    # An explicit courtesy accidental wins over the alter
    if note.accidental is not None:
        return mapping.ACCIDENTAL_TO_LIFT.get(note.accidental.strip(), empty)
    if note.pitch is None or note.pitch.alter is None:
        return empty
    return mapping.alter_to_lift(int(note.pitch.alter))


def _join(tokens: Iterable[str]) -> str:
    return str.join("_", sorted(tokens)) or empty


def _add_beam(slur: str, note: m.Note) -> str:
    """Adds the state of the primary beam, MuseScore writes it on the first note of a chord."""
    for number, text in note.beams:
        state = mapping.BEAMS_READ.get(text)
        if number == 1 and state is not None:
            slurs = [] if slur == empty else slur.split("_")
            return _join([*slurs, state])
    return slur


def _find_main_voices(notes: list[m.Note]) -> dict[int, int]:
    """
    Find the main voice of each staff in a measure.
    Main voice gets the position upper/lower, all others get upper2/lower2.

    How to find the main voice?
    The main voice is drawn with stems up and the others with stems down.
    Usually that's the first visible voice, but engravers sometimes enter
    the upper part in the second voice and flip the stems.
    So if the first voice has mostly down stems and exactly one other
    voice has mostly up stems, then that voice is the main voice.
    We use stem_score to count the number of up stems and down stems.
    """
    visible = [note for note in notes if not note.is_hidden]
    # A voice belongs to the staff with most of its notes. Notes of a cross-staff beam
    # on the other staff don't make it the main voice there.
    notes_per_staff: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for note in visible:
        notes_per_staff[note.voice][note.staff - 1] += 1
    home_staff = {
        voice: max(counts, key=lambda staff: counts[staff])
        for voice, counts in notes_per_staff.items()
    }
    first_voice: dict[int, int] = {}  # staff -> voice
    stem_score: dict[tuple[int, int], int] = defaultdict(int)  # (staff, voice) -> score
    for note in visible:
        staff = note.staff - 1
        if home_staff[note.voice] == staff:
            first_voice.setdefault(staff, note.voice)
        if note.stem == "up":
            stem_score[(staff, note.voice)] += 1
        elif note.stem == "down":
            stem_score[(staff, note.voice)] -= 1
    for note in visible:
        # A staff with only notes of voices from the other staff
        first_voice.setdefault(note.staff - 1, note.voice)
    result: dict[int, int] = {}
    for staff, voice in first_voice.items():
        if stem_score[(staff, voice)] < 0:
            # first voice isn't the main voice, look for another one
            alternatives = [
                v
                for (s, v), score in stem_score.items()
                if s == staff and score > 0 and home_staff[v] == staff
            ]
            if len(alternatives) == 1:
                result[staff] = alternatives[0]
                continue
        result[staff] = voice
    return result


def normalize_barlines_and_repeats(
    measures: Sequence[Sequence[EncodedSymbol]],
) -> list[EncodedSymbol]:
    """Combine shared measure boundaries and return a flat transcription.

    Pass consecutive measures from one part, selected for one image or a whole
    part comparison. Source measures already include their ending barlines.
    An ending barline and the next measure's opening repeat describe the same
    boundary: keep the repeat, combining repeatEnd + repeatStart as repeatEndStart.

    Keep symbols within each measure separate, even if no visible notes lie
    between them. Do not modify the source measures or merge across images.
    """
    result: list[EncodedSymbol] = []
    for index, measure in enumerate(measures):
        if index > 0 and measures[index - 1] and measure and measure[0].rhythm == "repeatStart":
            ending = measures[index - 1][-1].rhythm
            if ending in ("repeatEnd", "repeatEndStart"):
                result[-1] = EncodedSymbol("repeatEndStart")
                result.extend(measure[1:])
                continue
            if "barline" in ending:
                result.pop()
        result.extend(measure)
    return result


def _to_tokens(root: ET.Element, sounding_pitch: bool) -> list[list[Measure]]:
    return [PartTokenizer(sounding_pitch).tokenize(part) for part in read_parts(root)]


def music_xml_string_to_tokens(content: str, sounding_pitch: bool = False) -> list[list[Measure]]:
    """
    Returns a list of voices.
    Each voice is a list of measures.
    """
    return _to_tokens(ET.fromstring(content), sounding_pitch)  # noqa: S314


def music_xml_file_to_tokens(file_path: str, sounding_pitch: bool = False) -> list[list[Measure]]:
    with open(file_path, "rb") as f:
        return _to_tokens(ET.parse(f).getroot(), sounding_pitch)  # noqa: S314


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
