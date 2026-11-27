"""Passes which complete the model after all tokens are laid out, in the order of PASSES."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction

from homr.music_xml import mapping
from homr.music_xml import model as m
from homr.music_xml.accidentals import add_accidentals
from homr.music_xml.builder import get_xml_voice
from homr.music_xml.tuplets import mark_tuplets


def rebalance_voices(part: m.Part) -> None:
    """Stable, non-overlapping voices per staff for each measure."""
    for measure in part.measures:
        if not measure.is_rest_filler:
            _rebalance_measure(measure)


def _rebalance_measure(measure: m.Measure) -> None:
    events: list[list[m.Note]] = []
    for note in measure.notes():
        last = events[-1][0] if events else None
        if note.is_chord and last is not None and _same_event(last, note):
            events[-1].append(note)
        else:
            events.append([note])
    by_staff: dict[int, list[list[m.Note]]] = defaultdict(list)
    for event in events:
        by_staff[event[0].staff].append(event)
    for staff, staff_events in by_staff.items():
        active: list[tuple[int, int]] = []  # end and voice
        for event in sorted(staff_events, key=lambda e: (e[0].start, e[0].end)):
            first = event[0]
            active = [(end, voice) for end, voice in active if end > first.start]
            used = {voice for _, voice in active}
            # The second voice of the model (e.g. upper2) prefers the second voice
            voice = 2 if first.is_second_voice else 1
            if voice in used:
                voice = next(v for v in range(1, len(used) + 2) if v not in used)
            active.append((first.end, voice))
            for note in event:
                note.voice = get_xml_voice(staff, voice - 1)


def _same_event(a: m.Note, b: m.Note) -> bool:
    return (a.staff, a.start, a.end) == (b.staff, b.start, b.end)


def convert_ties(part: m.Part) -> None:
    """
    The model learns slurs and ties as one class, so a tie reaches us as a slur.
    A slur is a tie when it joins a chord to the next chord of its own voice and
    both notes have the same pitch. Ties cross barlines, so this runs per part.
    """
    previous: dict[tuple[int, int], list[m.Note]] = {}
    for measure in part.measures:
        for event in measure.chords():
            key = (event[0].staff, event[0].voice)
            before = previous.get(key)
            previous[key] = event
            if before is not None:
                _tie_event(before, event)


def _slur(note: m.Note, slur_type: str) -> m.Slur | None:
    return next((s for s in note.slurs if s.type == slur_type), None)


def _tie_event(before: list[m.Note], after: list[m.Note]) -> None:
    """
    Each slurred note looks for its own pitch in the next chord: the model marks the
    notehead which the curve touches, not the whole chord.
    """
    is_tied = False
    for start_note in before:
        begin = _slur(start_note, "start")
        if begin is None or start_note.pitch is None:
            continue
        for stop_note in after:
            if stop_note.pitch is None or stop_note.pitch.key() != start_note.pitch.key():
                continue
            end = _slur(stop_note, "stop")
            if end is None:
                continue
            start_note.slurs.remove(begin)
            stop_note.slurs.remove(end)
            start_note.ties.insert(0, "start")
            stop_note.ties.insert(0, "stop")
            is_tied = True
            break
    if is_tied and len(before) > 1 and len(after) > 1:
        # A tied chord is engraved with a tie on every held notehead
        after_by_pitch = {n.pitch.key(): n for n in after if n.pitch is not None}
        for start_note in before:
            if start_note.pitch is None:
                continue
            held = after_by_pitch.get(start_note.pitch.key())
            if held is None or "start" in start_note.ties or "stop" in held.ties:
                continue
            start_note.ties.insert(0, "start")
            held.ties.insert(0, "stop")


def add_stems(part: m.Part) -> None:
    """Where a staff has two voices in a measure, the first has stems up, the second down."""
    for measure in part.measures:
        notes = [n for n in measure.notes() if not n.is_rest]
        for staff in {n.staff for n in notes}:
            on_staff = [n for n in notes if n.staff == staff]
            if len({n.is_second_voice for n in on_staff if n.position}) < 2:
                continue
            for note in on_staff:
                note.stem = "down" if note.is_second_voice else "up"


@dataclass
class _BeamEvent:
    notes: list[m.Note]
    is_grace: bool
    state: str | None
    start: tuple[int, int]  # measure and time within it
    end: tuple[int, int]

    @property
    def position(self) -> str:
        return self.notes[0].position or ""

    @property
    def staff(self) -> int:
        return self.notes[0].staff


def convert_beams(part: m.Part) -> None:
    """
    Turns the primary beam states into complete beams. Groups are formed per
    position of the model rather than per voice; rests don't interrupt a group and it
    can cross a barline. A beam which continues on the other staff joins the open
    group there. Secondary beams and hooks follow from the durations.
    """
    open_groups: dict[tuple[str, bool], list[_BeamEvent]] = {}
    groups: list[list[_BeamEvent]] = []

    def close(key: tuple[str, bool]) -> None:
        group = open_groups.pop(key, [])
        if len(group) >= 2:
            groups.append(group)

    for event in _beam_events(part):
        key = (event.position, event.is_grace)
        if event.state is None or _beam_count(event.notes[0]) == 0:
            close(key)
            continue
        if event.state == "begin":
            close(key)
        elif key not in open_groups:
            other = _cross_staff_group(open_groups, event)
            if other is not None:
                open_groups[other].append(event)
                if event.state == "end":
                    close(other)
                continue
        open_groups.setdefault(key, []).append(event)
        if event.state == "end":
            close(key)
    for key in list(open_groups):
        close(key)
    for group in groups:
        _join_voice(group)
        _write_beams([event.notes[0] for event in group])


def _join_voice(group: list[_BeamEvent]) -> None:
    """
    A beam is drawn within one voice. A beam across staves belongs to the voice of the
    staff with most of its notes, the notes on the other staff are cross-staff notes.
    On a tie it's the lower staff, so that a passage of such beams keeps one voice.
    """
    staves = Counter(event.staff for event in group)
    main = max(staves, key=lambda staff: (staves[staff], staff))
    voice = next(event.notes[0].voice for event in group if event.staff == main)
    for event in group:
        for note in event.notes:
            note.voice = voice


def _beam_events(part: m.Part) -> list[_BeamEvent]:
    events: list[_BeamEvent] = []
    for measure_no, measure in enumerate(part.measures):
        for chord in measure.chords():
            first = chord[0]
            state = next((n.beam_state for n in chord if n.beam_state), None)
            if first.position is None or first.is_rest:
                continue
            events.append(
                _BeamEvent(
                    chord, first.is_grace, state, (measure_no, first.start), (measure_no, first.end)
                )
            )
    return sorted(events, key=lambda e: e.start)  # stable


def _cross_staff_group(
    open_groups: dict[tuple[str, bool], list[_BeamEvent]], event: _BeamEvent
) -> tuple[str, bool] | None:
    """The open group on the other staff which the event can continue, if there is one."""
    # The other staff is the staff of the group's first note: its last one can be on
    # this staff already, joined from here
    candidates = [
        key
        for key, group in open_groups.items()
        if key[1] == event.is_grace
        and group[0].staff != event.staff
        and group[-1].end <= event.start
    ]
    return candidates[0] if len(candidates) == 1 else None


def _beam_count(note: m.Note) -> int:
    return mapping.BEAMS_PER_TYPE.get(note.type or "", 0)


def _write_beams(notes: list[m.Note]) -> None:
    counts = [_beam_count(n) for n in notes]
    for index, note in enumerate(notes):
        for level in range(1, counts[index] + 1):
            has_previous = index > 0 and counts[index - 1] >= level
            has_next = index < len(notes) - 1 and counts[index + 1] >= level
            if has_previous and has_next:
                beam = "continue"
            elif has_previous:
                beam = "end"
            elif has_next:
                beam = "begin"
            else:
                beam = "forward hook" if index == 0 else "backward hook"
            note.beams.append((level, beam))


PASSES = [rebalance_voices, convert_ties, add_accidentals, add_stems, convert_beams, mark_tuplets]


def shorten_measure_rests_in_pickup(parts: list[m.Part]) -> None:
    """
    A part with nothing in its first measure gets a whole-measure rest there. If
    another part has a shorter first measure, it's a pickup and the rest has to be
    as short. A first measure without any notes or rests gets such a rest too.
    """
    if any(not p.measures or p.divisions <= 0 for p in parts):
        return
    played = [
        Fraction(p.measures[0].end(), p.divisions)
        for p in parts
        if any(not n.is_rest for n in p.measures[0].notes())
    ]
    if not played:
        return
    pickup = max(played)
    for part in parts:
        measure = part.measures[0]
        duration = max(1, int(pickup * part.divisions))
        notes = measure.notes()
        if not notes:
            for staff in (1, 2) if part.has_two_staves else (1,):
                voice = 1 if staff == 1 else 5
                measure.items.append(m.Note(staff, voice, is_measure_rest=True, duration=duration))
            continue
        only_measure_rests = all(n.is_measure_rest for n in notes)
        if only_measure_rests and Fraction(measure.end(), part.divisions) > pickup:
            for note in notes:
                note.duration = duration
