"""Where tuplets start and stop, per voice once the voices are final."""

from collections import defaultdict
from collections.abc import Callable
from fractions import Fraction

from homr.music_xml import model as m

# Start and duration in whole notes of a chord of a run of tuplet chords
_Run = list[tuple[list[m.Note], Fraction, Fraction]]


def mark_tuplets(part: m.Part) -> None:
    """
    A tuplet ends with the chord which ends on the beat grid again, e.g. after three
    eighth triplets or a quarter and an eighth triplet. Counting by time rather than by
    notes keeps a voice in phase whose tuplet begins with a rest in another voice.
    """
    whole = 4 * part.divisions
    if whole <= 0:
        return
    for measure in part.measures:
        by_voice: dict[int, list[list[m.Note]]] = defaultdict(list)
        for chord in measure.chords():
            if not chord[0].is_grace:
                by_voice[chord[0].voice].append(chord)
        for chords in by_voice.values():
            run: _Run = []
            run_format: tuple[int, int] | None = None
            for chord in sorted(chords, key=lambda c: c[0].start):
                tuplet = [n for n in chord if n.time_modification is not None]
                chord_format = tuplet[0].time_modification if tuplet else None
                if chord_format != run_format and run:
                    _mark_run(run)
                    run = []
                run_format = chord_format
                if tuplet:
                    start, duration = Fraction(chord[0].start, whole), Fraction(
                        tuplet[0].duration or 0, whole
                    )
                    run.append((tuplet, start, duration))
            if run:
                _mark_run(run)


def _mark_run(run: _Run) -> None:
    """
    Splits consecutive tuplet chords into tuplets. A tuplet ends once its own notes add
    up to a duration without a tuplet, e.g. three eighth triplets or a quarter and an
    eighth triplet. That keeps working if an earlier note has a wrong length. If that
    leaves notes over, e.g. if the tuplet starts with a rest in another voice, a tuplet
    ends where it reaches the beat grid instead. Notes left over form an incomplete tuplet.
    """
    by_length = _split(run, lambda start, total, end: total)
    by_grid = _split(run, lambda start, total, end: end)
    tuplets, left_over = by_length if len(by_length[1]) <= len(by_grid[1]) else by_grid
    if left_over:
        tuplets.append((left_over[0], left_over[-1]))
    for first, last in tuplets:
        for note in run[first][0]:
            note.tuplets.append("start")
        for note in run[last][0]:
            note.tuplets.append("stop")


def _split(
    run: _Run, measure: Callable[[Fraction, Fraction, Fraction], Fraction]
) -> tuple[list[tuple[int, int]], list[int]]:
    """The complete tuplets as (first, last) index into the run and the indices left over."""
    tuplets: list[tuple[int, int]] = []
    pending: list[int] = []
    total = Fraction(0)
    for index, (_, start, duration) in enumerate(run):
        pending.append(index)
        total += duration
        denominator = measure(start, total, start + duration).denominator
        if denominator & (denominator - 1) == 0:  # a power of two: on the beat grid
            tuplets.append((pending[0], index))
            pending = []
            total = Fraction(0)
    return tuplets, pending
