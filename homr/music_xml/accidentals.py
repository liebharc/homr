"""Printed accidentals, from the sounding pitches of the model."""

from homr.music_xml import mapping
from homr.music_xml import model as m


def _key_signature_alters(fifths: int) -> dict[str, int]:
    if fifths > 0:
        return dict.fromkeys("FCGDAEB"[:fifths], 1)
    if fifths < 0:
        return dict.fromkeys("BEADGCF"[:-fifths], -1)
    return {}


def add_accidentals(part: m.Part) -> None:
    """Adds printed signs to the notes, without changing sounding pitches."""
    key_alters: dict[str, int] = {}
    for measure in part.measures:
        for key in (item for item in measure.items if isinstance(item, m.Key)):
            key_alters = _key_signature_alters(key.fifths)
        _add_measure_accidentals(measure, key_alters)


def _add_measure_accidentals(measure: m.Measure, key_alters: dict[str, int]) -> None:
    """A sign holds until the end of the measure for its staff, step and octave."""
    active: dict[tuple[int, str, str], int] = {}
    printed: set[tuple[int, tuple[int, str, str], int]] = set()
    # Grace notes share the onset of the note they precede, but sound first
    for note in sorted(measure.notes(), key=lambda n: (n.start, not n.is_grace)):
        if note.pitch is None or "stop" in note.ties:
            continue
        step = note.pitch.step
        pitch_key = (note.staff, step, note.pitch.octave)
        alter = int(note.pitch.alter or 0)
        if alter != active.get(pitch_key, key_alters.get(step, 0)):
            # Two voices on the same note at the same time share one sign
            unison = (note.start, pitch_key, alter)
            if note.is_grace or unison not in printed:
                note.accidental = mapping.ALTER_TO_ACCIDENTAL[alter]
                if not note.is_grace:
                    printed.add(unison)
        active[pitch_key] = alter
