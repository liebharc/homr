"""
Alternative **kern -> EncodedSymbol parser built on music21.

Drop-in replacement for convert_kern_to_parts().  The same post-processing
normalisation steps are applied so token sequences are directly comparable.
"""

from __future__ import annotations

import music21 as m21
import music21.bar
import music21.chord
import music21.clef
import music21.key
import music21.meter
import music21.note
import music21.spanner
import music21.stream

from homr.circle_of_fifths import strip_naturals
from homr.transformer.vocabulary import EncodedSymbol, empty

_DTYPE_TO_KERN: dict[str, int] = {
    "breve": 0,
    "whole": 1,
    "half": 2,
    "quarter": 4,
    "eighth": 8,
    "16th": 16,
    "32nd": 32,
    "64th": 64,
    "128th": 128,
}


def _dur_to_kern(dur: object, ignore_tuplets: bool = False) -> tuple[int, int]:
    """Return (kern_int, dot_count) for a music21 Duration.

    ignore_tuplets: when True, use the visual note type from dur.type without
    applying any <time-modification> tuplet factor. Set this for MusicXML input
    where <type> is the notated value and tuplet adjustments reflect playback
    timing, not the written symbol (e.g. tremolo time-modifications).
    """
    base = _DTYPE_TO_KERN.get(getattr(dur, "type", "quarter"), 4)
    dots = min(int(getattr(dur, "dots", 0)), 2)
    if not ignore_tuplets:
        tuplets = getattr(dur, "tuplets", ())
        if tuplets:
            t = tuplets[0]
            actual = int(getattr(t, "numberNotesActual", 1))
            normal = int(getattr(t, "numberNotesNormal", 1))
            if normal > 0:
                base = round(base * actual / normal)
    return base, dots


def _rhythm_token(prefix: str, kern_val: int, dots: int, is_grace: bool) -> str:
    return f"{prefix}_{kern_val}{'G' if is_grace else ''}{'.' * dots}"


_ACC_MAP: dict[str, str] = {
    "sharp": "#",
    "double-sharp": "##",
    "flat": "b",
    "double-flat": "bb",
    "natural": "N",
}


def _get_lift(acc: object) -> str:
    if acc is None:
        return empty
    return _ACC_MAP.get(str(getattr(acc, "name", "")), empty)


_ARTIC_MAP: dict[str, str] = {
    "Staccato": "staccato",
    "Staccatissimo": "staccatissimo",
    "Accent": "accent",
    "StrongAccent": "accent",
    "Tenuto": "tenuto",
    "Fermata": "fermata",
    "Arpeggiation": "arpeggiate",
    "BreathMark": "breathMark",
    "Spiccato": "spiccato",
}

_EXPR_MAP: dict[str, str] = {
    "Trill": "trill",
    "Mordent": "mordent",
    "InvertedMordent": "trill",
    "Turn": "turn",
    "InvertedTurn": "turn",
}


def _get_articulations(note: object) -> tuple[str, str]:
    arts: list[str] = []
    for a in getattr(note, "articulations", []):
        mapped = _ARTIC_MAP.get(type(a).__name__, "")
        if mapped:
            arts.append(mapped)
    for e in getattr(note, "expressions", []):
        for parent in type(e).__mro__:
            mapped = _EXPR_MAP.get(parent.__name__, "")
            if mapped:
                arts.append(mapped)
                break

    slur_parts: list[str] = []
    tie = getattr(note, "tie", None)
    if tie is not None:
        t_type = getattr(tie, "type", "")
        if t_type in ("start", "continue"):
            slur_parts.append("slurStart")
        if t_type in ("stop", "continue"):
            slur_parts.append("slurStop")

    seen_slur_keys: set[tuple[int, int]] = set()
    for sp in getattr(note, "getSpannerSites", list)():
        if isinstance(sp, m21.spanner.Slur):
            first = sp.getFirst()
            last = sp.getLast()
            key = (id(first), id(last))
            if key in seen_slur_keys:
                continue
            seen_slur_keys.add(key)
            if sp.isFirst(note):
                slur_parts.append("slurStart")
            if sp.isLast(note):
                slur_parts.append("slurStop")

    arts = sorted(set(arts))
    slur_parts = sorted(slur_parts)
    art_str = "_".join(arts) if arts else empty
    slur_str = "_".join(slur_parts) if slur_parts else empty
    return art_str, slur_str


_DEFAULT_CLEF_LINE: dict[str, int] = {"G": 2, "F": 4, "C": 3}


def _clef_to_symbol(clef_el: object) -> EncodedSymbol:
    sign = str(getattr(clef_el, "sign", "G") or "G")
    line = getattr(clef_el, "line", None)
    if line is None:
        line = _DEFAULT_CLEF_LINE.get(sign, 2)
    return EncodedSymbol(f"clef_{sign}{int(line)}", empty, empty, empty, empty)


def _barline_tokens(
    right: object,
    next_left: object,
) -> list[EncodedSymbol]:
    """Produce barline token(s) for the boundary between two consecutive measures."""
    is_r_end = isinstance(right, m21.bar.Repeat) and getattr(right, "direction", "") == "end"
    is_l_start = (
        isinstance(next_left, m21.bar.Repeat) and getattr(next_left, "direction", "") == "start"
    )

    if is_r_end and is_l_start:
        return [EncodedSymbol("repeatEndStart")]
    if is_r_end:
        return [EncodedSymbol("repeatEnd")]
    if is_l_start:
        return [EncodedSymbol("repeatStart")]

    btype = str(getattr(right, "type", "") or "")
    if btype == "double":
        return [EncodedSymbol("doublebarline")]
    if btype == "final":
        return [EncodedSymbol("bolddoublebarline")]
    return [EncodedSymbol("barline")]


def _note_to_symbol(
    note: object, dur: object = None, ignore_tuplets: bool = False
) -> EncodedSymbol:
    d = dur if dur is not None else getattr(note, "duration", None)
    kern_val, dots = _dur_to_kern(d, ignore_tuplets)
    is_grace = bool(getattr(d, "isGrace", False))
    rhythm = _rhythm_token("note", kern_val, dots, is_grace)
    pitch = getattr(note, "pitch", None)
    step = str(getattr(pitch, "step", "C"))
    octave = int(getattr(pitch, "octave", 4) or 4)
    pitch_str = f"{step}{octave}"
    lift = _get_lift(getattr(pitch, "accidental", None))
    art, slur = _get_articulations(note)
    return EncodedSymbol(rhythm, pitch_str, lift, art, slur)


def _rest_to_symbol(rest: object, ignore_tuplets: bool = False) -> EncodedSymbol:
    d = getattr(rest, "duration", None)
    kern_val, dots = _dur_to_kern(d, ignore_tuplets)
    is_grace = bool(getattr(d, "isGrace", False))
    rhythm = _rhythm_token("rest", kern_val, dots, is_grace)
    return EncodedSymbol(rhythm, empty, empty, empty, empty)


def _expand_element(el: object, ignore_tuplets: bool = False) -> list[EncodedSymbol]:
    """Expand a Note, Rest, or Chord into EncodedSymbol instances."""
    if isinstance(el, m21.note.Note):
        return [_note_to_symbol(el, ignore_tuplets=ignore_tuplets)]
    if isinstance(el, m21.note.Rest):
        return [_rest_to_symbol(el, ignore_tuplets=ignore_tuplets)]
    if isinstance(el, m21.chord.Chord):
        dur = getattr(el, "duration", None)
        chord_art, chord_slur = _get_articulations(el)
        result: list[EncodedSymbol] = []
        for i, n in enumerate(getattr(el, "notes", [])):
            sym = _note_to_symbol(n, dur, ignore_tuplets=ignore_tuplets)
            if i == 0:
                merged_art = chord_art if sym.articulation == empty else sym.articulation
                merged_slur = chord_slur if sym.slur == empty else sym.slur
                sym = EncodedSymbol(sym.rhythm, sym.pitch, sym.lift, merged_art, merged_slur)
            result.append(sym)
        return result
    return []


def _remove_redundant_key_changes(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    last = EncodedSymbol("")
    result = []
    for sym in symbols:
        if sym.rhythm.startswith("keySignature") and sym.rhythm == last.rhythm:
            continue
        result.append(sym)
        last = sym
    return result


def _fix_final_repeat_start(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    if not symbols:
        return symbols
    if symbols[-1].rhythm == "repeatEndStart":
        symbols[-1].rhythm = "repeatEnd"
    if symbols[-1].rhythm == "repeatStart":
        symbols[-1].rhythm = "barline"
    return symbols


def _convert_m21_staff_group(
    staves: list[m21.stream.PartStaff], ignore_tuplets: bool = False
) -> list[EncodedSymbol]:
    """Convert multiple PartStaff objects (e.g., piano grand staff) into one token list.

    Notes at the same beat offset across staves are combined into chords. All clefs are
    emitted; key and time signatures are deduplicated (one emission per measure change).
    ignore_tuplets: pass True for MusicXML input to use visual note types.
    """
    tokens: list[EncodedSymbol] = []
    all_measures = [
        list(getattr(staff, "getElementsByClass", lambda _: [])("Measure")) for staff in staves
    ]
    n_measures = max((len(m) for m in all_measures), default=0)
    first_staff_measures = all_measures[0] if all_measures else []

    for m_idx in range(n_measures):
        clefs: list[object] = []
        key_sig: object | None = None
        time_sig: object | None = None
        for staff_measures in all_measures:
            if m_idx >= len(staff_measures):
                continue
            for el in staff_measures[m_idx]:
                if isinstance(el, m21.clef.Clef):
                    clefs.append(el)
                elif isinstance(el, m21.key.KeySignature) and key_sig is None:
                    key_sig = el
                elif isinstance(el, m21.meter.TimeSignature) and time_sig is None:
                    time_sig = el

        for clef in clefs:
            tokens.append(_clef_to_symbol(clef))
        if key_sig is not None:
            tokens.append(EncodedSymbol(f"keySignature_{getattr(key_sig, 'sharps', 0)}"))
        if time_sig is not None:
            tokens.append(EncodedSymbol(f"timeSignature/{getattr(time_sig, 'denominator', 4)}"))

        by_offset: dict[float, list[object]] = {}
        for staff_measures in all_measures:
            if m_idx >= len(staff_measures):
                continue
            measure = staff_measures[m_idx]
            voices = list(getattr(measure, "voices", []))
            sources: list[object] = voices if voices else [measure]
            for source in sources:
                for el in getattr(source, "notesAndRests", []):
                    off = float(getattr(el, "offset", 0.0))
                    by_offset.setdefault(off, []).append(el)

        for off in sorted(by_offset):
            symbols_at: list[EncodedSymbol] = []
            for el in by_offset[off]:
                symbols_at.extend(_expand_element(el, ignore_tuplets))
            for i, sym in enumerate(symbols_at):
                if i > 0:
                    tokens.append(EncodedSymbol("chord"))
                tokens.append(sym)

        right_bl = None
        next_left_bl = None
        if m_idx < len(first_staff_measures):
            right_bl = getattr(first_staff_measures[m_idx], "rightBarline", None)
            if m_idx + 1 < len(first_staff_measures):
                next_left_bl = getattr(first_staff_measures[m_idx + 1], "leftBarline", None)
        tokens.extend(_barline_tokens(right_bl, next_left_bl))

    return tokens


def _convert_m21_part(part: object, ignore_tuplets: bool = False) -> list[EncodedSymbol]:
    tokens: list[EncodedSymbol] = []
    measures = list(getattr(part, "getElementsByClass", lambda _: [])("Measure"))

    for m_idx, measure in enumerate(measures):
        next_measure = measures[m_idx + 1] if m_idx + 1 < len(measures) else None

        # Emit clef / key / time signature from this measure's attributes.
        for el in measure:
            if isinstance(el, m21.clef.Clef):
                tokens.append(_clef_to_symbol(el))
            elif isinstance(el, m21.key.KeySignature):
                tokens.append(EncodedSymbol(f"keySignature_{el.sharps}"))
            elif isinstance(el, m21.meter.TimeSignature):
                tokens.append(EncodedSymbol(f"timeSignature/{el.denominator}"))

        # Collect notes/rests from all voices, grouped by time offset.
        by_offset: dict[float, list[object]] = {}
        voices = list(getattr(measure, "voices", []))
        sources: list[object] = voices if voices else [measure]
        for source in sources:
            for el in getattr(source, "notesAndRests", []):
                off = float(getattr(el, "offset", 0.0))
                by_offset.setdefault(off, []).append(el)

        for off in sorted(by_offset):
            symbols_at: list[EncodedSymbol] = []
            for el in by_offset[off]:
                symbols_at.extend(_expand_element(el, ignore_tuplets))

            for i, sym in enumerate(symbols_at):
                if i > 0:
                    tokens.append(EncodedSymbol("chord"))
                tokens.append(sym)

        # Emit the barline between this measure and the next.
        right_bl = getattr(measure, "rightBarline", None)
        next_left_bl = getattr(next_measure, "leftBarline", None) if next_measure else None
        tokens.extend(_barline_tokens(right_bl, next_left_bl))

    return tokens


def convert_kern_to_parts_music21(lines: list[str]) -> list[list[EncodedSymbol]]:
    """Parse **kern via music21 and return one token list per part.

    Multi-movement kern files (parsed by music21 as an Opus) are handled by
    concatenating same-indexed parts across all scores, mirroring the native parser.
    The same normalisation steps are applied per score before concatenation:
    _remove_redundant_key_changes -> _fix_final_repeat_start -> strip_naturals
    """
    kern_text = "\n".join(lines)
    try:
        parsed = m21.converter.parse(kern_text, format="humdrum")
    except Exception:  # noqa: BLE001
        return []

    if isinstance(parsed, m21.stream.Opus):
        scores = list(getattr(parsed, "scores", []))
    else:
        scores = [parsed]

    n_parts = max((len(list(getattr(sc, "parts", []))) for sc in scores), default=0)
    result: list[list[EncodedSymbol]] = [[] for _ in range(n_parts)]
    for sc in scores:
        for p_idx, part in enumerate(getattr(sc, "parts", [])):
            tokens = _convert_m21_part(part)
            tokens = _remove_redundant_key_changes(tokens)
            tokens = _fix_final_repeat_start(tokens)
            tokens = strip_naturals(tokens)
            result[p_idx].extend(tokens)
    return result
