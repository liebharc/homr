"""
Alternative MusicXML -> EncodedSymbol parser built on music21.

Drop-in replacement for music_xml_file_to_tokens().  Returns one flat token list
per part; the same normalization steps as the music21 kern parser are applied.
"""

from __future__ import annotations

import music21 as m21
import music21.stream

from homr.transformer.vocabulary import EncodedSymbol, sort_token_chords
from training.omr_datasets.music21_kern_parser import (
    _convert_m21_part,
    _convert_m21_staff_group,
    _fix_final_repeat_start,
    _remove_redundant_key_changes,
)


def convert_xml_to_parts_music21(xml_text: str) -> list[list[EncodedSymbol]]:
    """Parse MusicXML via music21 and return one flat token list per part.

    Multi-staff parts (e.g., piano grand staff) are detected via StaffGroup spanners
    and merged into a single token list, matching the native parser's behavior of
    keeping both staves of a grand staff as one part.
    chord markers are resolved via sort_token_chords before returning.
    strip_naturals and _strip_position are applied by the caller.
    """
    try:
        score = m21.converter.parse(xml_text, format="musicxml")
    except Exception:  # noqa: BLE001
        return []

    staff_to_group: dict[int, int] = {}
    group_to_staves: dict[int, list[m21.stream.PartStaff]] = {}
    for sg in score.spannerBundle.getByClass("StaffGroup"):
        gid = id(sg)
        staves = [s for s in sg.spannerStorage if isinstance(s, m21.stream.PartStaff)]
        if not staves:
            continue
        group_to_staves[gid] = staves
        for staff in staves:
            staff_to_group[id(staff)] = gid

    result: list[list[EncodedSymbol]] = []
    emitted_groups: set[int] = set()

    for part in getattr(score, "parts", []):
        part_gid = staff_to_group.get(id(part))
        if part_gid is not None:
            gid = part_gid
            if gid not in emitted_groups:
                emitted_groups.add(gid)
                tokens = _convert_m21_staff_group(group_to_staves[gid])
                tokens = _remove_redundant_key_changes(tokens)
                tokens = _fix_final_repeat_start(tokens)
                tokens = [t for chord in sort_token_chords(tokens) for t in chord]
                result.append(tokens)
        else:
            tokens = _convert_m21_part(part)
            tokens = _remove_redundant_key_changes(tokens)
            tokens = _fix_final_repeat_start(tokens)
            tokens = [t for chord in sort_token_chords(tokens) for t in chord]
            result.append(tokens)

    return result
