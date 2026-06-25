"""
NED scoring: given a kern ground truth and tool output, compute a NedResult.

Supports two scoring backends:
  - Native token-based comparison (fast, kern-specific component breakdown)
  - musicdiff structural comparison (slower, more robust cross-format alignment)

Public API: compute_ned, NedResult, TokenEvent
"""

import contextlib
import copy
import difflib
import io
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import editdistance

from homr.circle_of_fifths import strip_naturals
from homr.transformer.vocabulary import EncodedSymbol, sort_token_chords
from training.omr_datasets.humdrum_kern_parser import convert_kern_to_parts
from training.omr_datasets.music_xml_parser import music_xml_file_to_tokens


@dataclass
class NedResult:
    ned: float
    distance: int
    kern_len: int
    xml_len: int
    rhythm_ned: float
    pitch_ned: float
    lift_ned: float
    articulation_ned: float
    slur_ned: float


class TokenEvent(TypedDict):
    staff: str
    event_type: str  # 'match' | 'delete' | 'insert' | 'substitute'
    exp_rhythm: str | None
    exp_pitch: str | None
    exp_lift: str | None
    exp_articulation: str | None
    exp_slur: str | None
    act_rhythm: str | None
    act_pitch: str | None
    act_lift: str | None
    act_articulation: str | None
    act_slur: str | None


# ---------------------------------------------------------------------------
# Native token-based scoring
# ---------------------------------------------------------------------------


def _strip_position(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    return [EncodedSymbol(s.rhythm, s.pitch, s.lift, s.articulation, s.slur) for s in symbols]


def _flatten_part(measures: Sequence[Sequence[EncodedSymbol]]) -> list[EncodedSymbol]:
    flat = [s for measure in measures for s in measure]
    return [t for chord in sort_token_chords(flat) for t in chord]


def _component_dist(a: list[EncodedSymbol], b: list[EncodedSymbol], field: str) -> int:
    return editdistance.eval([getattr(s, field) for s in a], [getattr(s, field) for s in b])


def _split_grand_staff(xml_text: str) -> str:
    """
    Convert a single grand-staff <Part> into two separate <Part> elements.

    Tools like homr output piano music as one <Part> with <staves>2</staves> and each
    note tagged <staff>1</staff> or <staff>2</staff>.  music_xml_file_to_tokens treats
    this as a single voice, so all notes land in xml_upper and xml_lower is empty,
    inflating NED massively.  This function detects that pattern and splits the part
    before tokenisation so the staves are compared independently.
    """
    try:
        root = ET.fromstring(xml_text)  # noqa: S314
    except ET.ParseError:
        return xml_text

    parts = root.findall("part")
    if len(parts) != 1:
        return xml_text
    staves_el = parts[0].find(".//staves")
    if staves_el is None or int(staves_el.text or "1") < 2:
        return xml_text

    def _staff_num(el: ET.Element) -> int:
        s = el.find("staff")
        return int(s.text) if s is not None and s.text else 0

    # Build two lists of measure elements, one per staff.
    split: list[list[ET.Element]] = [[], []]
    for measure in parts[0].findall("measure"):
        for idx in range(2):
            split[idx].append(ET.Element("measure", measure.attrib))

        for child in measure:
            tag = child.tag

            if tag in ("backup", "forward"):
                continue  # timing reset between staves; not needed in split parts

            if tag == "note":
                s = _staff_num(child) - 1
                if 0 <= s < 2:
                    split[s][-1].append(child)
                continue

            if tag == "direction":
                s = _staff_num(child) - 1
                if 0 <= s < 2:
                    split[s][-1].append(child)
                else:
                    split[0][-1].append(child)
                    split[1][-1].append(copy.deepcopy(child))
                continue

            if tag == "attributes":
                attrs: list[ET.Element] = [ET.Element("attributes"), ET.Element("attributes")]
                for ac in child:
                    if ac.tag in ("staves", "part-symbol"):
                        continue
                    if ac.tag in ("clef", "staff-details"):
                        num = int(ac.get("number", "1")) - 1
                        if 0 <= num < 2:
                            el = copy.deepcopy(ac)
                            el.attrib.pop("number", None)
                            attrs[num].append(el)
                        continue
                    attrs[0].append(ac)
                    attrs[1].append(copy.deepcopy(ac))
                for idx, attr_el in enumerate(attrs):
                    if len(attr_el):
                        split[idx][-1].append(attr_el)
                continue

            # barline, print, sound, ... -> both parts
            split[0][-1].append(child)
            split[1][-1].append(copy.deepcopy(child))

    # Reconstruct the score with two parts.
    new_root = ET.Element(root.tag, root.attrib)
    for child in root:
        if child.tag == "part-list":
            pl = ET.SubElement(new_root, "part-list")
            for pid in ("P1", "P2"):
                sp = ET.SubElement(pl, "score-part", id=pid)
                ET.SubElement(sp, "part-name").text = pid
        elif child.tag != "part":
            new_root.append(copy.deepcopy(child))

    for idx, measures in enumerate(split):
        p = ET.SubElement(new_root, "part", id=f"P{idx + 1}")
        for m in measures:
            p.append(m)

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(new_root, encoding="unicode")


def _kern_parts(kern_text: str, kern_parser: str) -> list[list[EncodedSymbol]]:
    """Return per-part token lists using the selected kern parser."""
    if kern_parser == "music21":
        from training.omr_datasets.music21_kern_parser import (  # noqa: PLC0415
            convert_kern_to_parts_music21,
        )

        return convert_kern_to_parts_music21(kern_text.splitlines())
    return convert_kern_to_parts(kern_text.splitlines())  # "native"


def _xml_parts_from_text(xml_text: str, xml_parser: str) -> list[list[EncodedSymbol]]:
    """Return flat per-part token lists using the selected XML parser."""
    if xml_parser == "music21":
        from training.omr_datasets.music21_xml_parser import (  # noqa: PLC0415
            convert_xml_to_parts_music21,
        )

        return convert_xml_to_parts_music21(xml_text)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".musicxml", delete=False, encoding="utf-8"
    ) as f:
        f.write(xml_text)
        xml_path = f.name
    try:
        xml_voices = music_xml_file_to_tokens(xml_path)
    finally:
        os.remove(xml_path)
    return [_flatten_part(p) for p in xml_voices]


def _is_xml(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("<?xml") or "<score-partwise" in stripped[:500]


def _pred_parts(raw_output: str, kern_parser: str, xml_parser: str) -> list[list[EncodedSymbol]]:
    """Parse tool output (MusicXML or **kern) into per-part token lists.

    MusicXML is parsed with xml_parser; **kern is parsed directly with kern_parser
    - no intermediate kern->XML conversion.
    """
    if _is_xml(raw_output):
        xml_text = _split_grand_staff(raw_output)
        raw = _xml_parts_from_text(xml_text, xml_parser)
        return [_strip_position(strip_naturals(p)) for p in raw]
    raw = _kern_parts(raw_output, kern_parser)
    return [_strip_position([t for chord in sort_token_chords(p) for t in chord]) for p in raw]


def _parse_output(
    kern_text: str,
    raw_output: str,
    kern_parser: str = "native",
    xml_parser: str = "native",
) -> tuple[list[list[EncodedSymbol]], list[list[EncodedSymbol]]]:
    """Parse ground-truth kern and tool raw output into aligned per-part token lists."""
    kern_raw = _kern_parts(kern_text, kern_parser)
    gt_parts = [
        _strip_position([t for chord in sort_token_chords(p) for t in chord]) for p in kern_raw
    ]
    return gt_parts, _pred_parts(raw_output, kern_parser, xml_parser)


def _ned_from_parts(
    kern_parts: list[list[EncodedSymbol]],
    xml_parts: list[list[EncodedSymbol]],
) -> NedResult:
    n = max(len(kern_parts), len(xml_parts), 1)

    def _kp(i: int) -> list[EncodedSymbol]:
        return kern_parts[i] if i < len(kern_parts) else []

    def _xp(i: int) -> list[EncodedSymbol]:
        return xml_parts[i] if i < len(xml_parts) else []

    total_dist = sum(editdistance.eval(_kp(i), _xp(i)) for i in range(n))
    kern_len = sum(len(k) for k in kern_parts)
    xml_len = sum(len(x) for x in xml_parts)
    denominator = max(kern_len + xml_len, 1)

    def _cned(field: str) -> float:
        return sum(_component_dist(_kp(i), _xp(i), field) for i in range(n)) / denominator

    return NedResult(
        ned=total_dist / denominator,
        distance=total_dist,
        kern_len=kern_len,
        xml_len=xml_len,
        rhythm_ned=_cned("rhythm"),
        pitch_ned=_cned("pitch"),
        lift_ned=_cned("lift"),
        articulation_ned=_cned("articulation"),
        slur_ned=_cned("slur"),
    )


def _part_staff_name(i: int, n: int) -> str:
    if n == 2:
        return "upper" if i == 0 else "lower"
    return f"part_{i}"


def _events_for_parts(
    kern_parts: list[list[EncodedSymbol]],
    xml_parts: list[list[EncodedSymbol]],
) -> list[TokenEvent]:
    n = max(len(kern_parts), len(xml_parts), 1)
    events: list[TokenEvent] = []
    for i in range(n):
        k = kern_parts[i] if i < len(kern_parts) else []
        x = xml_parts[i] if i < len(xml_parts) else []
        events.extend(_alignment_events(k, x, _part_staff_name(i, n)))
    return events


def _make_event(
    staff: str,
    event_type: str,
    exp: EncodedSymbol | None,
    act: EncodedSymbol | None,
) -> TokenEvent:
    return TokenEvent(
        staff=staff,
        event_type=event_type,
        exp_rhythm=exp.rhythm if exp is not None else None,
        exp_pitch=exp.pitch if exp is not None else None,
        exp_lift=exp.lift if exp is not None else None,
        exp_articulation=exp.articulation if exp is not None else None,
        exp_slur=exp.slur if exp is not None else None,
        act_rhythm=act.rhythm if act is not None else None,
        act_pitch=act.pitch if act is not None else None,
        act_lift=act.lift if act is not None else None,
        act_articulation=act.articulation if act is not None else None,
        act_slur=act.slur if act is not None else None,
    )


def _alignment_events(
    expected: list[EncodedSymbol],
    actual: list[EncodedSymbol],
    staff: str,
) -> list[TokenEvent]:
    matcher = difflib.SequenceMatcher(None, expected, actual, autojunk=False)
    events: list[TokenEvent] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                events.append(_make_event(staff, "match", expected[i1 + k], actual[j1 + k]))
        elif tag == "delete":
            for k in range(i2 - i1):
                events.append(_make_event(staff, "delete", expected[i1 + k], None))
        elif tag == "insert":
            for k in range(j2 - j1):
                events.append(_make_event(staff, "insert", None, actual[j1 + k]))
        elif tag == "replace":
            n_exp = i2 - i1
            n_act = j2 - j1
            n_sub = min(n_exp, n_act)
            for k in range(n_sub):
                events.append(_make_event(staff, "substitute", expected[i1 + k], actual[j1 + k]))
            for k in range(n_sub, n_exp):
                events.append(_make_event(staff, "delete", expected[i1 + k], None))
            for k in range(n_sub, n_act):
                events.append(_make_event(staff, "insert", None, actual[j1 + k]))
    return events


def compute_ned(kern_text: str, raw_output: str) -> NedResult:
    """Compute OMR-NED between kern ground truth and tool output (MusicXML or **kern)."""
    kern_parts, pred_parts = _parse_output(kern_text, raw_output)
    return _ned_from_parts(kern_parts, pred_parts)


# ---------------------------------------------------------------------------
# musicdiff-based scoring
# ---------------------------------------------------------------------------


def _musicdiff_register_once() -> None:
    """Register converter21 and apply musicdiff memoizer performance patch (idempotent)."""
    from types import SimpleNamespace  # noqa: PLC0415

    import converter21  # noqa: PLC0415
    import musicdiff.comparison as _md_cmp  # noqa: PLC0415

    converter21.register()

    # Every memoizer cache hit calls copy.deepcopy on the (ops_list, edit_distance) result,
    # recursively cloning potentially large op lists on each hit. Replace with a shallow copy
    # since ops are never mutated after creation. Technique from transcoda/evaluation/omr_ned.py.
    current_deepcopy = _md_cmp.copy.deepcopy  # type: ignore[attr-defined]
    if not getattr(current_deepcopy, "_homr_fast_memo_copy", False):
        _orig = current_deepcopy

        def _fast_deepcopy(value: object, memo: object = None) -> object:
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], list):
                return (list(value[0]), value[1])
            return _orig(value, memo)

        _fast_deepcopy._homr_fast_memo_copy = True  # type: ignore[attr-defined]
        _md_cmp.copy = SimpleNamespace(  # type: ignore[attr-defined,assignment]
            copy=_md_cmp.copy.copy,  # type: ignore[attr-defined]
            deepcopy=_fast_deepcopy,
        )


def _musicdiff_parse_scores(kern_text: str, raw_output: str) -> tuple:
    """Parse kern ground truth and tool output into music21 Score objects.

    For kern predictions, acceptSyntaxErrors=True lets converter21 repair malformed
    measure durations (e.g. notes that don't fill a bar) rather than failing outright.
    Each repaired error is counted as an additional edit-distance unit via
    AnnScore.num_syntax_errors_fixed, so they are penalised without completely
    breaking the alignment.
    """
    import music21 as m21  # noqa: PLC0415

    with contextlib.redirect_stderr(io.StringIO()):
        gt_raw = m21.converter.parse(kern_text, format="humdrum")
    if isinstance(gt_raw, m21.stream.Opus):
        gt_raw = gt_raw.scores[0] if gt_raw.scores else m21.stream.Score()
    gt_score: m21.stream.Score = (
        gt_raw if isinstance(gt_raw, m21.stream.Score) else m21.stream.Score()
    )

    if _is_xml(raw_output):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".musicxml", delete=False, encoding="utf-8"
        ) as f:
            f.write(raw_output)
            pred_path = f.name
        try:
            pred_raw = m21.converter.parse(pred_path, forceSource=True)
        finally:
            os.remove(pred_path)
    else:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".krn", delete=False, encoding="utf-8"
        ) as f:
            f.write(raw_output)
            pred_path = f.name
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                pred_raw = m21.converter.parse(pred_path, forceSource=True, acceptSyntaxErrors=True)
        finally:
            os.remove(pred_path)

    if isinstance(pred_raw, m21.stream.Opus):
        pred_raw = pred_raw.scores[0] if pred_raw.scores else m21.stream.Score()
    pred_score: m21.stream.Score = (
        pred_raw if isinstance(pred_raw, m21.stream.Score) else m21.stream.Score()
    )
    return gt_score, pred_score


def _musicdiff_ned_for_sample(kern_text: str, raw_output: str) -> NedResult:
    """Compute OMR-NED using musicdiff's full structural comparison (DetailLevel.Default).

    Component NEDs (rhythm, pitch, lift, articulation, slur) are not available in this
    mode and are returned as NaN. The overall NED uses musicdiff's own formula:
    (OMR-ED + syntax_fixes) / (numsyms_gt + numsyms_pred).

    Call _musicdiff_register_once() before entering a batch loop.
    """
    from musicdiff.annotation import AnnScore  # noqa: PLC0415
    from musicdiff.comparison import Comparison  # noqa: PLC0415

    gt_score, pred_score = _musicdiff_parse_scores(kern_text, raw_output)

    ann_gt: AnnScore = AnnScore(gt_score)
    ann_pred: AnnScore = AnnScore(pred_score)

    gt_numsyms = ann_gt.notation_size()
    pred_numsyms = ann_pred.notation_size()
    syntax_fixes: int = ann_pred.num_syntax_errors_fixed

    _, omr_ed = Comparison.annotated_scores_diff(ann_pred, ann_gt)
    total_ed = omr_ed + syntax_fixes
    total_syms = gt_numsyms + pred_numsyms
    omr_ned = total_ed / total_syms if total_syms > 0 else 0.0

    nan = float("nan")
    return NedResult(
        ned=omr_ned,
        distance=total_ed,
        kern_len=gt_numsyms,
        xml_len=pred_numsyms,
        rhythm_ned=nan,
        pitch_ned=nan,
        lift_ned=nan,
        articulation_ned=nan,
        slur_ned=nan,
    )


def _musicdiff_detailed_ned_for_sample(kern_text: str, raw_output: str) -> NedResult:
    """Compute OMR-NED with per-component breakdown using multiple musicdiff DetailLevel runs.

    Runs 5 separate comparisons to isolate component costs:
      rhythm_ned       - notes/rests only (DetailLevel.NotesAndRests); covers pitch, duration,
                         and accidental errors without any decorations
      pitch_ned        - NaN (pitch cannot be cleanly separated from rhythm within a note)
      lift_ned         - beams/flags NED, isolated from note errors
                         = (cost(NotesAndRests|Beams) - cost(NotesAndRests)) / beam_symbols
      articulation_ned - articulations NED, isolated from note errors
                         = (cost(NotesAndRests|Articulations) - cost(NotesAndRests)) / artic_symbols
      slur_ned         - slurs NED (DetailLevel.Slurs; independent of NotesAndRests)

    Slower than the plain 'musicdiff' mode due to the additional comparisons.
    Call _musicdiff_register_once() before entering a batch loop.
    """
    from musicdiff.annotation import AnnScore  # noqa: PLC0415
    from musicdiff.comparison import Comparison  # noqa: PLC0415
    from musicdiff.detaillevel import DetailLevel  # noqa: PLC0415

    gt_score, pred_score = _musicdiff_parse_scores(kern_text, raw_output)

    def _compare(dl: int) -> tuple[int, int, int, int]:
        """Return (ed, syntax_fixes, gt_size, pred_size) for one DetailLevel."""
        ann_gt = AnnScore(gt_score, detail=dl)
        ann_pred = AnnScore(pred_score, detail=dl)
        _, ed = Comparison.annotated_scores_diff(ann_pred, ann_gt)
        return (
            ed,
            ann_pred.num_syntax_errors_fixed,
            ann_gt.notation_size(),
            ann_pred.notation_size(),
        )  # noqa: E501

    ed_full, syntax_fixes, gt_full, pred_full = _compare(DetailLevel.Default)
    total_syms = gt_full + pred_full
    ned = (ed_full + syntax_fixes) / total_syms if total_syms > 0 else 0.0

    ed_notes, _, gt_notes, pred_notes = _compare(DetailLevel.NotesAndRests)
    notes_syms = gt_notes + pred_notes
    rhythm_ned = ed_notes / notes_syms if notes_syms > 0 else 0.0

    ed_beams, _, gt_beams, pred_beams = _compare(DetailLevel.NotesAndRests | DetailLevel.Beams)
    beam_cost = ed_beams - ed_notes
    beam_syms = (gt_beams + pred_beams) - (gt_notes + pred_notes)
    lift_ned = beam_cost / beam_syms if beam_syms > 0 else 0.0

    ed_artic, _, gt_artic, pred_artic = _compare(
        DetailLevel.NotesAndRests | DetailLevel.Articulations
    )
    artic_cost = ed_artic - ed_notes
    artic_syms = (gt_artic + pred_artic) - (gt_notes + pred_notes)
    articulation_ned = artic_cost / artic_syms if artic_syms > 0 else 0.0

    ed_slurs, _, gt_slurs, pred_slurs = _compare(DetailLevel.Slurs)
    slur_syms = gt_slurs + pred_slurs
    slur_ned = ed_slurs / slur_syms if slur_syms > 0 else 0.0

    return NedResult(
        ned=ned,
        distance=ed_full + syntax_fixes,
        kern_len=gt_full,
        xml_len=pred_full,
        rhythm_ned=rhythm_ned,
        pitch_ned=float("nan"),
        lift_ned=lift_ned,
        articulation_ned=articulation_ned,
        slur_ned=slur_ned,
    )
