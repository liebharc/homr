"""Repairs the grouping of staffs into systems before the transcription."""

import itertools
import math
from collections import Counter

import numpy as np

from homr import constants
from homr.debug import Debug
from homr.model import MultiStaff, Staff
from homr.simple_logging import eprint


def _flatten_staffs(staffs: list[MultiStaff]) -> list[Staff]:
    return [s for multi_staff in staffs for s in multi_staff.staffs]


def _layout(row: MultiStaff) -> tuple[bool, ...]:
    """The layout of a system: is_grandstaff of each staff."""
    return tuple(staff.is_grandstaff for staff in row.staffs)


def _is_uniform(staffs: list[MultiStaff]) -> bool:
    return len({_layout(row) for row in staffs}) == 1


def _most_common(layouts: Counter[tuple[bool, ...]]) -> tuple[bool, ...] | None:
    """The most common layout, None if there is none or two are equally common."""
    if len(layouts) == 0:
        return None
    (layout, count), *others = layouts.most_common(2)
    if others and others[0][1] == count:
        return None
    return layout


def _vertical_gap(upper: Staff, lower: Staff) -> float | None:
    """The gap between two staffs where both have points, None if there is no such x."""
    upper_points = {round(point.x): point for point in upper.grid}
    gaps = [
        point.y[0] - upper_points[round(point.x)].y[-1]
        for point in lower.grid
        if round(point.x) in upper_points
    ]
    return float(np.median(gaps)) if gaps else None


def _possible_grandstaffs(staffs: list[MultiStaff], strict: bool) -> set[tuple[int, int]]:
    """
    Pairs of neighboring single staffs (by id) which can form a grand staff: they
    overlap and, in unit sizes, are not further apart than the grand staffs on the
    page. If strict, staffs of different systems must also be closer to each other
    than to single staffs right above and below, else the pairing is a guess.
    """
    flat_staffs = _flatten_staffs(staffs)
    system_of = {id(staff): i for i, row in enumerate(staffs) for staff in row.staffs}
    lines = constants.number_of_lines_on_a_staff
    grandstaff_gaps = [
        (point.y[lines] - point.y[lines - 1]) * (lines - 1) / (point.y[lines - 1] - point.y[0])
        for staff in flat_staffs
        if staff.is_grandstaff
        for point in staff.grid
        if len(point.y) == 2 * lines
    ]
    max_gap = 1.5 * float(np.median(grandstaff_gaps)) if grandstaff_gaps else math.inf
    measured = [_vertical_gap(upper, lower) for upper, lower in itertools.pairwise(flat_staffs)]
    # Without overlap, the distance of the bounding boxes is good enough for a comparison
    gaps = [
        lower.min_y - upper.max_y if gap is None else gap
        for gap, (upper, lower) in zip(measured, itertools.pairwise(flat_staffs), strict=True)
    ]
    result = set()
    for i, (upper, lower) in enumerate(itertools.pairwise(flat_staffs)):
        gap = measured[i]
        unit_size = min(upper.average_unit_size, lower.average_unit_size)
        if upper.is_grandstaff or lower.is_grandstaff or gap is None or gap > max_gap * unit_size:
            continue
        neighbors = [
            gaps[j]
            for j in (i - 1, i + 1)
            if 0 <= j < len(gaps)
            and not flat_staffs[j].is_grandstaff
            and not flat_staffs[j + 1].is_grandstaff
        ]
        connected = system_of[id(upper)] == system_of[id(lower)]
        if not strict or connected or all(gap < 0.8 * n for n in neighbors):  # noqa: PLR2004
            result.add((id(upper), id(lower)))
    return result


def _fit_to_layout(
    staffs: list[Staff], layout: tuple[bool, ...], grandstaffs: set[tuple[int, int]]
) -> list[Staff] | None:
    """
    Fits the staffs to the layout. Two single staffs in grandstaffs can form a grand
    staff, if its brace was missed. Returns None if the staffs don't fit.
    """
    result: list[Staff] = []
    i = 0
    for is_grandstaff in layout:
        if i < len(staffs) and staffs[i].is_grandstaff == is_grandstaff:
            result.append(staffs[i])
            i += 1
        elif (
            is_grandstaff
            and i + 1 < len(staffs)
            and not staffs[i].is_grandstaff
            and not staffs[i + 1].is_grandstaff
            and (id(staffs[i]), id(staffs[i + 1])) in grandstaffs
        ):
            result.append(staffs[i].merge(staffs[i + 1]))
            i += 2
        else:
            return None
    return result if i == len(staffs) else None


def _fit_neighbors(
    staffs: list[MultiStaff],
    start: int,
    layout: tuple[bool, ...],
    grandstaffs: set[tuple[int, int]],
) -> tuple[list[Staff], int] | None:
    """
    Fits the systems from start on to the layout. Returns the staffs and the next index.
    Without a grand staff in the layout, only systems with a connected system among
    them are joined, else nothing tells which single staffs belong together.
    """
    for end in range(start + 1, len(staffs) + 1):
        group = _flatten_staffs(staffs[start:end])
        if len(group) > 2 * len(layout):
            return None
        if not any(layout) and len(group) == end - start > 1:
            continue
        fitted = _fit_to_layout(group, layout, grandstaffs)
        if fitted is not None:
            return fitted, end
    return None


def _repair_systems(
    staffs: list[MultiStaff], layout: tuple[bool, ...], strict: bool = False
) -> list[MultiStaff]:
    """
    Splits systems which repeat the layout and joins neighbor systems which together
    have it. Other systems are kept, as the layout can
    change on a page, e.g. a piano intro before the voice enters. Two single staffs
    become a grand staff only if they pair up, see _possible_grandstaffs.
    """
    grandstaffs = _possible_grandstaffs(staffs, strict)
    result: list[MultiStaff] = []
    i = 0
    while i < len(staffs):
        if _layout(staffs[i]) == layout:
            result.append(staffs[i])
            i += 1
            continue
        row = staffs[i].staffs
        repeats = len(row) // len(layout)
        if repeats > 1 and _layout(staffs[i]) == layout * repeats:
            result.extend(
                MultiStaff(row[k : k + len(layout)], []) for k in range(0, len(row), len(layout))
            )
            i += 1
            continue
        fit = _fit_neighbors(staffs, i, layout, grandstaffs)
        if fit is None:
            result.append(staffs[i])
            i += 1
        else:
            fitted, i = fit
            result.append(MultiStaff(fitted, []))
    return result


def _repair_score(staffs: list[MultiStaff], layout: tuple[bool, ...]) -> float:
    """
    The staves in systems with the layout after the repair. Systems which split a
    detected system or merge single staffs into a grand staff count half, as they
    contradict the detection. Joined neighbor systems count fully, as connections are
    often missed.
    """
    system_of = {id(staff): row for row in staffs for staff in row.staffs}
    score = 0.0
    for row in _repair_systems(staffs, layout, strict=True):
        if _layout(row) != layout:
            continue
        sources = [system_of.get(id(staff)) for staff in row.staffs]
        contradicts = any(
            source is None or not set(map(id, source.staffs)) <= set(map(id, row.staffs))
            for source in sources
        )
        score += (len(row.staffs) + sum(layout)) * (0.5 if contradicts else 1)
    return score


def _common_layout(staffs: list[MultiStaff]) -> tuple[bool, ...] | None:
    """
    The layout most systems share. Systems and neighbor systems which mix single and
    grand staffs decide, if a layout of them occurs at least twice, as they show how a
    system is built. Otherwise the layout which repairs the most staves decides, e.g.
    two single staffs as close as a grand staff whose brace was missed. Layouts of
    several grand staffs need to be detected twice, as a crease can connect them.
    """
    mixed: Counter[tuple[bool, ...]] = Counter()
    for row in staffs:
        if len(set(_layout(row))) > 1:
            mixed[_layout(row)] += 1
    for upper, lower in itertools.pairwise(staffs):
        pair = _layout(upper) + _layout(lower)
        if pair in ((False, True), (True, False)):
            mixed[pair] += 1
    layout = _most_common(mixed)
    if layout is not None and mixed[layout] > 1:
        return layout
    counts = Counter(_layout(row) for row in staffs)
    candidates = [
        layout
        for layout in sorted(set(counts) | set(mixed))
        if not (len(layout) > 1 and all(layout) and counts[layout] < 2)  # noqa: PLR2004
    ]
    if len(candidates) == 0:
        return None
    return max(candidates, key=lambda layout: (_repair_score(staffs, layout), counts[layout]))


def _drop_single_staff_at_end(
    staffs: list[MultiStaff], layout: tuple[bool, ...]
) -> list[MultiStaff]:
    """
    On a page of grand staffs, a single staff below the last system is most likely a
    falsely detected staff. Drops it if it is the only single staff.
    """
    single = [row for row in staffs if _layout(row) == (False,)]
    only_grandstaffs = False not in layout
    if not only_grandstaffs or single != [staffs[-1]] or len(staffs) < 3:  # noqa: PLR2004
        return staffs
    eprint("Removing a single staff below the last system, it doesn't fit the layout")
    return staffs[:-1]


def _find_periodic_core(flat_staffs: list[Staff]) -> tuple[int, int, int] | None:
    """
    Finds the shortest repeating layout of single and grand staffs, e.g. a voice and a
    piano. Up to one period of other staffs is allowed at each edge of the page.
    Returns (period, front_trim, back_trim) with the least trim, then the shortest
    period, or None if the layout doesn't repeat at least twice.
    """
    layout = [s.is_grandstaff for s in flat_staffs]
    n = len(layout)
    best: tuple[int, int, int, int] | None = None
    for period in range(1, n // 2 + 1):
        for front_trim in range(period + 1):
            for back_trim in range(period + 1):
                core = layout[front_trim : n - back_trim]
                if len(core) < 2 * period or len(core) % period != 0:
                    continue
                rows = [tuple(core[i : i + period]) for i in range(0, len(core), period)]
                if not all(row == rows[0] for row in rows):
                    continue
                candidate = (front_trim + back_trim, period, front_trim, back_trim)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
    if best is None:
        return None
    _, period, front_trim, back_trim = best
    return period, front_trim, back_trim


def _regroup_by_pattern(staffs: list[MultiStaff]) -> list[MultiStaff] | None:
    """
    Groups the staffs by a repeating pattern of single and grand staffs. Staffs at the
    edges which break the pattern stay systems of their own, but only if nothing was
    grouped before. Returns None if there is no such pattern.
    """
    flat_staffs = _flatten_staffs(staffs)
    core = _find_periodic_core(flat_staffs)
    if core is None:
        return None
    period, front_trim, back_trim = core
    end = len(flat_staffs) - back_trim
    pattern = {staff.is_grandstaff for staff in flat_staffs[front_trim : front_trim + period]}
    mixes_single_and_grandstaffs = len(pattern) == 2  # noqa: PLR2004
    trimmed = front_trim > 0 or back_trim > 0
    nothing_grouped = all(len(row.staffs) == 1 for row in staffs)
    if not mixes_single_and_grandstaffs or (trimmed and not nothing_grouped):
        return None
    eprint("Systems repeat every", period, "staffs, grouping them by this pattern")
    return (
        [MultiStaff([staff], []) for staff in flat_staffs[:front_trim]]
        + [MultiStaff(flat_staffs[i : i + period], []) for i in range(front_trim, end, period)]
        + [MultiStaff([staff], []) for staff in flat_staffs[end:]]
    )


def _convert_to_layout(row: MultiStaff, layout: tuple[bool, ...]) -> MultiStaff | None:
    """
    Merges or splits grand staffs so that the row has the layout. Returns None if a
    grand staff can't be split or two staffs can't be merged.
    """
    if _layout(row) == layout:
        return row
    single_staffs: list[Staff] = []
    for staff in row.staffs:
        parts = staff.split() if staff.is_grandstaff else (staff,)
        if parts is None:
            return None
        single_staffs.extend(parts)
    remaining = iter(single_staffs)
    staffs = []
    for is_grandstaff in layout:
        staff = next(remaining)
        if is_grandstaff:
            lower = next(remaining)
            if _vertical_gap(staff, lower) is None:
                return None
            staff = staff.merge(lower)
        staffs.append(staff)
    return MultiStaff(staffs, row.connections)


def _unify_grand_staffs(staffs: list[MultiStaff]) -> list[MultiStaff]:
    """
    If every system has the same number of staffs, but in some systems a brace was
    missed or wrongly found, converts them to the most common layout.
    """
    staffs_per_system = {sum(2 if s.is_grandstaff else 1 for s in row.staffs) for row in staffs}
    if len(staffs_per_system) != 1:
        return staffs
    layout = _most_common(Counter(_layout(row) for row in staffs))
    if layout is None:
        return staffs
    result = []
    for row in staffs:
        converted = _convert_to_layout(row, layout)
        if converted is None:
            return staffs
        result.append(converted)
    return result


def _ensure_same_number_of_staffs(staffs: list[MultiStaff]) -> list[MultiStaff]:
    """
    Repairs the grouping of staffs into systems towards the layout most systems share.
    Systems with another layout are kept, as the layout can change on a page.
    """
    staffs = _unify_grand_staffs(staffs)
    if _is_uniform(staffs) and len(staffs[0].staffs) > 1:
        return staffs
    layout = _common_layout(staffs)
    repaired = staffs if layout is None else _repair_systems(staffs, layout)
    if not _is_uniform(repaired):
        # Nothing grouped, or e.g. a crease connected staffs of different systems
        regrouped = _regroup_by_pattern(staffs)
        if regrouped is not None:
            return regrouped
    return repaired if layout is None else _drop_single_staff_at_end(repaired, layout)


def repair_systems(debug: Debug, staffs: list[MultiStaff]) -> list[MultiStaff]:
    """Repairs the grouping of staffs into systems, see _ensure_same_number_of_staffs."""
    staffs = _ensure_same_number_of_staffs(staffs)
    eprint("Systems after the repair of the layout:", [len(staff.staffs) for staff in staffs])
    debug.write_bounding_boxes_alternating_colors("systems", staffs)
    return staffs
