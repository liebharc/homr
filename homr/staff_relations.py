"""
Decides how neighboring staffs relate: separate systems, the same system or a grand
staff. A small CNN looks at the left margin of the two staffs, where braces, brackets
and the bar line at the start of a system are. The crop follows the staff lines, so that
page curl and perspective change it only little.
"""

import os

import cv2
import numpy as np

from homr.model import InputPredictions, MultiStaff, Staff, StaffPoint
from homr.type_definitions import NDArray

SEPARATE, SAME_SYSTEM, GRAND_STAFF = range(3)
CLASSES = ["separate", "same system", "grand staff"]
# Probability above which the model is sure that two staffs are in the same system
certain_probability = 0.9

# Resolution of the crops in pixels per unit size (distance of two staff lines)
pixels_per_unit = 4
# Units left of the staffs and of the start of the staffs in the crop
margin_units = 10
start_units = 14
# Rows of the crop: 3 units above, upper staff, gap, lower staff, 3 units below
staff_rows = 16
outer_rows = 12
gap_rows = 40
crop_height = 2 * (outer_rows + staff_rows) + gap_rows
crop_width = (margin_units + start_units) * pixels_per_unit
channels = 5

script_location = os.path.dirname(os.path.realpath(__file__))
model_name = "staff_relations_pdmx2-cc40992a36690003f7ed2bebe197016f19242fa6.onnx"
model_path = os.path.join(script_location, "staff_relations", model_name)


def _line_at(staff: Staff, line: int, x: NDArray) -> NDArray:
    """The y of a staff line at the given x, constant beyond the ends of the staff."""
    xs = np.array([point.x for point in staff.grid])
    ys = np.array([point.y[line] for point in staff.grid])
    return np.interp(x, xs, ys)


def page_channels(predictions: InputPredictions, unit_size: float) -> tuple[NDArray, float]:
    """
    The image and the segmentation as channels (height, width, channels) in [0, 1],
    scaled down to pixels_per_unit. Returns the channels and the scale.
    """
    scale = min(1.0, pixels_per_unit / unit_size)
    gray = predictions.preprocessed
    if gray.ndim == 3:  # noqa: PLR2004
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    layers = [
        255 - gray,
        predictions.staff * 255,
        predictions.symbols * 255,
        predictions.stems_rest * 255,
        predictions.clefs_keys * 255,
    ]
    size = (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale)))
    scaled = [
        cv2.resize(np.clip(layer, 0, 255).astype(np.uint8), size, interpolation=cv2.INTER_AREA)
        for layer in layers
    ]
    return np.stack(scaled, axis=-1).astype(np.float32) / 255, scale


def _remap(page: NDArray, map_x: NDArray, map_y: NDArray) -> NDArray:
    # cv2.remap handles at most 4 channels
    parts = [
        cv2.remap(
            page[..., i : i + 4],
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for i in range(0, page.shape[-1], 4)
    ]
    parts = [p if p.ndim == 3 else p[..., np.newaxis] for p in parts]  # noqa: PLR2004
    return np.concatenate(parts, axis=-1)


def _rows(upper: Staff, lower: Staff, x: NDArray) -> NDArray:
    """The y of every crop row at the given x (rows x columns), in page coordinates."""
    unit = (upper.average_unit_size + lower.average_unit_size) / 2
    upper_top, upper_bottom = _line_at(upper, 0, x), _line_at(upper, -1, x)
    lower_top, lower_bottom = _line_at(lower, 0, x), _line_at(lower, -1, x)
    # Anchors: row index -> y, interpolated linearly in between
    anchor_rows = np.cumsum([0, outer_rows, staff_rows, gap_rows, staff_rows, outer_rows])
    anchor_ys = [
        upper_top - 3 * unit,
        upper_top,
        upper_bottom,
        lower_top,
        lower_bottom,
        lower_bottom + 3 * unit,
    ]
    rows = np.empty((crop_height, len(x)))
    for i in range(len(anchor_rows) - 1):
        start, end = anchor_rows[i], anchor_rows[i + 1]
        share = ((np.arange(start, end) - start) / (end - start))[:, np.newaxis]
        rows[start:end] = anchor_ys[i] + share * (anchor_ys[i + 1] - anchor_ys[i])
    return rows


def pair_crop(page: NDArray, scale: float, upper: Staff, lower: Staff) -> NDArray:
    """
    The left margin and the start of two vertically neighboring staffs (channels, height,
    width) at pixels_per_unit.
    """
    unit = (upper.average_unit_size + lower.average_unit_size) / 2
    left = min(upper.min_x, lower.min_x)
    x = left + (np.arange(crop_width) / pixels_per_unit - margin_units) * unit
    rows = _rows(upper, lower, x)
    map_x = np.broadcast_to(x, rows.shape) * scale
    crop = _remap(page, map_x, rows * scale)
    return np.ascontiguousarray(crop.transpose(2, 0, 1))


def overlaps_horizontally(upper: Staff, lower: Staff) -> bool:
    overlap = min(upper.max_x, lower.max_x) - max(upper.min_x, lower.min_x)
    return overlap > 0.3 * min(upper.max_x - upper.min_x, lower.max_x - lower.min_x)


def vertical_pairs(staffs: list[Staff]) -> list[tuple[int, int]]:
    """Each staff with the next staff below it, among those which overlap horizontally."""
    order = sorted(range(len(staffs)), key=lambda i: staffs[i].min_y)
    pairs = []
    for position, i in enumerate(order):
        below = [
            j
            for j in order[position + 1 :]
            if overlaps_horizontally(staffs[i], staffs[j]) and staffs[j].min_y > staffs[i].min_y
        ]
        if below:
            pairs.append((i, min(below, key=lambda j: staffs[j].min_y)))
    return pairs


def page_inputs(
    predictions: InputPredictions, staffs: list[Staff]
) -> tuple[list[tuple[int, int]], NDArray]:
    """The vertical pairs of a page with their crops."""
    pairs = vertical_pairs(staffs)
    if len(pairs) == 0:
        return pairs, np.zeros((0, channels, crop_height, crop_width), dtype=np.float32)
    unit = float(np.median([staff.average_unit_size for staff in staffs]))
    page, scale = page_channels(predictions, unit)
    crops = np.stack([pair_crop(page, scale, staffs[i], staffs[j]) for i, j in pairs])
    return pairs, crops.astype(np.float32)


class StaffRelationModel:
    def __init__(self, path: str = model_path) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def predict(self, crops: NDArray, feats: NDArray | None = None) -> NDArray:
        """Probabilities of the classes for each pair (pairs x classes)."""
        if len(crops) == 0:
            return np.zeros((0, len(CLASSES)), dtype=np.float32)
        inputs = {"crop": crops} if feats is None else {"crop": crops, "features": feats}
        logits = self.session.run(None, inputs)[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


def _select_grand_staffs(pairs: list[tuple[int, int]], probabilities: NDArray) -> list[int]:
    """Indices of the pairs which form grand staffs, each staff is in at most one."""
    candidates = [
        k
        for k, p in enumerate(probabilities)
        if p[GRAND_STAFF] == max(p) and p[SEPARATE] < 0.5  # noqa: PLR2004
    ]
    used: set[int] = set()
    selected = []
    for k in sorted(candidates, key=lambda k: -probabilities[k][GRAND_STAFF]):
        if pairs[k][0] in used or pairs[k][1] in used:
            continue
        used.update(pairs[k])
        selected.append(k)
    return selected


def _merge_grand_staff(upper: Staff, lower: Staff) -> Staff:
    """
    Staff.merge keeps the x positions which both grids have, as without the model. If
    they have none, the lower staff is resampled to the grid of the upper one.
    """
    common = {round(p.x) for p in upper.grid} & {round(p.x) for p in lower.grid}
    if len(common) >= 2:  # noqa: PLR2004
        return upper.merge(lower)
    points = [p for p in upper.grid if lower.min_x <= p.x <= lower.max_x]
    if len(points) == 0:
        return upper
    x = np.array([p.x for p in points])
    lines = np.array([_line_at(lower, line, x) for line in range(len(lower.grid[0].y))])
    resampled = Staff(
        [StaffPoint(p.x, [float(y) for y in lines[:, k]], p.angle) for k, p in enumerate(points)]
    )
    resampled.symbols = lower.symbols
    cut = Staff(points)
    cut.symbols = upper.symbols
    return cut.merge(resampled)


def group_staffs(
    staffs: list[Staff], pairs: list[tuple[int, int]], probabilities: NDArray
) -> list[MultiStaff]:
    """
    Joins the staffs of a pair into one system if the model doesn't see them as separate,
    and merges the pairs which it sees as grand staffs. A system of several staffs is certain
    if the model is sure that all its pairs belong together.
    """
    system_of = list(range(len(staffs)))

    def find(i: int) -> int:
        while system_of[i] != i:
            i = system_of[i]
        return i

    for (i, j), p in zip(pairs, probabilities, strict=True):
        if p[SEPARATE] < 0.5:  # noqa: PLR2004
            system_of[find(j)] = find(i)
    grand_staffs = {pairs[k][0]: pairs[k][1] for k in _select_grand_staffs(pairs, probabilities)}
    lower_staffs = set(grand_staffs.values())
    systems: dict[int, list[Staff]] = {}
    for i, staff in enumerate(staffs):
        if i in lower_staffs:
            continue
        if i in grand_staffs:
            staff = _merge_grand_staff(staff, staffs[grand_staffs[i]])  # noqa: PLW2901
        systems.setdefault(find(i), []).append(staff)
    uncertain = {
        find(i)
        for (i, j), p in zip(pairs, probabilities, strict=True)
        if find(i) == find(j) and p[SEPARATE] > 1 - certain_probability
    }
    result = [
        MultiStaff(system, [], certain=len(system) > 1 and root not in uncertain)
        for root, system in systems.items()
    ]
    return sorted(result, key=lambda system: system.staffs[0].min_y)


_models: dict[str, StaffRelationModel] = {}


def get_model(path: str = model_path) -> StaffRelationModel:
    if path not in _models:
        _models[path] = StaffRelationModel(path)
    return _models[path]


def join_pieces_with_model(predictions: InputPredictions, staffs: list[Staff]) -> list[Staff]:
    """Joins staffs which the model sees as pieces of one staff."""
    pairs = horizontal_pairs(staffs)
    if len(pairs) == 0 or not os.path.exists(horizontal_model_path):
        return staffs
    unit = float(np.median([staff.average_unit_size for staff in staffs]))
    page, scale = page_channels(predictions, unit)
    crops, feats = horizontal_inputs(page, scale, staffs, pairs)
    probabilities = get_model(horizontal_model_path).predict(crops, feats)
    return join_staff_pieces(staffs, pairs, probabilities)


# Horizontal pairs: two staffs in the same row, which can be pieces of one staff
SAME_STAFF = 1
# Columns of the end of the left staff, of the gap and of the start of the right staff
end_units = 16
horizontal_end_width = end_units * pixels_per_unit
horizontal_gap_width = 64
horizontal_crop_width = 2 * horizontal_end_width + horizontal_gap_width
horizontal_crop_height = 2 * outer_rows + staff_rows
horizontal_features = 5
horizontal_model_name = "staff_pieces_h1-165e413e0edb2d8d2486e2bface8f345e6641ec9.onnx"
horizontal_model_path = os.path.join(script_location, "staff_relations", horizontal_model_name)


def _center_at(staff: Staff, x: float) -> float:
    """
    The y of the center line at x, extrapolated beyond the ends of the staff along the
    slope of its last 20 units.
    """
    xs = np.array([point.x for point in staff.grid])
    ys = np.array([(point.y[0] + point.y[-1]) / 2 for point in staff.grid])
    if xs[0] <= x <= xs[-1] or len(xs) < 2:  # noqa: PLR2004
        return float(np.interp(x, xs, ys))
    reach = 20 * staff.average_unit_size
    near = xs >= xs[-1] - reach if x > xs[-1] else xs <= xs[0] + reach
    if near.sum() < 2:  # noqa: PLR2004
        near = slice(-2, None) if x > xs[-1] else slice(0, 2)
    slope, intercept = np.polyfit(xs[near], ys[near], 1)
    return float(slope * x + intercept)


def _offset(left: Staff, right: Staff) -> float:
    """Vertical offset of the right staff, both extrapolated to the middle of the gap."""
    middle = (left.max_x + right.min_x) / 2
    return _center_at(right, middle) - _center_at(left, middle)


def is_same_row(left: Staff, right: Staff) -> bool:
    """The right staff starts right of the start of the left one, at about the same height."""
    if right.min_x <= left.min_x or right.max_x <= left.max_x:
        return False
    unit = (left.average_unit_size + right.average_unit_size) / 2
    return abs(_offset(left, right)) < 4 * unit


def horizontal_pairs(staffs: list[Staff]) -> list[tuple[int, int]]:
    """Each staff with the closest staff right of it in the same row."""
    pairs = []
    for i, left in enumerate(staffs):
        right = [j for j, staff in enumerate(staffs) if j != i and is_same_row(left, staff)]
        if right:
            pairs.append((i, min(right, key=lambda j: staffs[j].min_x)))
    return pairs


def horizontal_crop(page: NDArray, scale: float, left: Staff, right: Staff) -> NDArray:
    """
    The end of the left staff, the gap and the start of the right staff (channels, height,
    width). The rows follow the lines of the left staff, then of the right staff and
    blend between them in the gap.
    """
    unit = (left.average_unit_size + right.average_unit_size) / 2
    share = np.arange(horizontal_end_width) / horizontal_end_width
    x_end = left.max_x + (share - 1) * end_units * unit
    x_gap = np.linspace(left.max_x, right.min_x, horizontal_gap_width)
    x_start = right.min_x + share * end_units * unit
    x = np.concatenate([x_end, x_gap, x_start])
    blend = np.concatenate(
        [np.zeros(horizontal_end_width), np.linspace(0, 1, horizontal_gap_width)]
        + [np.ones(horizontal_end_width)]
    )
    tops = (1 - blend) * _line_at(left, 0, x) + blend * _line_at(right, 0, x)
    bottoms = (1 - blend) * _line_at(left, -1, x) + blend * _line_at(right, -1, x)
    anchor_rows = np.cumsum([0, outer_rows, staff_rows, outer_rows])
    anchor_ys = [tops - 3 * unit, tops, bottoms, bottoms + 3 * unit]
    rows = np.empty((horizontal_crop_height, len(x)))
    for i in range(len(anchor_rows) - 1):
        start, end = anchor_rows[i], anchor_rows[i + 1]
        part = ((np.arange(start, end) - start) / (end - start))[:, np.newaxis]
        rows[start:end] = anchor_ys[i] + part * (anchor_ys[i + 1] - anchor_ys[i])
    map_x = np.broadcast_to(x, rows.shape) * scale
    crop = _remap(page, map_x, rows * scale)
    return np.ascontiguousarray(crop.transpose(2, 0, 1))


def horizontal_pair_features(left: Staff, right: Staff) -> NDArray:
    unit = (left.average_unit_size + right.average_unit_size) / 2
    offset = _offset(left, right)
    return np.array(
        [
            offset / unit,
            (right.min_x - left.max_x) / unit / 10,
            np.log(left.average_unit_size / right.average_unit_size),
            (left.max_x - left.min_x) / unit / 100,
            (right.max_x - right.min_x) / unit / 100,
        ],
        dtype=np.float32,
    )


def horizontal_inputs(
    page: NDArray, scale: float, staffs: list[Staff], pairs: list[tuple[int, int]]
) -> tuple[NDArray, NDArray]:
    if len(pairs) == 0:
        shape = (0, channels, horizontal_crop_height, horizontal_crop_width)
        return np.zeros(shape, np.float32), np.zeros((0, horizontal_features), np.float32)
    crops = np.stack([horizontal_crop(page, scale, staffs[i], staffs[j]) for i, j in pairs])
    feats = np.stack([horizontal_pair_features(staffs[i], staffs[j]) for i, j in pairs])
    return crops.astype(np.float32), feats


def join_pieces(left: Staff, right: Staff) -> Staff:
    """One staff from two pieces in a row, where they overlap the left one is used."""
    grid = list(left.grid) + [p for p in right.grid if p.x > left.max_x]
    joined = Staff(grid)
    joined.symbols = left.symbols + right.symbols
    return joined


def join_staff_pieces(
    staffs: list[Staff], pairs: list[tuple[int, int]], probabilities: NDArray
) -> list[Staff]:
    """Joins the pairs which the model sees as pieces of the same staff."""
    right_of = {
        i: j
        for (i, j), p in zip(pairs, probabilities, strict=True)
        if p[SAME_STAFF] > 0.5  # noqa: PLR2004
    }
    used: set[int] = set()
    result = []
    for i in sorted(range(len(staffs)), key=lambda i: staffs[i].min_x):
        if i in used:
            continue
        used.add(i)
        staff = staffs[i]
        j = i
        while right_of.get(j) is not None and right_of[j] not in used:
            j = right_of[j]
            used.add(j)
            staff = join_pieces(staff, staffs[j])
        result.append(staff)
    return sorted(result, key=lambda staff: staff.min_y)
