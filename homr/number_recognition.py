"""
Reads the numbers of time signatures and of multi-measure rests. The transformer reads
only the beat type of a time signature and can't count the measures of a multi-measure
rest. A small CNN classifies the connected components near these symbols as digits.
"""

import os
import re
from dataclasses import dataclass
from fractions import Fraction

import cv2
import numpy as np

from homr import constants
from homr.model import BarLine, Staff
from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, nonote
from homr.type_definitions import NDArray

DIGITS = [str(d) for d in range(10)]
CLASSES = [*DIGITS, "other"]
OTHER = len(DIGITS)
READ_DENOMINATORS = (2, 4, 8, 16)

# A crop is crop_units x crop_units staff units around a component, scaled to crop_size
crop_size = 40
crop_units = 4.5
# Probability above which a component counts as digit
digit_probability = 0.6

script_location = os.path.dirname(os.path.realpath(__file__))
model_name = "digits_v2.onnx"
model_path = os.path.join(script_location, "number_recognition", model_name)


@dataclass
class StaffLines:
    top: float
    unit: float

    @property
    def bottom(self) -> float:
        return self.top + 4 * self.unit

    @property
    def middle(self) -> float:
        return self.top + 2 * self.unit


@dataclass
class Component:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


def staff_lines(staff: Staff, height: int) -> StaffLines | None:
    """The five lines of the staff closest to the vertical center of the staff image."""
    count = constants.number_of_lines_on_a_staff
    lengths = [len(p.y) for p in staff.grid]
    length = max(set(lengths), key=lengths.count)
    ys = np.median([p.y for p in staff.grid if len(p.y) == length], axis=0)
    groups = [ys[i : i + count] for i in range(0, length, count)]
    group = min(groups, key=lambda g: abs(g.mean() - height / 2))
    return StaffLines(float(group[0]), float(np.median(np.diff(group))))


def _binarize(gray: NDArray) -> NDArray:
    """Ink as 1, background as 0."""
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary.astype(np.uint8)


def _remove_staff_lines(binary: NDArray, lines: StaffLines) -> NDArray:
    """Removes the pixels of the staff lines which aren't part of a symbol crossing them."""
    result = binary.copy()
    thickness = max(1, round(lines.unit * 0.25))
    height = binary.shape[0]
    for i in range(5):
        y = round(lines.top + i * lines.unit)
        top = max(0, y - thickness)
        bottom = min(height, y + thickness + 1)
        above = binary[max(0, top - 1)] if top > 0 else np.zeros(binary.shape[1], np.uint8)
        below = binary[bottom] if bottom < height else np.zeros(binary.shape[1], np.uint8)
        # A column of the band is only the line if there is no ink just above and below
        is_line = (above == 0) & (below == 0)
        result[top:bottom, is_line] = 0
    return result


def _components(binary: NDArray) -> list[Component]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [
        Component(int(s[0]), int(s[1]), int(s[2]), int(s[3])) for s in stats[1:count] if s[4] > 1
    ]


def crop_around(gray: NDArray, center: tuple[float, float], unit: float) -> NDArray:
    """A crop_size x crop_size crop of crop_units staff units, white outside of the image."""
    scale = crop_size / (crop_units * unit)
    matrix = np.array(
        [
            [scale, 0, crop_size / 2 - center[0] * scale],
            [0, scale, crop_size / 2 - center[1] * scale],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        gray,
        matrix,
        (crop_size, crop_size),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


class DigitClassifier:
    def __init__(self, path: str = model_path) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def predict(self, crops: list[NDArray]) -> NDArray:
        """Class probabilities for each crop (crops x classes)."""
        if not crops:
            return np.zeros((0, len(CLASSES)), dtype=np.float32)
        batch = np.stack(crops).astype(np.float32)[:, None] / 255
        logits = self.session.run(None, {"crops": batch})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


_classifier: DigitClassifier | None = None


def _get_classifier() -> DigitClassifier | None:
    global _classifier  # noqa: PLW0603
    if _classifier is None:
        if not os.path.exists(model_path):
            eprint("Digit model is missing, can't read numbers", model_path)
            return None
        _classifier = DigitClassifier()
    return _classifier


def _merge_overlapping(components: list[Component], max_gap: float) -> list[Component]:
    """
    Merges components which overlap horizontally and are at most max_gap apart vertically,
    e.g. the parts of a digit which a staff line or a faint print split.
    """
    result = sorted(components, key=lambda c: c.x)
    merged = True
    while merged:
        merged = False
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                a, b = result[i], result[j]
                overlap = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
                gap = max(a.y, b.y) - min(a.y + a.h, b.y + b.h)
                if overlap > 0.5 * min(a.w, b.w) and gap <= max_gap:
                    x0, y0 = min(a.x, b.x), min(a.y, b.y)
                    x1, y1 = max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)
                    result[i] = Component(x0, y0, x1 - x0, y1 - y0)
                    del result[j]
                    merged = True
                    break
            if merged:
                break
    return result


def _read_number(
    gray: NDArray,
    components: list[Component],
    unit: float,
    classifier: DigitClassifier,
    max_gap: float,
    center_x: float,
    max_offset: float,
) -> tuple[int, list[Component]] | None:
    """
    Classifies the components and returns the number whose digits are closest to center_x
    and its digits, or None if no number is within max_offset of it.
    """
    if not components:
        return None
    crops = [crop_around(gray, c.center, unit) for c in components]
    probabilities = classifier.predict(crops)
    digits = [
        (component, int(np.argmax(p)))
        for component, p in zip(components, probabilities, strict=True)
        if np.argmax(p) != OTHER and np.max(p) >= digit_probability
    ]
    if not digits:
        return None
    digits.sort(key=lambda d: d[0].x)
    # Digits which belong to one number follow each other closely
    groups: list[list[tuple[Component, int]]] = [[digits[0]]]
    for digit in digits[1:]:
        last = groups[-1][-1][0]
        if digit[0].x - (last.x + last.w) <= max_gap * unit:
            groups[-1].append(digit)
        else:
            groups.append([digit])

    def distance(group: list[tuple[Component, int]]) -> float:
        left = group[0][0].x
        right = group[-1][0].x + group[-1][0].w
        return abs((left + right) / 2 - center_x)

    best = min(groups, key=distance)
    if distance(best) > max_offset:
        return None
    return int("".join(str(d[1]) for d in best)), [d[0] for d in best]


@dataclass
class TimeSignatureReading:
    numerator: int | None
    denominator: int | None


def read_time_signature(
    gray: NDArray, lines: StaffLines, x: float, classifier: DigitClassifier
) -> TimeSignatureReading:
    """Reads the digits in the upper and lower half of the staff around x."""
    binary = _remove_staff_lines(_binarize(gray), lines)
    unit = lines.unit
    # The digits often touch at the middle line
    cut = max(1, round(0.12 * unit))
    binary[round(lines.middle) - cut : round(lines.middle) + cut + 1] = 0
    left = max(0, int(x - 3 * unit))
    right = min(gray.shape[1], int(x + 3 * unit))

    components = _components(binary[:, left:right])

    def half(top: float, bottom: float) -> int | None:
        inside = [
            c for c in components if top - 0.6 * unit <= c.y and c.y + c.h <= bottom + 0.6 * unit
        ]
        candidates = [
            Component(c.x + left, c.y, c.w, c.h)
            for c in _merge_overlapping(inside, max_gap=0.4 * unit)
            if 1.2 * unit <= c.h <= 2.8 * unit and c.w <= 2.5 * unit  # noqa: PLR2004
        ]
        result = _read_number(
            gray, candidates, unit, classifier, max_gap=0.8, center_x=x, max_offset=3 * unit
        )
        return result[0] if result else None

    return TimeSignatureReading(half(lines.top, lines.middle), half(lines.middle, lines.bottom))


def read_number_above(
    gray: NDArray, lines: StaffLines, left: float, right: float, classifier: DigitClassifier
) -> int | None:
    """Reads a number above the staff between left and right, e.g. of a multi-measure rest."""
    unit = lines.unit
    top = max(0, int(lines.top - 6 * unit))
    bottom = int(lines.top - 0.2 * unit)
    x0 = max(0, int(left))
    x1 = min(gray.shape[1], int(right))
    if bottom <= top or x1 <= x0:
        return None
    binary = _binarize(gray)[top:bottom]
    components = [
        Component(c.x, c.y + top, c.w, c.h)
        for c in _merge_overlapping(_components(binary), max_gap=0.25 * unit)
    ]
    candidates = [
        c
        for c in components
        if x0 <= c.x and c.x + c.w <= x1 and 1.0 * unit <= c.h <= 3.5 * unit and c.w <= 3 * unit
    ]
    # The number of a multi-measure rest is centered over the measure, unlike e.g. a
    # measure number at the start of a line
    result = _read_number(
        gray,
        candidates,
        unit,
        classifier,
        max_gap=1.0,
        center_x=(left + right) / 2,
        max_offset=0.35 * (right - left),
    )
    if result is None:
        return None
    number, digits = result
    # It stands alone, unlike e.g. the number of a tempo mark like "♩ = 50"
    group_left = min(d.x for d in digits)
    group_right = max(d.x + d.w for d in digits)
    group_top = min(d.y for d in digits)
    group_bottom = max(d.y + d.h for d in digits)
    for c in components:
        if c in digits or max(c.w, c.h) < 0.4 * unit:  # noqa: PLR2004
            continue
        overlaps_vertically = c.y < group_bottom and c.y + c.h > group_top
        gap = max(group_left - (c.x + c.w), c.x - group_right)
        if overlaps_vertically and gap < 1.0 * unit:
            return None
    return number


_BARLINE = re.compile(
    r"(barline|doublebarline|bolddoublebarline|repeatStart|repeatEnd|repeatEndStart)$"
)


def _is_barline(symbol: EncodedSymbol) -> bool:
    return bool(_BARLINE.match(symbol.rhythm))


def _is_timed(symbol: EncodedSymbol) -> bool:
    return symbol.rhythm.startswith(("note", "rest")) and "G" not in symbol.rhythm


def _measures(symbols: list[EncodedSymbol]) -> list[list[int]]:
    """Indices of the symbols per measure, symbols before the first note or rest excluded."""
    measures: list[list[int]] = [[]]
    for i, symbol in enumerate(symbols):
        if _is_barline(symbol):
            if any(_is_timed(symbols[j]) for j in measures[-1]):
                measures.append([])
        else:
            measures[-1].append(i)
    return [m for m in measures if any(_is_timed(symbols[j]) for j in m)]


def _symbol_x(symbol: EncodedSymbol) -> float | None:
    if symbol.coordinates is None:
        return None
    center = np.asarray(symbol.coordinates, dtype=np.float64).reshape(-1)
    if len(center) < 2 or np.isnan(center[0]):  # noqa: PLR2004
        return None
    return float(center[0])


def _staff_extent(gray: NDArray, lines: StaffLines) -> tuple[float, float]:
    """The first and last x at which most staff lines have ink."""
    binary = _binarize(gray)
    rows = [round(lines.top + i * lines.unit) for i in range(5)]
    has_lines = (binary[rows].sum(axis=0) >= 3).astype(np.uint8)  # noqa: PLR2004
    # Ignore short gaps, e.g. where a symbol makes the image lighter
    kernel = np.ones(max(1, round(lines.unit)), np.uint8)
    has_lines = cv2.morphologyEx(has_lines[None], cv2.MORPH_CLOSE, kernel[None])[0]
    xs = np.where(has_lines)[0]
    if len(xs) == 0:
        return 0.0, float(gray.shape[1])
    return float(xs[0]), float(xs[-1])


def _intervals(
    barlines: list[float], lines: StaffLines, extent: tuple[float, float]
) -> list[tuple[float, float]]:
    start, end = extent
    # The first barline may be the line at the start of the system
    bounds = [b for b in barlines if start + 4 * lines.unit < b < end - 2 * lines.unit]
    intervals = list(zip([start, *bounds], [*bounds, end], strict=True))
    return [(a, b) for a, b in intervals if b - a > 2 * lines.unit]


def measure_intervals(
    gray: NDArray,
    lines: StaffLines,
    barlines: list[float],
    measure_count: int,
    starts_with_barline: bool,
) -> tuple[list[tuple[float, float]], bool]:
    """
    The x ranges of the measures and whether they match the measures of the transformer.
    If the staff starts with a barline after the clef, e.g. a repeat, the range before
    it holds no measure.
    """
    intervals = _intervals(sorted(barlines), lines, _staff_extent(gray, lines))
    if starts_with_barline:
        intervals = intervals[1:]
    return intervals, len(intervals) == measure_count


def _measure_bounds(
    measure_no: int,
    x: float | None,
    intervals: list[tuple[float, float]],
    matches_measures: bool,
) -> tuple[float, float] | None:
    """The x range of a measure, by its number or else by the position of its rest."""
    if matches_measures:
        return intervals[measure_no]
    if x is None:
        return None
    for a, b in intervals:
        if a <= x <= b:
            return a, b
    return None


def _measure_length(symbols: list[EncodedSymbol], measure: list[int]) -> Fraction:
    return sum(
        (symbols[i].get_duration().fraction for i in measure if _is_timed(symbols[i])),
        Fraction(0),
    )


def _is_multirest(rhythm: str) -> bool:
    return rhythm.startswith("rest_") and rhythm.endswith("m")


def _full_measure_rest(
    symbols: list[EncodedSymbol], measure: list[int], typical_length: Fraction | None
) -> int | None:
    """The index of the rest if it's the only symbol of the measure and fills it."""
    timed = [i for i in measure if _is_timed(symbols[i])]
    has_chord = any(symbols[i].rhythm == "chord" for i in measure)
    if len(timed) != 1 or has_chord or not symbols[timed[0]].rhythm.startswith("rest"):
        return None
    rest = symbols[timed[0]]
    if _is_multirest(rest.rhythm):
        return timed[0]
    # Excludes e.g. a short rest before a repeat sign which splits a measure
    duration = rest.get_duration().fraction
    if duration >= Fraction(1, 2) or (typical_length is not None and duration >= typical_length):
        return timed[0]
    return None


def _starts_with_barline(symbols: list[EncodedSymbol]) -> bool:
    for symbol in symbols:
        if _is_timed(symbol):
            return False
        if _is_barline(symbol):
            return True
    return False


def read_numbers(gray: NDArray, symbols: list[EncodedSymbol], staff: Staff) -> list[EncodedSymbol]:
    """
    Adds the numerator to time signatures, e.g. timeSignature/4 becomes timeSignature3/4,
    and turns a rest with a number above it into a multi-measure rest, e.g. rest_14m.
    The staff and its bar lines must be in the coordinates of the staff image. Rests of a
    grand staff aren't read, the time signature is the same in both staffs.
    """
    if gray.ndim == 3:  # noqa: PLR2004
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    lines = staff_lines(staff, gray.shape[0])
    if lines is None:
        return symbols
    classifier = _get_classifier()
    if classifier is None:
        return symbols
    for symbol in symbols:
        if not symbol.rhythm.startswith("timeSignature/"):
            continue
        x = _symbol_x(symbol)
        if x is None:
            continue
        reading = read_time_signature(gray, lines, x, classifier)
        denominator = symbol.rhythm.split("/")[1]
        # Both digits must be read, the digits win over the transformer's beat type
        if reading.numerator and reading.denominator in READ_DENOMINATORS:
            if str(reading.denominator) != denominator:
                eprint("Beat type", denominator, "replaced by", reading)
            symbol.rhythm = f"timeSignature{reading.numerator}/{reading.denominator}"
        else:
            eprint("Time signature not read", reading, "beat type", denominator)

    if staff.is_grandstaff:
        return symbols
    measures = _measures(symbols)
    barlines = [s.center[0] for s in staff.symbols if isinstance(s, BarLine)]
    intervals, matches_measures = measure_intervals(
        gray, lines, barlines, len(measures), _starts_with_barline(symbols)
    )
    lengths = [
        _measure_length(symbols, m) for m in measures if sum(_is_timed(symbols[i]) for i in m) > 1
    ]
    typical_length = sorted(lengths)[len(lengths) // 2] if lengths else None
    rests = []
    for measure_no, measure in enumerate(measures):
        rest_index = _full_measure_rest(symbols, measure, typical_length)
        if rest_index is None:
            continue
        rest = symbols[rest_index]
        bounds = _measure_bounds(measure_no, _symbol_x(rest), intervals, matches_measures)
        rests.append((measure_no, rest, bounds))
    # The attention of rests in a row is imprecise, if two point to the same measure
    # neither is reliable, else both would get its number
    all_bounds = [bounds for _, _, bounds in rests]
    for measure_no, rest, bounds in rests:
        shared = not matches_measures and all_bounds.count(bounds) > 1
        number = None
        if bounds is not None and not shared:
            number = read_number_above(gray, lines, bounds[0], bounds[1], classifier)
        if number is None or number < 1:
            # The transformer only tells that it's a multirest, not for how many measures
            if _is_multirest(rest.rhythm):
                eprint("Multirest without number in measure", measure_no)
                number = 1
            else:
                continue
        eprint("Rest with number", number, "in measure", measure_no)
        if number == 1:
            rest.rhythm = "measureRest"
            rest.pitch = rest.lift = rest.articulation = rest.slur = nonote
        else:
            rest.rhythm = f"rest_{number}m"
    return symbols
