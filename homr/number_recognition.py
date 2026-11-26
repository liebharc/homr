"""
Reads the numbers of time signatures and of multi-measure rests. The transformer reads
only the beat type of a time signature and can't count the measures of a multi-measure
rest. A small CNN reads a crop of the staff around the symbol: the numerator and the
denominator of a time signature, and the number above a multi-measure rest.
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

# The crop in staff units: CROP_HALF_WIDTH left and right of the symbol, CROP_ABOVE above
# the top line and CROP_BELOW below the bottom line, at CROP_PX pixels per unit
CROP_PX = 8
CROP_HALF_WIDTH = 3
CROP_ABOVE = 5
CROP_BELOW = 1
CROP_WIDTH = 2 * CROP_HALF_WIDTH * CROP_PX
CROP_HEIGHT = (CROP_ABOVE + 4 + CROP_BELOW) * CROP_PX

# Outputs: numbers as digit slots, right aligned, class 0 is an empty slot
NUMERATOR_DIGITS = 2
COUNT_DIGITS = 3
DIGIT_CLASSES = 11
DENOMINATORS = [0, 1, 2, 4, 8, 16, 32]  # 0: no time signature
READ_DENOMINATORS = (2, 4, 8, 16)
# Probability above which a reading is used
min_probability = 0.6

script_location = os.path.dirname(os.path.realpath(__file__))
model_name = "numbers_v1.onnx"
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


def crop_at(gray: NDArray, x: float, lines: StaffLines) -> NDArray:
    """The model input around x, white outside of the image."""
    scale = CROP_PX / lines.unit
    matrix = np.array(
        [
            [scale, 0, CROP_WIDTH / 2 - x * scale],
            [0, scale, CROP_ABOVE * CROP_PX - lines.top * scale],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        gray,
        matrix,
        (CROP_WIDTH, CROP_HEIGHT),
        flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def decode_digits(probabilities: NDArray) -> tuple[int | None, float]:
    """
    The number of right aligned digit slots (slots x DIGIT_CLASSES) and its probability.
    None if there is no number or the slots don't form one.
    """
    classes = probabilities.argmax(axis=1)
    probability = float(np.prod(probabilities.max(axis=1)))
    digits = [int(c) - 1 for c in classes if c > 0]
    leading_empty = int(np.argmax(classes > 0)) if digits else len(classes)
    # Digits must follow each other without empty slots and the first must not be a 0
    if not digits or len(digits) != len(classes) - leading_empty or digits[0] == 0:
        return None, probability
    return int("".join(str(d) for d in digits)), probability


@dataclass
class Reading:
    numerator: int | None
    denominator: int | None
    count: int | None
    probability_time_signature: float
    probability_count: float


class NumberReader:
    def __init__(self, path: str = model_path) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def read(self, crops: list[NDArray]) -> list[Reading]:
        if not crops:
            return []
        batch = np.stack(crops).astype(np.float32)[:, None] / 255
        numerators, denominators, counts = self.session.run(None, {"crops": batch})
        result = []
        for numerator_logits, denominator_logits, count_logits in zip(
            numerators, denominators, counts, strict=True
        ):
            numerator, p_numerator = decode_digits(_softmax(numerator_logits))
            p_denominators = _softmax(denominator_logits[None])[0]
            denominator = DENOMINATORS[int(p_denominators.argmax())] or None
            count, p_count = decode_digits(_softmax(count_logits))
            result.append(
                Reading(
                    numerator,
                    denominator,
                    count,
                    p_numerator * float(p_denominators.max()),
                    p_count,
                )
            )
        return result


def _softmax(logits: NDArray) -> NDArray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


_reader: NumberReader | None = None


def _get_reader() -> NumberReader | None:
    global _reader  # noqa: PLW0603
    if _reader is None:
        if not os.path.exists(model_path):
            eprint("Number model is missing, can't read numbers", model_path)
            return None
        _reader = NumberReader()
    return _reader


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
    """
    The index of the longest rest if the measure has only rests and that rest fills it.
    The transformer often reads the bar of a multirest as several rests.
    """
    timed = [i for i in measure if _is_timed(symbols[i])]
    if not timed or not all(symbols[i].rhythm.startswith("rest") for i in timed):
        return None
    multirests = [i for i in timed if _is_multirest(symbols[i].rhythm)]
    if multirests:
        return multirests[0]
    longest = max(timed, key=lambda i: symbols[i].get_duration().fraction)
    rest = symbols[longest]
    # Excludes e.g. a short rest before a repeat sign which splits a measure
    duration = rest.get_duration().fraction
    if duration >= Fraction(1, 2) or (typical_length is not None and duration >= typical_length):
        return longest
    return None


def _content_start(
    symbols: list[EncodedSymbol], measure: list[int], start: float, lines: StaffLines
) -> float:
    """The start of a measure after its clef, key and time signature."""
    xs = [
        x
        for i in measure
        if symbols[i].rhythm.startswith(("clef", "keySignature", "timeSignature"))
        for x in [_symbol_x(symbols[i])]
        if x is not None and x > start
    ]
    return max(xs) + 1.5 * lines.unit if xs else start


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
    reader = _get_reader()
    if reader is None:
        return symbols
    time_signatures = [
        (symbol, x)
        for symbol in symbols
        if symbol.rhythm.startswith("timeSignature/")
        for x in [_symbol_x(symbol)]
        if x is not None
    ]
    readings = reader.read([crop_at(gray, x, lines) for _, x in time_signatures])
    for (symbol, _), reading in zip(time_signatures, readings, strict=True):
        denominator = symbol.rhythm.split("/")[1]
        # Both numbers must be read, they win over the transformer's beat type
        if (
            reading.numerator
            and reading.denominator in READ_DENOMINATORS
            and reading.probability_time_signature >= min_probability
        ):
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
    others: dict[int, list[int]] = {}
    for measure_no, measure in enumerate(measures):
        rest_index = _full_measure_rest(symbols, measure, typical_length)
        if rest_index is None:
            continue
        rest = symbols[rest_index]
        bounds = _measure_bounds(measure_no, _symbol_x(rest), intervals, matches_measures)
        if bounds is not None:
            bounds = (_content_start(symbols, measure, bounds[0], lines), bounds[1])
        rests.append((measure_no, rest, bounds))
        others[id(rest)] = [
            i
            for i in measure
            if i != rest_index and symbols[i].rhythm.startswith(("rest", "chord"))
        ]
    # The attention of rests in a row is imprecise, if two point to the same measure
    # neither is reliable, else both would get its number
    all_bounds = [bounds for _, _, bounds in rests]
    readable = [
        (measure_no, rest, bounds)
        for measure_no, rest, bounds in rests
        if bounds is not None and (matches_measures or all_bounds.count(bounds) == 1)
    ]
    # The number of a multirest is centered over the measure
    readings = reader.read([crop_at(gray, (b[0] + b[1]) / 2, lines) for _, _, b in readable])
    numbers = {
        id(rest): reading.count
        for (_, rest, _), reading in zip(readable, readings, strict=True)
        if reading.probability_count >= min_probability
    }
    dropped: set[int] = set()
    for measure_no, rest, _ in rests:
        number = numbers.get(id(rest))
        if number is None or number < 1:
            # The transformer only tells that it's a multirest, not for how many measures
            if _is_multirest(rest.rhythm):
                eprint("Multirest without number in measure", measure_no)
                number = 1
            else:
                continue
        eprint("Rest with number", number, "in measure", measure_no)
        # The other rests of the measure were misread parts of the multirest
        dropped.update(others[id(rest)])
        if number == 1:
            rest.rhythm = "measureRest"
            rest.pitch = rest.lift = rest.articulation = rest.slur = nonote
        else:
            rest.rhythm = f"rest_{number}m"
    return [symbol for i, symbol in enumerate(symbols) if i not in dropped]
