import re
from collections import Counter

import cv2
import numpy as np

from homr import constants
from homr.debug import AttentionDebug
from homr.model import Staff
from homr.results import ClefType, ResultStaff, ResultTimeSignature
from homr.simple_logging import eprint
from homr.tr_omr_parser import TrOMRParser
from homr.transformer.configs import default_config
from homr.transformer.staff2score import Staff2Score
from homr.type_definitions import NDArray

inference: Staff2Score | None = None


def parse_staff_tromr(
    staff: Staff, staff_image: NDArray, debug: AttentionDebug | None
) -> ResultStaff | None:
    return predict_best(staff_image, debug=debug, staff=staff)


def apply_clahe(staff_image: NDArray, clip_limit: float = 2.0, kernel_size: int = 8) -> NDArray:
    gray_image = cv2.cvtColor(staff_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(kernel_size, kernel_size))
    gray_image = clahe.apply(gray_image)

    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


def build_image_options(staff_image: NDArray) -> list[NDArray]:
    denoised1 = cv2.fastNlMeansDenoisingColored(staff_image, None, 10, 10, 7, 21)
    return [
        staff_image,
        denoised1,
        apply_clahe(denoised1),
    ]


def _fill_in_time_signature(staff: ResultStaff) -> None:
    average_measure_length = np.median([m.length_in_quarters() for m in staff.measures])
    for symbol in staff.get_symbols():
        if isinstance(symbol, ResultTimeSignature):
            beat_duration = 4 / symbol.denominator * constants.duration_of_quarter
            symbol.numerator = round(average_measure_length / beat_duration)


def predict_best(
    org_image: NDArray, staff: Staff, debug: AttentionDebug | None = None
) -> ResultStaff | None:
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score(default_config)
    images = build_image_options(org_image)
    notes = staff.get_notes_and_groups()
    best_distance: float = 0
    best_attempt = 0
    best_result: ResultStaff = ResultStaff([])
    for attempt, image in enumerate(images):
        if debug is not None:
            debug.reset()

        result = inference.predict(
            image,
            debug=debug,
        )
        parser = TrOMRParser()
        result_staff = parser.parse_tr_omr_output(str.join("", result))

        clef_type = _get_clef_type(result[0])
        if clef_type is None:
            # Returning early is no clef is found is not optimal,
            # but it makes sure that we get a result and it's a corner case,
            # which is not worth the effort to handle right now.
            eprint("Failed to find clef type in", result)
            return result_staff
        actual = [symbol for symbol in result[0].split("+") if symbol.startswith("note")]
        expected = [note.to_tr_omr_note(clef_type) for note in notes]
        actual = _flatten_result(actual)
        expected = _flatten_result(expected)
        distance = _differences(actual, expected)
        diff_accidentals = abs(
            _number_of_accidentals_in_model(staff) - parser.number_of_accidentals()
        )
        measure_length_variance = _measure_length_variance(result_staff)
        number_of_structural_elements = (
            _superfluous_number(parser.number_of_clefs())
            + _superfluous_number(parser.number_of_key_signatures())
            + _superfluous_number(parser.number_of_time_signatures())
        )
        total_rating = (
            distance + diff_accidentals + measure_length_variance + number_of_structural_elements
        ) / max(min(len(expected), len(actual)), 1)

        if best_result.is_empty() or total_rating < best_distance:
            best_distance = total_rating
            best_result = result_staff
            best_attempt = attempt

    _fill_in_time_signature(best_result)
    eprint("Taking attempt", best_attempt + 1, "with distance", best_distance, best_result)
    return best_result


def _superfluous_number(count: int) -> int:
    """
    Assumes that the item should be present at most once.
    """
    return count - 1 if count > 1 else 0


def _number_of_accidentals_in_model(staff: Staff) -> int:
    return len(staff.get_accidentals())


def _get_clef_type(result: str) -> ClefType | None:
    match = re.search(r"clef-([A-G])([0-9])", result)
    if match is None:
        return None
    return ClefType(match.group(1), int(match.group(2)))


def _flatten_result(result: list[str]) -> list[str]:
    notes = []
    for group in result:
        for symbol in group.split("|"):
            just_pitch = symbol.split("_")[0]
            just_pitch = just_pitch.replace("#", "").replace("b", "")
            notes.append(just_pitch)
    return notes


def _measure_length_variance(result: ResultStaff) -> float:
    durations = [m.length_in_quarters() for m in result.measures]
    return float(np.std(durations - np.mean(durations)))  # type: ignore


def _differences(actual: list[str], expected: list[str]) -> int:
    counter1 = Counter(actual)
    counter2 = Counter(expected)
    return sum((counter1 - counter2).values()) + sum((counter2 - counter1).values())
