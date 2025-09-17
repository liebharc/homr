import cv2

from homr.model import Staff
from homr.simple_logging import eprint
from homr.transformer.staff2score import Staff2Score
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray

inference: Staff2Score | None = None


def parse_staff_tromr(staff: Staff, staff_image: NDArray) -> list[EncodedSymbol]:
    return predict_best(staff_image, staff=staff)


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


def predict_best(org_image: NDArray, staff: Staff) -> list[EncodedSymbol]:
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score()

    images = [org_image]
    if len(staff.symbols) > 0:
        images = build_image_options(org_image)
    expected_notes = staff.get_number_of_notes()
    best_distance: float = 0
    best_attempt = 0
    best_result: list[EncodedSymbol] = []
    for attempt, image in enumerate(images):
        result = inference.predict(image)
        actual_notes = len([symbol for symbol in result if "note" in symbol.rhythm])
        distance = abs(expected_notes - actual_notes)
        if len(best_result) == 0 or distance < best_distance:
            best_distance = distance
            best_result = result
            best_attempt = attempt

        results_are_close = 4
        if best_distance < results_are_close:
            eprint(
                "Stopping at attempt", best_attempt + 1, "with distance", best_distance, best_result
            )
            return best_result

    eprint("Taking attempt", best_attempt + 1, "with distance", best_distance, best_result)
    return best_result
