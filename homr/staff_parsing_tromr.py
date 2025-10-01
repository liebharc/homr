import cv2

from homr.model import Staff
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


def augment_staff_image(staff_image: NDArray) -> NDArray:
    denoised1 = cv2.fastNlMeansDenoisingColored(staff_image, None, 10, 10, 7, 21)
    return apply_clahe(denoised1)


def predict_best(org_image: NDArray, staff: Staff) -> list[EncodedSymbol]:
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score()

    image = augment_staff_image(org_image)
    return inference.predict(image)
