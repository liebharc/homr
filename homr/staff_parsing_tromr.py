import cv2

from homr.debug import AttentionDebug
from homr.transformer.configs import default_config
from homr.transformer.staff2score import Staff2Score
from homr.type_definitions import NDArray

inference: Staff2Score | None = None


def parse_staff_tromr(staff_image: NDArray, debug: AttentionDebug | None) -> str:
    return predict(staff_image, debug=debug)


def apply_clahe(staff_image: NDArray, clip_limit: float = 2.0, kernel_size: int = 8) -> NDArray:
    gray_image = cv2.cvtColor(staff_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(kernel_size, kernel_size))
    gray_image = clahe.apply(gray_image)

    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


def predict(image: NDArray, debug: AttentionDebug | None = None) -> str:
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score(default_config)

    if debug is not None:
        debug.reset()

    result = inference.predict(
        image,
        debug=debug,
    )
    return result
