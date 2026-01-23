import cv2

from homr.type_definitions import NDArray


def apply_clahe(image: NDArray) -> NDArray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    return clahe.apply(image)
