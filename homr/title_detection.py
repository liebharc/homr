import re

import easyocr  # type: ignore

from homr.debug import Debug
from homr.model import Staff

reader = easyocr.Reader(["de", "en"], gpu=False, verbose=False)


def cleanup_text(text: str) -> str:
    """
    Remove all special characters from the text. Merge multiple whitespaces into a single space.
    """
    return re.sub(r"[^a-zA-Z0-9]+", " ", text).strip()


def detect_title(debug: Debug, top_staff: Staff) -> str:
    image = debug.original_image
    height = int(15 * top_staff.average_unit_size)
    y = max(int(top_staff.min_y) - height, 0)
    x = max(int(top_staff.min_x) - 50, 0)
    width = int(top_staff.max_x - top_staff.min_x) + 100
    width = min(width, image.shape[1] - x)
    height = min(height, image.shape[0] - y)
    above_staff = image[y : y + height, x : x + width]

    tesseract_input = debug.write_model_input_image("_tesseract_input.png", above_staff)
    result = reader.readtext(tesseract_input, detail=0, paragraph=True)
    if len(result) == 0:
        return ""
    return cleanup_text(result[0])
