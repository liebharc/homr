# flake8: noqa: S101

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
from rapidocr import RapidOCR

from homr.debug import Debug
from homr.model import Staff
from homr.simple_logging import eprint

# Globals
_reader: RapidOCR | None = None
_executor = ThreadPoolExecutor(max_workers=1)


class TextBox:
    def __init__(self, text: str, box: np.ndarray):
        # RapidOCR boxes are arrays consisting of 4 rows and 2 cols,
        # representing 4 vertex coordinates: top-left, top-right, bottom-right, bottom-left
        assert box.shape == (4, 2), "invalid ocr result"

        self.text = text
        self.left = box[0, 0]
        self.top = box[0, 1]
        self.right = box[2, 0]
        self.bottom = box[2, 1]

        self.cal_property()

    def cal_property(self) -> None:
        self.height = self.bottom - self.top
        self.width = self.right - self.left
        self.center = (self.left + self.right) / 2

    def merge(self, other: "TextBox") -> None:
        self.text = self.text + " " + other.text
        self.right = max(self.right, other.right)
        self.left = min(self.left, other.left)
        self.top = min(self.top, other.top)
        self.bottom = max(self.bottom, other.bottom)

        self.cal_property()

    def cal_score(self, page_center: int, largest_height: float) -> None:
        # Score each box; higher scores are more likely to be the title.
        # How to score:
        # prefer boxes near the page center (70% weight), then taller boxes (30% weight).
        centered = 1 - abs(self.center - page_center) / page_center
        height = self.height / largest_height
        self.score = height * 0.3 + centered * 0.7

    def print_score(self) -> None:
        eprint("Title candidate:", repr(self.text), "score:", round(self.score, 3))


def _merge_texts(texts: list[TextBox]) -> list[TextBox]:
    # In practice, RapidOCR sometimes splits title into two boxes, so they need to be merged.
    merged: list[TextBox] = []
    # After sorting, boxes are traversed left-to-right, then top-to-bottom.
    for current in sorted(texts, key=lambda item: (item.left, item.top)):
        for previous in merged:
            x_distance = current.left - previous.right
            y_overlap = min(previous.bottom, current.bottom) - max(current.top, previous.top)
            min_height = min(current.height, previous.height)
            # Merge when vertical overlap is large enough (>50%) and
            # horizontal gap is small enough (<50%).
            if y_overlap > 0.5 * min_height and x_distance < 0.5 * min_height:
                previous.merge(current)
                break
        else:
            merged.append(current)
    return merged


def _detect_title_task(debug: Debug, top_staff: Staff) -> str:
    assert _reader, "reader is not initialized"

    image = debug.original_image
    above_staff = image[: int(top_staff.min_y), :]
    ocr_results = _reader(above_staff)
    if not ocr_results:
        return ""

    texts = [
        TextBox(text, box) for box, text in zip(ocr_results.boxes, ocr_results.txts, strict=True)
    ]
    texts = _merge_texts(texts)

    largest = max(text.height for text in texts)
    page_center = image.shape[1] // 2
    for text in texts:
        text.cal_score(page_center, largest)

    ranked = sorted(
        texts,
        key=lambda text: text.score,
        reverse=True,
    )
    if debug.debug:
        for text in ranked:
            text.print_score()
    return ranked[0].text


def download_ocr_weights() -> None:
    """
    Pre-download and initialize OCR reader weights before detection runs.
    """
    global _reader  # noqa: PLW0603
    assert _reader is None
    _reader = RapidOCR()


def detect_title(debug: Debug, top_staff: Staff) -> Future[str]:
    """
    Runs the title detection in a separate thread and returns a Future.
    """
    return _executor.submit(_detect_title_task, debug, top_staff)
