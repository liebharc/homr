import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from rapidocr_onnxruntime import RapidOCR

from homr.debug import Debug
from homr.model import Staff

# Globals
_reader: RapidOCR | None = None
_reader_lock = threading.Lock()
_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)


def cleanup_text(text: str) -> str:
    """
    Remove all special characters from the text. Merge multiple whitespaces into a single space.
    """
    return re.sub(r"[^a-zA-Z0-9]+", " ", text).strip()


def _initialize_reader() -> None:
    """
    Thread-safe initialization of the OCR reader.
    """
    global _reader  # noqa: PLW0603
    if _reader is None:
        with _reader_lock:
            if _reader is None:  # double-checked locking
                _reader = RapidOCR()


def _detect_title_task(debug: Debug, top_staff: Staff) -> str:
    _initialize_reader()

    if _reader is None:
        raise ValueError("reader is not initialized")
    image = debug.original_image
    height: int = int(15 * top_staff.average_unit_size)
    y: int = max(int(top_staff.min_y) - height, 0)
    x: int = max(int(top_staff.min_x) - 50, 0)
    width: int = int(top_staff.max_x - top_staff.min_x) + 100
    width = min(width, image.shape[1] - x)
    height = min(height, image.shape[0] - y)
    above_staff = image[y : y + height, x : x + width]

    ocr_input: str = debug.write_model_input_image("_tesseract_input.png", above_staff)
    ocr_results = _reader(ocr_input)
    if not ocr_results:
        return ""
    ocr_results = _reader(ocr_input)

    # Each result is (bbox, text, confidence)
    # Pick the text with the largest bbox area
    def bbox_area(bbox: list[list[float]]) -> float:
        # bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    largest_text = max(ocr_results[0], key=lambda r: bbox_area(r[0]))[1]  # get text of largest area
    return cleanup_text(largest_text)


def download_ocr_weights() -> None:
    """
    Pre-download and initialize OCR reader weights before detection runs.
    """
    _initialize_reader()


def detect_title(debug: Debug, top_staff: Staff) -> Future[str]:
    """
    Runs the title detection in a separate thread and returns a Future.
    """
    return _executor.submit(_detect_title_task, debug, top_staff)
