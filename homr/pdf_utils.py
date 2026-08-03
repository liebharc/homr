import os

import cv2
import numpy as np
import pypdfium2 as pdfium

from homr.autocrop import autocrop
from homr.type_definitions import NDArray


def _pad_to_width(page: NDArray, max_width: int) -> NDArray:
    width = page.shape[1]
    assert width <= max_width, "shouldn't be padded"  # noqa: S101
    left = (max_width - width) // 2
    right = max_width - width - left
    return cv2.copyMakeBorder(page, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _vstack_pages(pages: list[NDArray]) -> NDArray:
    max_width = max(page.shape[1] for page in pages)
    return np.vstack([_pad_to_width(page, max_width) for page in pages])


def render_pdf_to_image(pdf_path: str, dpi: int = 300) -> None:
    scale = dpi / 72.0
    pdf = pdfium.PdfDocument(pdf_path)
    assert pdf, f"invalid PDF {pdf_path}"  # noqa: S101
    try:
        pages: list[NDArray] = []
        for page in pdf:
            bitmap = page.render(scale=scale)
            rgb = np.array(bitmap.to_pil().convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pages.append(autocrop(bgr))
    finally:
        pdf.close()
    output_path = os.path.splitext(pdf_path)[0] + ".png"
    cv2.imwrite(output_path, _vstack_pages(pages))
