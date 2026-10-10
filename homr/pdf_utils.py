import os

import cv2
import numpy as np
import pypdfium2 as pdfium

from homr.autocrop import autocrop


def render_pdf_to_image(pdf_path: str, dpi: int = 300) -> list[str]:
    "Renders a pdf to images and returns all paths to them"
    scale = dpi / 72.0
    pdf = pdfium.PdfDocument(pdf_path)
    assert pdf, f"invalid PDF {pdf_path}"  # noqa: S101
    path_prefix = os.path.splitext(pdf_path)[0]
    paths = []
    try:
        for i, page in enumerate(pdf):
            if i == 0:
                # Omit the page suffix because the first rendered image determines the MusicXML name.
                output_path = f"{path_prefix}.png"
            else:
                output_path = f"{path_prefix}_{i}.png"
            bitmap = page.render(scale=scale)
            rgb = np.array(bitmap.to_pil().convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, autocrop(bgr))
            paths.append(output_path)
    finally:
        pdf.close()
    return paths
