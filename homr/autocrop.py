import cv2
import numpy as np

from homr.type_definitions import NDArray


def autocrop(img: NDArray) -> NDArray:
    """
    Find the largest contour on the image, which is expected to be the paper of sheet music
    and extracts it from the image. If no contour is found, then the image is assumed to be
    a full page view of sheet music and is returned as is. Areas beside the paper which
    have another color than the paper, e.g. a desk, are painted in the paper color.
    """
    result, _top_left = autocrop_with_offset(img)
    return result


# Light and shadows change the brightness of paper but hardly its color.
# Mixed daylight and lamp light change it at most by this much.
max_chromaticity_difference = 0.07


def _chromaticity(pixels: NDArray) -> NDArray:
    """Color without brightness: every channel divided by the sum of the channels."""
    values = pixels.astype(np.float32) + 1
    return values / values.sum(axis=-1, keepdims=True)


def _is_paper(pixels: NDArray, paper_chromaticity: NDArray) -> bool:
    if pixels.size == 0:
        return True
    chromaticity = np.median(_chromaticity(pixels.reshape(-1, 3)), axis=0)
    return bool(np.abs(chromaticity - paper_chromaticity).sum() <= max_chromaticity_difference)


def _keep_sides_with_paper(
    img: NDArray, box: tuple[int, int, int, int], paper_chromaticity: NDArray
) -> tuple[int, int, int, int]:
    """
    Extends the box to the image border on every side where the strip beside it is
    paper, e.g. in a shadow.
    """
    x, y, w, h = box
    height, width = img.shape[:2]
    # Only the strips beside the box, the corners can contain e.g. a desk at another side
    keep_left = _is_paper(img[y : y + h, :x], paper_chromaticity)
    keep_right = _is_paper(img[y : y + h, x + w :], paper_chromaticity)
    keep_top = _is_paper(img[:y, x : x + w], paper_chromaticity)
    keep_bottom = _is_paper(img[y + h :, x : x + w], paper_chromaticity)
    left = 0 if keep_left else x
    top = 0 if keep_top else y
    right = width if keep_right else x + w
    bottom = height if keep_bottom else y + h
    return left, top, right - left, bottom - top


def _remove_surroundings(crop: NDArray, sheet: NDArray, paper: NDArray) -> NDArray:
    """
    Removes what is beside the sheet, e.g. the corners of a desk beside a rotated
    sheet, by filling it with the color of the paper. The sheet itself is unchanged.
    """
    beside_sheet = np.full(crop.shape[:2], 1, dtype=np.uint8)
    cv2.fillPoly(beside_sheet, [cv2.convexHull(sheet)], 0)
    paper_color = np.median(paper, axis=0)
    paper_chromaticity = np.median(_chromaticity(paper), axis=0)
    # Single pixels are too noisy to compare their color
    smooth = cv2.blur(crop, (7, 7))
    difference = np.abs(_chromaticity(smooth) - paper_chromaticity).sum(axis=2)
    # The color of dark pixels is noise, e.g. ink in a shadow must stay
    bright = smooth.mean(axis=2) > 0.25 * paper_color.mean()
    background = beside_sheet & bright & (difference > max_chromaticity_difference)
    # Also the pixels at the edge of the paper, which mix paper and background
    background = cv2.dilate(background.astype(np.uint8), np.ones((5, 5), np.uint8))
    result = crop.copy()
    result[(background & beside_sheet) > 0] = paper_color.astype(np.uint8)
    return result


def autocrop_with_offset(img: NDArray) -> tuple[NDArray, tuple[int, int]]:
    """
    Same as autocrop, but also returns the top left corner of the crop in img.
    """
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    dominant_color_gray_scale = int(np.argmax(hist.flatten()))

    # threshold
    thresh = cv2.threshold(gray, dominant_color_gray_scale - 30, 255, cv2.THRESH_BINARY)[1]

    # apply morphology
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((9, 9), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel)

    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    only_one_item_on_background = 2
    contours = contours[0] if len(contours) == only_one_item_on_background else contours[1]  # type: ignore
    area_thresh = 0.0
    big_contour = None
    for c in contours:
        area = cv2.contourArea(c)  # type: ignore
        if area > area_thresh:
            area_thresh = area
            big_contour = c

    if big_contour is None:
        return img, (0, 0)

    # get bounding box
    x, y, w, h = cv2.boundingRect(big_contour)  # type: ignore
    page_width = img.shape[1]
    page_height = img.shape[0]
    # If we can't find a large contour, then we assume that the picture doesn't have page borders
    is_full_page_view = w < page_width * 0.25 or h < page_height * 0.25
    if is_full_page_view:
        return img, (0, 0)
    # A sample of the paper is enough to know its color
    paper = img[y : y + h, x : x + w][thresh[y : y + h, x : x + w] > 0][::10]
    if len(paper) == 0:
        return img, (0, 0)
    paper_chromaticity = np.median(_chromaticity(paper), axis=0)
    # The contour was eroded, keep a margin so that staffs at the edge of the paper stay
    margin = max(5, max(page_width, page_height) // 100)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(page_width, x + w + margin), min(page_height, y + h + margin)
    x, y, w, h = _keep_sides_with_paper(img, (x0, y0, x1 - x0, y1 - y0), paper_chromaticity)
    if (x, y, w, h) == (0, 0, page_width, page_height):
        return img, (0, 0)

    crop = img[y : y + h, x : x + w]
    return _remove_surroundings(crop, big_contour - (x, y), paper), (x, y)  # type: ignore
