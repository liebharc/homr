import cv2
import numpy as np

from homr.type_definitions import NDArray


def autocrop(img: NDArray) -> NDArray:
    """
    Find the largest contour on the image, which is expected to be the paper of sheet music
    and extracts it from the image. If no contour is found, then the image is assumed to be
    a full page view of sheet music and is returned as is.
    """
    result, _top_left = autocrop_with_offset(img)
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
    is_full_page_view = x < page_width * 0.25 or y < page_height * 0.25
    if is_full_page_view:
        return img, (0, 0)

    # crop result
    result = img[y : y + h, x : x + w]
    return result, (x, y)
