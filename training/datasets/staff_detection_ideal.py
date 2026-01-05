"""
A fast staff detection for ideal staffs with no distortion at all:
Staff lines are perfectly horizontal and bar lines are perfectly vertical.

The idea is based on https://github.com/BreezeWhite/oemer/blob/main/oemer/staffline_extraction.py 
and passed through a couple of iterations in a LLM to get this prototype implementation.

Algorithm overview:
1. Convert image to binary (black pixels = staff lines and symbols).
2. Compute a horizontal histogram (sum of black pixels per row).
3. Detect peaks in the row histogram corresponding to individual staff lines.
   - Collapse nearby rows to handle thick lines.
   - Group lines in sets of 5 to identify each staff.
4. For each staff, compute its horizontal extent by ignoring empty columns.
5. Expand the staff vertically to include surrounding symbols (notes, clefs, etc.).
6. Merge adjacent staffs if their expanded regions touch, forming a grand staff.
7. Detect vertical bar lines for each staff group:
   - Consider only the vertical span from the top staff line to the bottom staff line of the group.
   - Validate columns by requiring continuous black pixels spanning approximately the staff-line height.
   - Merge adjacent candidate columns into a single bar line.
8. Output:
   - Print staff group positions, heights, and number of bar lines.
   - Save a debug histogram image and a result image with staff and barline annotations.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt


ROW_THRESHOLD_RATIO = 0.7
SPACING_TOLERANCE = 6
STAFF_LINES = 5
BAR_HEIGHT_TOL = 0.5


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------

def load_binary_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found")
    _, bin_img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return img, bin_img


def save_row_histogram(row_hist, peaks, out_path):
    plt.figure(figsize=(6, 8))
    plt.plot(row_hist, np.arange(len(row_hist)))
    plt.scatter(row_hist[peaks], peaks, color="red", s=5)
    plt.gca().invert_yaxis()
    plt.xlabel("black pixel count")
    plt.ylabel("row index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ------------------------------------------------------------
# Staff-line detection
# ------------------------------------------------------------

def collapse_thick_lines(rows):
    lines = []
    current = [rows[0]]
    for r in rows[1:]:
        if r == current[-1] + 1:
            current.append(r)
        else:
            lines.append(int(np.mean(current)))
            current = [r]
    lines.append(int(np.mean(current)))
    return np.array(lines)


def detect_staff_lines(bin_img):
    row_hist = bin_img.sum(axis=1)
    thresh = row_hist.max() * ROW_THRESHOLD_RATIO

    raw_peaks = np.where(row_hist > thresh)[0]
    if len(raw_peaks) == 0:
        return []

    peak_rows = collapse_thick_lines(raw_peaks)

    staffs = []
    i = 0
    while i <= len(peak_rows) - STAFF_LINES:
        group = peak_rows[i:i + STAFF_LINES]
        spacings = np.diff(group)
        if np.all(np.abs(spacings - spacings[0]) <= SPACING_TOLERANCE):
            staffs.append((int(group[0]), int(group[-1])))
            i += STAFF_LINES
        else:
            i += 1

    return staffs


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------

def detect_staff_width(bin_img, top, bottom):
    region = bin_img[top:bottom + 1, :]
    col_hist = region.sum(axis=0)
    nz = np.where(col_hist > 0)[0]
    if len(nz) == 0:
        return 0, bin_img.shape[1] - 1
    return int(nz[0]), int(nz[-1])


def expand_staff_height(bin_img, top, bottom, left, right):
    h = bin_img.shape[0]

    t = top
    while t > 0 and bin_img[t - 1, left:right + 1].any():
        t -= 1

    b = bottom
    while b < h - 1 and bin_img[b + 1, left:right + 1].any():
        b += 1

    return t, b


def merge_grandstaffs(staff_boxes):
    merged = []
    for box in staff_boxes:
        if not merged:
            merged.append(box)
            continue

        last = merged[-1]
        if box[0] <= last[1] + 1:
            merged[-1] = (
                min(last[0], box[0]),
                max(last[1], box[1]),
                min(last[2], box[2]),
                max(last[3], box[3]),
            )
        else:
            merged.append(box)

    return merged


# ------------------------------------------------------------
# Barline detection (validated on top and bottom of the whole staff group)
# ------------------------------------------------------------

def detect_barlines(
    bin_img,
    group_line_top,
    group_line_bottom,
    left,
    right,
):
    expected_h = group_line_bottom - group_line_top + 1
    min_h = int(expected_h * (1.0 - BAR_HEIGHT_TOL))
    max_h = int(expected_h * (1.0 + BAR_HEIGHT_TOL))

    region = bin_img[group_line_top:group_line_bottom + 1, left:right + 1]
    h, w = region.shape

    candidate_cols = []

    for x in range(w):
        col = region[:, x]
        y = 0
        while y < h:
            if col[y]:
                y0 = y
                while y < h and col[y]:
                    y += 1
                run_h = y - y0
                if min_h <= run_h <= max_h:
                    candidate_cols.append(left + x)
                    break
            else:
                y += 1

    if not candidate_cols:
        return []

    # merge adjacent columns
    barlines = []
    current = [candidate_cols[0]]
    for c in candidate_cols[1:]:
        if c == current[-1] + 1:
            current.append(c)
        else:
            barlines.append(int(np.mean(current)))
            current = [c]
    barlines.append(int(np.mean(current)))
    return barlines


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(image_path):
    img, bin_img = load_binary_image(image_path)
    base = Path(image_path).with_suffix("")

    # 1. Detect exact staff-line bounds
    staff_lines = detect_staff_lines(bin_img)

    # 2. Expand to include symbols
    expanded = []
    for top, bottom in staff_lines:
        left, right = detect_staff_width(bin_img, top, bottom)
        t2, b2 = expand_staff_height(bin_img, top, bottom, left, right)
        expanded.append((t2, b2, left, right))

    # 3. Merge into grand staffs
    staff_groups = merge_grandstaffs(expanded)

    # 4. Detect barlines (based on top/bottom of the whole group)
    barlines = []
    for (t, b, l, r) in staff_groups:
        barlines.append(
            detect_barlines(
                bin_img,
                t,   # top of staff group
                b,   # bottom of staff group
                l,
                r,
            )
        )

    return staff_groups, barlines


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python staff_detect.py <image>")
        sys.exit(1)
    main(sys.argv[1])
