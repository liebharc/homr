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
    # Use morphological opening to strengthen horizontal lines and remove stems/beams
    # This is more robust than simple global histogram thresholding
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    morphed = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    
    row_hist = morphed.sum(axis=1)
    
    # Local maxima detection to resolve peak rows even if the space between them is not empty
    peak_rows = []
    min_h = bin_img.shape[1] * 0.3 * 255 # 30% width
    for i in range(1, len(row_hist) - 1):
        if row_hist[i] >= row_hist[i-1] and row_hist[i] >= row_hist[i+1] and row_hist[i] >= min_h:
            # Handle plateaus (flat peaks)
            if row_hist[i] == row_hist[i-1]:
                continue
            j = i
            while j < len(row_hist) - 1 and row_hist[j+1] == row_hist[i]:
                j += 1
            peak_rows.append(int((i + j) / 2))
    
    peak_rows = np.array(peak_rows)

    staffs = []
    if len(peak_rows) < STAFF_LINES:
        return []

    # For each peak as a potential top line
    used_indices = set()
    for i in range(len(peak_rows)):
        if i in used_indices:
            continue
            
        # Try different potential spacings by looking at the next peak
        for j in range(i + 1, min(i + 3, len(peak_rows))):
            p1 = peak_rows[i]
            p2 = peak_rows[j]
            spacing = p2 - p1
            if spacing < 5: 
                continue
            
            group = [i, j]
            next_target = p2 + spacing
            for k in range(j + 1, len(peak_rows)):
                if abs(peak_rows[k] - next_target) <= SPACING_TOLERANCE:
                    group.append(k)
                    next_target = peak_rows[k] + spacing
                if len(group) == 5:
                    break
            
            if len(group) == 5:
                staffs.append((int(peak_rows[group[0]]), int(peak_rows[group[-1]])))
                for idx in group:
                    used_indices.add(idx)
                break

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
    """
    staff_boxes: list of (t, b, l, r, line_t, line_b)
    """
    if not staff_boxes:
        return []
    
    # Sort by top position
    staff_boxes.sort(key=lambda x: x[0])
    
    merged = []
    for box in staff_boxes:
        if not merged:
            # list of: [t, b, l, r, [list of (lt, lb) individual lines]]
            merged.append([box[0], box[1], box[2], box[3], [(box[4], box[5])]])
            continue

        last = merged[-1]
        # Use unexpanded staff-line bounds for the merging decision
        # box: (t, b, l, r, line_t, line_b)
        last_line_b = max(r[1] for r in last[4])
        current_line_t = box[4]
        
        # If the gap between staff lines is within 60px, it's likely a grandstaff.
        if current_line_t <= last_line_b + 60:
            last[0] = min(last[0], box[0])
            last[1] = max(last[1], box[1])
            last[2] = min(last[2], box[2])
            last[3] = max(last[3], box[3])
            last[4].append((box[4], box[5]))
        else:
            merged.append([box[0], box[1], box[2], box[3], [(box[4], box[5])]])

    return merged


# ------------------------------------------------------------
# Barline detection (robust to gaps between staves)
# ------------------------------------------------------------

def detect_barlines(
    bin_img,
    staff_line_ranges, # list of (lt, lb) for each staff in group
    left,
    right,
):
    # Use morphological opening with a vertical kernel. 
    # Width=1 matches barlines but also stems.
    # Height=25 filters out most smaller vertical bits.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    morphed = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    
    candidate_cols = []

    # Check each column in the staff region
    for x in range(left, right + 1):
        # A column is a barline if it has vertical runs covering ALMOST ALL of each staff.
        # Barlines in MusiXQA are very consistent. Stems are often shorter or broken.
        matched_staves = 0
        for (lt, lb) in staff_line_ranges:
            h_staff = lb - lt + 1
            active = np.count_nonzero(morphed[lt:lb+1, x])
            if active >= h_staff * 0.95: # Ultra strict
                # Strict check: barlines shouldn't connect to note heads above/below
                # Check a wider window and range of 0s
                above = 0
                below = 0
                if lt - 10 >= 0:
                    above = np.count_nonzero(bin_img[lt-10:lt-2, x-2:x+3])
                if lb + 10 < bin_img.shape[0]:
                    below = np.count_nonzero(bin_img[lb+2:lb+10, x-2:x+3])
                
                if above == 0 and below == 0:
                    matched_staves += 1
        
        if matched_staves == len(staff_line_ranges): 
            candidate_cols.append(x)

    if not candidate_cols:
        return []

    # merge adjacent columns
    barlines = []
    current = [candidate_cols[0]]
    for c in candidate_cols[1:]:
        if c <= current[-1] + 10: 
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

    # 1. Detect exact staff-line bounds
    staff_lines = detect_staff_lines(bin_img)

    # 2. Expand to include symbols
    staff_boxes = []
    for line_top, line_bottom in staff_lines:
        left, right = detect_staff_width(bin_img, line_top, line_bottom)
        t, b = expand_staff_height(bin_img, line_top, line_bottom, left, right)
        staff_boxes.append((t, b, left, right, line_top, line_bottom))

    # 3. Merge into grand staffs
    merged_groups = merge_grandstaffs(staff_boxes)
    
    # 4. Filter groups: remove those that are too short or weird aspect ratio
    # A staff should have a decent width. MusiXQA images are ~1200 wide.
    staff_groups = []
    group_barlines = []
    
    for group in merged_groups:
        t, b, l, r, line_ranges = group
        width = r - l + 1
        if width < 500: # Filter out titles, clef-only bits etc
            continue
            
        bls = detect_barlines(bin_img, line_ranges, l, r)
        
        # Final sanity check: at least 2 barlines (start and end)
        if len(bls) < 2:
            continue
            
        # Unpack lt/lb for compatibility: use min/max of ranges
        lt = min(r[0] for r in line_ranges)
        lb = max(r[1] for r in line_ranges)
        
        staff_groups.append((t, b, l, r, lt, lb))
        group_barlines.append(bls)

    return staff_groups, group_barlines


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python staff_detect.py <image>")
        sys.exit(1)
    main(sys.argv[1])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python staff_detect.py <image>")
        sys.exit(1)
    main(sys.argv[1])
