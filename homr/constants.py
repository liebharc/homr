number_of_lines_on_a_staff = 5

max_number_of_ledger_lines = 4


def tolerance_for_staff_line_detection(unit_size: float) -> float:
    return unit_size / 3


def max_line_gap_size(unit_size: float) -> float:
    return 5 * unit_size


def is_short_line(unit_size: float) -> float:
    return unit_size / 5


def is_short_connected_line(unit_size: float) -> float:
    return 2 * unit_size


def min_height_for_brace_rough(unit_size: float) -> float:
    return 2 * unit_size


def max_width_for_brace_rough(unit_size: float) -> float:
    return 3 * unit_size


def min_height_for_brace(unit_size: float) -> float:
    return 4 * unit_size


def tolerance_for_touching_clefs(unit_size: float) -> int:
    return int(round(unit_size * 2))


def tolerance_for_staff_at_any_point(unit_size: float) -> int:
    return 0


def tolerance_note_grouping(unit_size: float) -> float:
    return 1 * unit_size


def bar_line_max_width(unit_size: float) -> float:
    return 2 * unit_size


def bar_line_min_height(unit_size: float) -> float:
    return 3 * unit_size


def black_spot_removal_threshold(unit_size: float) -> float:
    return 2 * unit_size


staff_line_segment_x_tolerance = 10

# We don't have to worried about mis-detections,
# because if not all staffs group the same way then we break the staffs up again
minimum_connections_to_form_combined_staff = 1

duration_of_quarter = 16

image_noise_limit = 50

staff_position_tolerance = 50

max_angle_for_lines_to_be_parallel = 10


NOTEHEAD_SIZE_RATIO = 1.285714  # width/height

grandstaff_x_distance_threshold_factor = 5
grandstaff_y_overlap_threshold_factor = 0.5

# A brace/bracket candidate blob can end up merged with unrelated ink that
# happens to touch it during preprocessing (e.g. a neighboring staff's clef).
# Such contamination is reliably thinner than the brace itself, so we keep
# only the vertical range where the blob is at least this fraction as wide
# as its own widest row, which recovers the true brace span regardless of
# how much extra ink got attached to it.
brace_core_width_ratio = 0.5

# prepare_brace_dot_image's morphological dilation (homr/brace_dot_detection.py) uses a
# 5px-wide kernel to bridge small gaps in brace/bracket ink into single blobs. A whole-system
# bracket that crosses a staff line's own ink can occasionally break into two contours there:
# the main bracket (correctly wide) plus a small, sparse leftover fragment too thin to fully
# dilate back up to kernel width. That leftover is small enough to fit entirely within a
# single staff pair's span, so it can still score well in _score_brace_with_staff_pair despite
# being noise, not a real brace - this is an absolute pixel width tied to the kernel itself,
# not to staff unit size, since it is about the morphology op, not the music engraving.
# 5 (not 4): observed leftover fragments were 2-4px wide, while every genuine candidate
# (barline connectors, real braces) observed across smb/polish-scores/testdata was >=5px -
# exactly the dilation kernel's own width.
min_width_for_brace_dot_candidate = 5
