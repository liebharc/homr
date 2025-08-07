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


def tolerance_for_touching_bar_lines(unit_size: float) -> int:
    return int(round(unit_size * 2))


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


def bar_line_to_staff_tolerance(unit_size: float) -> float:
    return 4 * unit_size


def black_spot_removal_threshold(unit_size: float) -> float:
    return 2 * unit_size


staff_line_segment_x_tolerance = 10

notehead_type_threshold = 0.8

max_color_distance_of_staffs = 0.25

# We don't have to worried about mis-detections,
# because if not all staffs group the same way then we break the staffs up again
minimum_connections_to_form_combined_staff = 1

duration_of_quarter = 16

image_noise_limit = 50

staff_position_tolerance = 50

max_angle_for_lines_to_be_parallel = 5


NOTEHEAD_SIZE_RATIO = 1.285714  # width/height


def minimum_rest_width_or_height(unit_size: float) -> float:
    return 0.7 * unit_size


def maximum_rest_width_or_height(unit_size: float) -> float:
    return 3.5 * unit_size


def minimum_accidental_width_or_height(unit_size: float) -> float:
    return 0.5 * unit_size


def maximum_accidental_width_or_height(unit_size: float) -> float:
    return 3 * unit_size


# We use ³ as triplet indicator as it's not a valid duration name
# or note name and thus we have no risk of confusion
triplet_symbol = "³"
