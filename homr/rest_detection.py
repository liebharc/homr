import numpy as np

from homr import constants
from homr.bounding_boxes import RotatedBoundingBox
from homr.model import Rest, Staff


def add_rests_to_staffs(staffs: list[Staff], rests: list[RotatedBoundingBox]) -> list[Rest]:
    result = []
    central_staff_line_indexes = [1, 2]
    for staff in staffs:
        for rest in rests:
            if not staff.is_on_staff_zone(rest):
                continue
            point = staff.get_at(rest.center[0])
            if point is None:
                continue

            center = rest.center
            idx_of_closest_y = np.argmin(np.abs([y_value - center[1] for y_value in point.y]))
            is_in_center = idx_of_closest_y in central_staff_line_indexes
            if not is_in_center:
                continue

            minimum_width_or_height = constants.minimum_rest_width_or_height(
                point.average_unit_size
            )
            maximum_width_or_height = constants.maximum_rest_width_or_height(
                point.average_unit_size
            )

            if rest.size[0] < minimum_width_or_height or rest.size[1] < minimum_width_or_height:
                continue

            if rest.size[0] > maximum_width_or_height or rest.size[1] > maximum_width_or_height:
                continue

            bbox = rest.to_bounding_box()
            rest_symbol = Rest(bbox)
            staff.add_symbol(rest_symbol)
            result.append(rest_symbol)
    return result
