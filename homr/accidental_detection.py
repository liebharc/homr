from homr import constants
from homr.bounding_boxes import RotatedBoundingBox
from homr.model import Accidental, Staff


def add_accidentals_to_staffs(
    staffs: list[Staff], accidentals: list[RotatedBoundingBox]
) -> list[Accidental]:
    result = []
    for staff in staffs:
        for accidental in accidentals:
            if not staff.is_on_staff_zone(accidental):
                continue
            point = staff.get_at(accidental.center[0])
            if point is None:
                continue

            min_width_or_height = constants.minimum_accidental_width_or_height(
                staff.average_unit_size
            )
            max_width_or_height = constants.maximum_accidental_width_or_height(
                staff.average_unit_size
            )

            if (
                accidental.size[0] < min_width_or_height
                or accidental.size[0] > max_width_or_height
                or accidental.size[1] < min_width_or_height
                or accidental.size[1] > max_width_or_height
            ):
                continue

            position = point.find_position_in_unit_sizes(accidental)
            accidental_bbox = accidental.to_bounding_box()
            clef_symbol = Accidental(accidental_bbox, position)
            staff.add_symbol(clef_symbol)
            result.append(clef_symbol)
    return result
