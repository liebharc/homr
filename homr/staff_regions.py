from homr.model import MultiStaff, Staff


def _overlaps_horizontally(staff: Staff, other: Staff) -> bool:
    return not (other.min_x > staff.max_x or staff.min_x > other.max_x)


class StaffRegions:
    def __init__(self, staffs: list[MultiStaff]) -> None:
        self.staffs = [s for ms in staffs for s in ms.staffs]

    def get_start_of_closest_staff_above(self, staff: Staff) -> float:
        staffs_above = [
            other.max_y
            for other in self.staffs
            if other.min_y < staff.min_y and _overlaps_horizontally(staff, other)
        ]
        if len(staffs_above) == 0:
            return 0

        return max(staffs_above)

    def get_start_of_closest_staff_below(self, staff: Staff) -> float:
        staffs_below = [
            other.min_y
            for other in self.staffs
            if other.max_y > staff.max_y and _overlaps_horizontally(staff, other)
        ]
        if len(staffs_below) == 0:
            return 1e12  # should be larger than the height of every reasonable image

        return min(staffs_below)
