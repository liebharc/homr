"""
Staff images engraved by Verovio from the MusicXML which MuseScore exported, as a second
renderer next to MuseScore. A part of a piece is either fully rendered by Verovio or fully by
MuseScore, so the same bars never end up in training and validation with different images.
"""

import hashlib
import json
import random
import re
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from homr.circle_of_fifths import strip_naturals
from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.music_xml_parser import Measure
from training.transformer.image_utils import crop_info_path

# Share of the parts rendered by Verovio
VEROVIO_SHARE_SINGLE_STAFF = 0.5
VEROVIO_SHARE_GRAND_STAFF = 0.1
FONTS = ["Leipzig", "Bravura", "Gootville", "Leland", "Petaluma"]
# Staff space of 7.5 to 13.5 px on the 1400 px wide page, MuseScore's Lieder and PDMX pages
# have 7 to 12 px
SCALES = [25, 30, 35, 40, 45]
RASTER_WIDTH = 1400
STORED_MARGIN_IN_STAFF_SPACES = 6
# Dynamics farther away from the staff lines belong to another staff
MAX_DYNAMIC_DISTANCE_IN_STAFF_SPACES = 8
# Text below or above a staff, it's not part of the labels
_IGNORED_CLASSES = {"verse", "syl", "label", "labelAbbr", "mNum", "dir", "tempo", "harm"}

_SVG = "{http://www.w3.org/2000/svg}"


@dataclass
class VerovioStaff:
    """A staff in raster pixels: its lines and everything which belongs to it."""

    left: float
    right: float
    line_top: float
    line_bottom: float
    top: float
    bottom: float
    staff_space: float

    def merge(self, other: "VerovioStaff") -> "VerovioStaff":
        return VerovioStaff(
            min(self.left, other.left),
            max(self.right, other.right),
            min(self.line_top, other.line_top),
            max(self.line_bottom, other.line_bottom),
            min(self.top, other.top),
            max(self.bottom, other.bottom),
            self.staff_space,
        )


@dataclass
class VerovioSystem:
    page: int
    bars: int
    staffs: list[VerovioStaff]
    # Staves (counted over all parts) with a clef which Verovio draws in the wrong place
    misplaced_clefs: set[int] = field(default_factory=set)


def _random_for(name: str) -> random.Random:
    seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def choose_parts(name: str, staves_per_part: list[int]) -> set[int]:
    """The parts of a piece which Verovio renders, the same for every run."""
    rng = _random_for(name)
    result = set()
    for part, staves in enumerate(staves_per_part):
        share = VEROVIO_SHARE_GRAND_STAFF if staves > 1 else VEROVIO_SHARE_SINGLE_STAFF
        if rng.random() < share:
            result.add(part)
    return result


def _duration(element: ET.Element) -> int:
    return int(element.findtext("duration") or 0)


def fill_voices(root: ET.Element) -> None:
    """
    Pads every voice which ends early on its staff to the end of the measure with an invisible
    <forward>. Verovio draws the clef change of the next measure after the last element of a
    layer, in the middle of the measure if the voice ends early. Cross-staff voices stay as they
    are, padding them on one staff misplaces the clef as well.
    """
    for part in root.findall("part"):
        for measure in part.findall("measure"):
            cursor = start = measure_end = 0
            ends: dict[tuple[str, str], int] = {}
            for element in measure:
                if element.tag == "note":
                    if element.find("grace") is not None:
                        continue
                    if element.find("chord") is None:
                        start = cursor
                        cursor += _duration(element)
                    key = (element.findtext("staff") or "1", element.findtext("voice") or "1")
                    ends[key] = max(ends.get(key, 0), start + _duration(element))
                elif element.tag == "backup":
                    cursor -= _duration(element)
                elif element.tag == "forward":
                    cursor += _duration(element)
                measure_end = max(measure_end, cursor)
            staves_of_voice: dict[str, set[str]] = {}
            for staff, voice in ends:
                staves_of_voice.setdefault(voice, set()).add(staff)
            for (staff, voice), end in ends.items():
                if len(staves_of_voice[voice]) > 1:
                    continue
                if end >= measure_end or cursor < end:
                    continue  # Complete, or the cursor can't go back to the end of the voice
                if cursor > end:
                    backup = ET.SubElement(measure, "backup")
                    ET.SubElement(backup, "duration").text = str(cursor - end)
                forward = ET.SubElement(measure, "forward")
                ET.SubElement(forward, "duration").text = str(measure_end - end)
                ET.SubElement(forward, "voice").text = voice
                ET.SubElement(forward, "staff").text = staff
                cursor = measure_end


def clef_changes_within_measures(root: ET.Element) -> dict[tuple[int, int], int]:
    """Number of clef changes after the start of a measure, by (measure, staff over all parts)."""
    result: dict[tuple[int, int], int] = {}
    staff_offset = 0
    for part in root.findall("part"):
        staves = 1
        for index, measure in enumerate(part.findall("measure")):
            cursor = 0
            for element in measure:
                if element.tag == "note":
                    if element.find("chord") is None and element.find("grace") is None:
                        cursor += _duration(element)
                elif element.tag == "backup":
                    cursor -= _duration(element)
                elif element.tag == "forward":
                    cursor += _duration(element)
                elif element.tag == "attributes":
                    staves = int(element.findtext("staves") or staves)
                    if cursor <= 0:
                        continue
                    for clef in element.findall("clef"):
                        key = (index, staff_offset + int(clef.get("number", "1")) - 1)
                        result[key] = result.get(key, 0) + 1
        staff_offset += staves
    return result


def _x_of(group: ET.Element) -> float | None:
    for use in group.iter(_SVG + "use"):
        match = re.search(r"translate\(([\d.-]+)", use.get("transform", ""))
        return float(match.group(1)) if match else None
    return None


def _class(element: ET.Element) -> str:
    """The element's class, empty for the bounding box which svgBoundingBoxes adds to it."""
    classes = (element.get("class") or "").split(" ")
    return "" if "bounding-box" in classes else classes[0]


def _boxes(element: ET.Element) -> list[tuple[float, float, float, float]]:
    """Bounding boxes (x, y, width, height) of the element and its children, without text."""
    result: list[tuple[float, float, float, float]] = []
    if _class(element) in _IGNORED_CLASSES:
        return result
    for child in element:
        if child.tag == _SVG + "g" and "bounding-box" in (child.get("class") or ""):
            for rect in child.iter(_SVG + "rect"):
                result.append(
                    (
                        float(rect.get("x", 0)),
                        float(rect.get("y", 0)),
                        float(rect.get("width", 0)),
                        float(rect.get("height", 0)),
                    )
                )
        else:
            result.extend(_boxes(child))
    return result


def _staff_lines(staff: ET.Element) -> tuple[float, float, float, float]:
    lines = [
        [float(n) for n in re.findall(r"[\d.-]+", path.get("d", ""))]
        for path in staff
        if path.tag == _SVG + "path"
    ][:5]
    xs = [v for line in lines for v in (line[0], line[2])]
    ys = [line[1] for line in lines]
    return min(xs), max(xs), min(ys), max(ys)


def _misplaced_clefs(staff: ET.Element) -> int:
    """Clefs which Verovio draws between the notes of a staff in a measure."""
    notes = [
        x
        for g in staff.iter(_SVG + "g")
        if _class(g) in ("note", "rest", "chord")
        for x in [_x_of(g)]
        if x is not None
    ]
    if not notes:
        return 0
    count = 0
    for g in staff.iter(_SVG + "g"):
        if _class(g) != "clef":
            continue
        x = _x_of(g)
        if x is not None and min(notes) < x < max(notes):
            count += 1
    return count


def parse_page(
    svg: str, page: int, first_measure: int, clef_changes: dict[tuple[int, int], int]
) -> list[VerovioSystem]:
    """The systems of a page rendered with svgBoundingBoxes, in raster pixels."""
    view = re.search(r'class="definition-scale"[^>]*viewBox="0 0 ([\d.]+)', svg)
    margin = re.search(r'class="page-margin" transform="translate\(([\d.-]+), ([\d.-]+)\)', svg)
    if view is None or margin is None:
        raise ValueError("Unexpected Verovio SVG")
    k = RASTER_WIDTH / float(view.group(1))
    mx, my = float(margin.group(1)), float(margin.group(2))

    def to_px(x: float, y: float) -> tuple[float, float]:
        return (x + mx) * k, (y + my) * k

    systems = []
    measure_index = first_measure
    for system in ET.fromstring(svg).iter(_SVG + "g"):  # noqa: S314
        if _class(system) != "system":
            continue
        measures = [g for g in system if _class(g) == "measure"]
        staffs: dict[int, VerovioStaff] = {}
        misplaced: set[int] = set()
        for measure in measures:
            staff_groups = [g for g in measure if _class(g) == "staff"]
            for n, staff in enumerate(staff_groups):
                left, right, top, bottom = _staff_lines(staff)
                x0, y0 = to_px(left, top)
                x1, y1 = to_px(right, bottom)
                content_top, content_bottom = y0, y1
                for bx, by, _bw, bh in _boxes(staff):
                    content_top = min(content_top, to_px(bx, by)[1])
                    content_bottom = max(content_bottom, to_px(bx, by + bh)[1])
                box = VerovioStaff(x0, x1, y0, y1, content_top, content_bottom, (y1 - y0) / 4)
                staffs[n] = staffs[n].merge(box) if n in staffs else box
                if _misplaced_clefs(staff) > clef_changes.get((measure_index, n), 0):
                    misplaced.add(n)
            _add_dynamics(measure, staffs, to_px)
            measure_index += 1
        systems.append(
            VerovioSystem(page, len(measures), [staffs[n] for n in sorted(staffs)], misplaced)
        )
    return systems


def _distance_to_lines(staff: VerovioStaff, y: float) -> float:
    return max(staff.line_top - y, y - staff.line_bottom, 0)


def _add_dynamics(
    measure: ET.Element,
    staffs: dict[int, VerovioStaff],
    to_px: Callable[[float, float], tuple[float, float]],
) -> None:
    """Dynamics are part of the labels, the staff next to them grows to include them."""
    for dynamic in (g for g in measure if _class(g) == "dynam"):
        for bx, by, bw, bh in _boxes(dynamic):
            x0, y0 = to_px(bx, by)
            x1, y1 = to_px(bx + bw, by + bh)
            center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
            candidates = [s for s in staffs.values() if s.left <= center_x <= s.right]
            if not candidates:
                continue
            staff = min(candidates, key=lambda s: _distance_to_lines(s, center_y))
            limit = MAX_DYNAMIC_DISTANCE_IN_STAFF_SPACES * staff.staff_space
            if _distance_to_lines(staff, center_y) > limit:
                continue
            staff.top = min(staff.top, y0)
            staff.bottom = max(staff.bottom, y1)


@dataclass
class VerovioScore:
    pages: list[str]
    systems: list[VerovioSystem]


def render(musicxml_file: str, name: str) -> VerovioScore:
    import verovio  # noqa: PLC0415 - only the dataset conversion needs it

    root = ET.parse(musicxml_file).getroot()  # noqa: S314
    clef_changes = clef_changes_within_measures(root)
    fill_voices(root)
    rng = _random_for(name + "-style")
    toolkit = verovio.toolkit()
    toolkit.setOptions(
        {
            "pageWidth": 2100,
            "pageHeight": 2970,
            "scale": rng.choice(SCALES),
            "font": rng.choice(FONTS),
            "header": "none",
            "footer": "none",
            "svgBoundingBoxes": True,
        }
    )
    if not toolkit.loadData(ET.tostring(root, encoding="unicode")):
        raise ValueError("Verovio failed to load " + musicxml_file)
    pages: list[str] = []
    systems: list[VerovioSystem] = []
    for page in range(1, toolkit.getPageCount() + 1):
        svg = toolkit.renderToSVG(page)
        first_measure = sum(s.bars for s in systems)
        systems.extend(parse_page(svg, page, first_measure, clef_changes))
        pages.append(svg)
    return VerovioScore(pages, systems)


def has_multi_measure_rests(musicxml_file: str) -> bool:
    return "<multiple-rest" in Path(musicxml_file).read_text(encoding="utf-8")


def split_into_staffs(  # noqa: PLR0913
    musicxml_file: str,
    voices: list[list[Measure]],
    staves_per_part: list[int],
    parts: set[int],
    just_token_files: bool,
    flat_folder: str,
) -> list[tuple[int, str, str, list[EncodedSymbol]]]:
    """
    (part, image, token file, symbols) of the staves of the given parts, rendered by Verovio.
    Empty if the Verovio layout doesn't match the voices.
    """
    from training.omr_datasets.convert_lieder import (  # noqa: PLC0415 - circular import
        MeasureCutter,
        contains_only_supported_clefs,
    )

    name = Path(musicxml_file).stem
    score = render(musicxml_file, name)
    if sum(s.bars for s in score.systems) != len(voices[0]):
        eprint(musicxml_file, "INFO: Verovio drew a different number of bars, using MuseScore")
        return []
    if any(len(s.staffs) != sum(staves_per_part) for s in score.systems):
        eprint(musicxml_file, "INFO: Verovio drew a different number of staves, using MuseScore")
        return []

    first_staff = [sum(staves_per_part[:part]) for part in range(len(staves_per_part))]
    usable = {
        part
        for part in parts
        if not any(
            first_staff[part] + i in system.misplaced_clefs
            for system in score.systems
            for i in range(staves_per_part[part])
        )
    }
    if not usable:
        return []

    cutters = [MeasureCutter(list(voice)) for voice in voices]
    pngs: dict[int, str] = {}
    result = []
    for number, system in enumerate(score.systems, start=1):
        for part, cutter in enumerate(cutters):
            symbols = cutter.extract_measures(system.bars)
            if part not in usable or not contains_only_supported_clefs(symbols):
                continue
            staves = system.staffs[first_staff[part] : first_staff[part] + staves_per_part[part]]
            staff = staves[0]
            for other in staves[1:]:
                staff = staff.merge(other)
            base = Path(flat_folder) / f"{name}-vrv-{number}-{part}"
            image_file = str(base) + ".png"
            if just_token_files:
                if not Path(image_file).exists():
                    continue
            else:
                if system.page not in pngs:
                    pngs[system.page] = _rasterize(score.pages[system.page - 1], f"{base}-page.png")
                _write_crop(pngs[system.page], staff, image_file)
            result.append((part, image_file, str(base) + ".tokens", strip_naturals(symbols)))
    for page_png in pngs.values():
        Path(page_png).unlink(missing_ok=True)
    return result


def _rasterize(svg: str, png_file: str) -> str:
    subprocess.run(  # noqa: S603
        ["rsvg-convert", "-b", "white", "-w", str(RASTER_WIDTH), "-o", png_file],  # noqa: S607
        input=svg.encode("utf-8"),
        check=True,
    )
    return png_file


def _write_crop(page_png: str, staff: VerovioStaff, image_file: str) -> None:
    image = cv2.imread(page_png, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to read " + page_png)
    # Same margins as convert_lieder: the data loader draws the final margin from the JSON
    y_offset = STORED_MARGIN_IN_STAFF_SPACES * staff.staff_space
    x_min = max(0, int(staff.left - 40))
    x_max = int(staff.right + 10)
    y_min = max(0, int(staff.top - y_offset))
    y_max = int(staff.bottom + y_offset)
    cv2.imwrite(image_file, image[y_min:y_max, x_min:x_max])
    crop_info = {
        "stems": [staff.top - y_min, staff.bottom - y_min],
        "lines": [staff.line_top - y_min, staff.line_bottom - y_min],
        "staff_space": staff.staff_space,
    }
    Path(crop_info_path(image_file)).write_text(json.dumps(crop_info))
