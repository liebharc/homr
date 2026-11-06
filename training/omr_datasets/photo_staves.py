"""
Photo pages (photo_paste.py) cut into staves the way homr reads them, each with its exact answer.

homr finds and straightens the staves of a generated photo exactly as when it reads one. Each
staff is traced back through the photo's map to the flat page, so it is known which staff it is
and which bars it holds; the answer comes from the Lieder converter's code for those bars. A staff
whose ends don't land on the ends of a drawn staff is dropped, so a crop never gets a wrong answer.
"""

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock

import cv2
from numpy.typing import NDArray

import homr.staff_parsing as staff_parsing
from homr.circle_of_fifths import strip_naturals
from homr.debug import Debug
from homr.staff_regions import StaffRegions
from homr.system_repair import repair_systems
from training.omr_datasets.convert_lieder import (
    MeasureCutter,
    _count_staffs,
    contains_only_supported_clefs,
    git_root,
)
from training.omr_datasets.music_xml_parser import music_xml_file_to_tokens
from training.transformer.training_vocabulary import token_lines_to_str

if TYPE_CHECKING:
    from homr.main import ProcessingConfig

END_SLACK = 0.04  # how far (share of the staff's width) a found staff's ends may land from the drawn ends

Target = tuple[tuple[int, int], list[float]]
Point = tuple[float, float]


def answers(
    musicxml_path: str, systems: list[dict[str, Any]]
) -> tuple[dict[tuple[int, int], str | None] | None, list[int]]:
    """The tokens of every (system, part), cut bar by bar in page order as the Lieder converter
    does (None for a part with an unsupported clef), and how many staves each part has. None when
    the parts don't fill the systems' staves, or homr's token code can't read the song."""
    try:
        voices = music_xml_file_to_tokens(musicxml_path)
        staves = [max(_count_staffs(v), 1) for v in voices]
        if any(len(s["staffs"]) != sum(staves) for s in systems):
            return None, staves
        if sum(s["bars"] for s in systems) != len(voices[0]):
            return None, staves
        cutters = [MeasureCutter(v) for v in voices]
        result: dict[tuple[int, int], str | None] = {}
        for i, system in enumerate(systems):
            for part, cutter in enumerate(cutters):
                symbols = cutter.extract_measures(system["bars"])
                result[i, part] = (
                    token_lines_to_str(strip_naturals(symbols))
                    if contains_only_supported_clefs(symbols)
                    else None
                )
        return result, staves
    except Exception:  # e.g. octave shifts, which the Lieder converter skips too
        return None, []


def targets(systems: list[dict[str, Any]], staves: list[int]) -> list[Target]:
    """The flat box of every (system, part): a grand staff is its staves together."""
    found = []
    for i, system in enumerate(systems):
        first = 0
        for part, count in enumerate(staves):
            boxes = system["staffs"][first : first + count]
            box = [
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            ]
            found.append(((i, part), box))
            first += count
    return found


def match_staff(
    ends: tuple[Point, Point, Point], drawn: list[Target], slack: float = END_SLACK
) -> tuple[int, int] | None:
    """The (system, part) whose box holds the found staff's middle and whose ends lie within
    `slack` of its left and right end. ends: the staff's left end, middle and right end on the
    flat page."""
    (left, _), (_, middle), (right, _) = ends
    match = None
    for key, (box_left, top, box_right, bottom) in drawn:
        allowed = slack * (box_right - box_left)
        if (
            top <= middle <= bottom
            and abs(left - box_left) <= allowed
            and abs(right - box_right) <= allowed
        ):
            match = key
    return match


def cut_page(
    image_path: Path,
    maps: tuple[NDArray, NDArray],
    page: dict[str, Any],
    config: "ProcessingConfig",
    out_dir: Path,
) -> tuple[list[str], Counter[str]]:
    """Cuts one generated photo (its answer beside it as .musicxml) into staves. Writes
    <photo>-<n>.png, .tokens and .where (which system and part it is) to out_dir. Returns the
    index lines (image,tokens relative to the git root) and why staves were dropped."""
    from homr.main import detect_staffs_in_image  # imports the whole reader, OCR included

    dropped: Counter[str] = Counter()
    tokens, staves = answers(str(image_path.with_suffix(".musicxml")), page["systems"])
    if tokens is None:
        dropped["no answer for the song"] += 1
        return [], dropped
    drawn = targets(page["systems"], staves)
    map_x, map_y = maps
    try:
        multi_staffs, image, debug, _, _, to_input = detect_staffs_in_image(str(image_path), config)
        multi_staffs = repair_systems(debug, multi_staffs)
    except Exception:
        dropped["homr found no staves"] += 1
        return [], dropped
    regions = StaffRegions(multi_staffs)
    original = staff_parsing.center_image_on_canvas
    captured: dict[str, NDArray] = {}

    def capture(staff_image: NDArray, *args: Any, **kwargs: Any) -> NDArray:
        # the straightened staff before homr pads it onto the canvas: training pads it itself
        captured["image"] = staff_image
        return original(staff_image, *args, **kwargs)

    lines = []
    n = 0
    for row in multi_staffs:
        for staff in row.staffs:
            n += 1
            try:
                with mock.patch.object(staff_parsing, "center_image_on_canvas", side_effect=capture):
                    staff_parsing.prepare_staff_image(
                        Debug(image, str(image_path), False), n, staff, image, regions
                    )
            except Exception:
                dropped["homr could not straighten it"] += 1
                continue
            # homr's staff (left end, middle, right end), not the crop: the crop reaches further
            # towards the farther neighbour, so its middle can lie above or below the staff
            mid_y = (staff.min_y + staff.max_y) / 2
            ends = []
            for x in (staff.min_x, (staff.min_x + staff.max_x) / 2, staff.max_x):
                px, py = to_input((x, mid_y))
                xi, yi = round(px), round(py)
                inside = 0 <= yi < map_x.shape[0] and 0 <= xi < map_x.shape[1] and map_x[yi, xi] >= 0
                ends.append((float(map_x[yi, xi]), float(map_y[yi, xi])) if inside else None)
            if None in ends:
                dropped["an end off the page"] += 1
                continue
            match = match_staff((ends[0], ends[1], ends[2]), drawn)  # type: ignore[arg-type]
            if match is None:
                dropped["ends not on a drawn staff"] += 1
                continue
            text = tokens[match]
            if text is None:
                dropped["unsupported clef"] += 1
                continue
            name = out_dir / f"{image_path.stem}-{n}"
            cv2.imwrite(str(name.with_suffix(".png")), captured["image"])
            name.with_suffix(".tokens").write_text(text, encoding="utf-8")
            name.with_suffix(".where").write_text(json.dumps(match), encoding="utf-8")
            png, token_file = name.with_suffix(".png"), name.with_suffix(".tokens")
            lines.append(f"{png.relative_to(git_root)},{token_file.relative_to(git_root)}")
    return lines, dropped
