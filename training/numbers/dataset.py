# flake8: noqa: T201

"""
Training data for homr.number_recognition: crops of staves around time signatures,
multi-measure rests and other symbols, with the numbers which they show.

Sources:
- Lieder and string quartet scores (OpenScore), engraved by Verovio. Their time signatures
  are changed at random and time signature changes and multi-measure rests are added, so
  that all numbers occur. The surrounding music is real.
- DeepScores dense pages (time signatures only, their multirest numbers have no label).

Each sample is a context crop, larger than the model input, with the staff in fixed
staff units: the anchor x in the middle, the top staff line at CONTEXT_TOP. The training
cuts the input from it with random errors of x, the staff lines and the scale.
Usage: poetry run python -m training.numbers.dataset
"""

import hashlib
import json
import random
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from homr.type_definitions import NDArray

git_root = Path(__file__).resolve().parent.parent.parent
dataset_root = git_root / "datasets"
lieder_scores = dataset_root / "Lieder-main" / "scores"
deepscores = dataset_root / "ds2_dense"
photo_templates = dataset_root / "homr-photo-pages-main" / "templates" / "templates.json"
output_root = dataset_root / "numbers"

# Context crop: pixels per staff unit, half width and the units above the top line and
# below the bottom line
CONTEXT_PX = 10
CONTEXT_HALF_WIDTH = 5
CONTEXT_ABOVE = 7
CONTEXT_BELOW = 3
CONTEXT_WIDTH = 2 * CONTEXT_HALF_WIDTH * CONTEXT_PX
CONTEXT_HEIGHT = (CONTEXT_ABOVE + 4 + CONTEXT_BELOW) * CONTEXT_PX
CONTEXT_TOP = CONTEXT_ABOVE * CONTEXT_PX

KIND_NEGATIVE, KIND_TIME_SIGNATURE, KIND_MULTIREST = 0, 1, 2
NONE = 0  # no number
IGNORE = -1  # not labeled, e.g. the time signature at a multirest
# A negative is at least this far from a labeled symbol, a positive's x error stays below
# 2.5 units, nothing in between is trained
MIN_NEGATIVE_DISTANCE = 4.5

FONTS = ["Leipzig", "Bravura", "Gootville", "Leland", "Petaluma"]
MULTIREST_STYLES = ["auto", "default", "block", "symbols"]
DENOMINATORS = [1, 2, 4, 8, 16, 32]
_SVG = "{http://www.w3.org/2000/svg}"
_DIGIT = re.compile(r"E08([0-9])")
_COMMON, _CUT = "E08A", "E08B"
# Symbols at which negatives are taken
_NEGATIVE_CLASSES = {
    "note",
    "rest",
    "mRest",
    "clef",
    "keySig",
    "barLine",
    "chord",
    "dynam",
    "tuplet",
    "fing",
    "dir",
    "tempo",
}


@dataclass
class Anchor:
    x: float
    kind: int
    numerator: int = NONE
    denominator: int = NONE
    count: int = NONE


@dataclass
class StaffOnPage:
    top: float  # y of the top line
    unit: float
    left: float
    right: float
    anchors: list[Anchor]
    negatives: list[float]


def context_crop(page: NDArray, x: float, top: float, unit: float) -> NDArray:
    """The context crop with x in the middle and the top staff line at CONTEXT_TOP."""
    scale = CONTEXT_PX / unit
    matrix = np.array(
        [
            [scale, 0, CONTEXT_WIDTH / 2 - x * scale],
            [0, scale, CONTEXT_TOP - top * scale],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        page,
        matrix,
        (CONTEXT_WIDTH, CONTEXT_HEIGHT),
        flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _samples_of_staff(
    staff: StaffOnPage, rng: random.Random, negatives_per_positive: float
) -> list[Anchor]:
    """The anchors and the negatives, which are far enough away from all anchors."""
    result = list(staff.anchors)
    labeled = [a.x for a in staff.anchors]
    limit = MIN_NEGATIVE_DISTANCE * staff.unit
    candidates = list(staff.negatives) + [
        rng.uniform(staff.left, staff.right) for _ in range(max(2, len(staff.negatives) // 4))
    ]
    candidates = [x for x in candidates if all(abs(x - a) > limit for a in labeled)]
    rng.shuffle(candidates)
    count = max(2, round(negatives_per_positive * max(1, len(staff.anchors))))
    result.extend(Anchor(x, KIND_NEGATIVE) for x in candidates[:count])
    return result


def crops_of_page(
    page: NDArray, staffs: list[StaffOnPage], rng: random.Random, negatives_per_positive: float
) -> tuple[list[NDArray], list[tuple[int, int, int, int]]]:
    images = []
    labels = []
    for staff in staffs:
        for anchor in _samples_of_staff(staff, rng, negatives_per_positive):
            images.append(context_crop(page, anchor.x, staff.top, staff.unit))
            labels.append((anchor.kind, anchor.numerator, anchor.denominator, anchor.count))
    return images, labels


# --- Verovio ---------------------------------------------------------------------------


def _read_score(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        name = next(
            n
            for n in z.namelist()
            if n.endswith((".xml", ".musicxml")) and not n.startswith("META-INF")
        )
        return z.read(name).decode("utf-8")


def _random_time(rng: random.Random) -> tuple[int, int]:
    denominator = rng.choices(DENOMINATORS, weights=[1, 20, 30, 25, 6, 2])[0]
    if rng.random() < 0.85:  # noqa: PLR2004
        numerator = rng.choices(
            list(range(1, 17)),
            weights=[3, 10, 10, 8, 5, 8, 4, 3, 5, 2, 2, 5, 1, 1, 1, 1],
        )[0]
    else:
        numerator = rng.randint(1, 32)
    return numerator, denominator


def _time_element(numerator: int, denominator: int, rng: random.Random) -> ET.Element:
    time = ET.Element("time")
    if (numerator, denominator) == (4, 4) and rng.random() < 0.5:  # noqa: PLR2004
        time.set("symbol", "common")
    elif (numerator, denominator) == (2, 2) and rng.random() < 0.5:  # noqa: PLR2004
        time.set("symbol", "cut")
    ET.SubElement(time, "beats").text = str(numerator)
    ET.SubElement(time, "beat-type").text = str(denominator)
    return time


def _attributes(measure: ET.Element) -> ET.Element:
    attributes = measure.find("attributes")
    if attributes is None:
        attributes = ET.Element("attributes")
        measure.insert(0, attributes)
    return attributes


# The order of the children of attributes in MusicXML
_ATTRIBUTE_ORDER = [
    "footnote",
    "level",
    "divisions",
    "key",
    "time",
    "staves",
    "part-symbol",
    "instruments",
    "clef",
    "staff-details",
    "transpose",
    "directive",
    "measure-style",
]


def _sort_attributes(attributes: ET.Element) -> None:
    children = list(attributes)
    children.sort(key=lambda c: _ATTRIBUTE_ORDER.index(c.tag) if c.tag in _ATTRIBUTE_ORDER else 99)
    for child in list(attributes):
        attributes.remove(child)
    attributes.extend(children)


def _set_time(measure: ET.Element, numerator: int, denominator: int, rng: random.Random) -> None:
    attributes = _attributes(measure)
    for time in attributes.findall("time"):
        attributes.remove(time)
    attributes.append(_time_element(numerator, denominator, rng))
    _sort_attributes(attributes)


def _rest_measure(
    number: str, divisions: int, length: float, staves: int, multirest: int | None
) -> ET.Element:
    measure = ET.Element("measure", number=number)
    if multirest is not None:
        attributes = ET.SubElement(measure, "attributes")
        style = ET.SubElement(attributes, "measure-style")
        ET.SubElement(style, "multiple-rest").text = str(multirest)
    duration = str(max(1, round(divisions * length)))
    for staff in range(1, staves + 1):
        if staff > 1:
            backup = ET.SubElement(measure, "backup")
            ET.SubElement(backup, "duration").text = duration
        note = ET.SubElement(measure, "note")
        ET.SubElement(note, "rest", measure="yes")
        ET.SubElement(note, "duration").text = duration
        ET.SubElement(note, "voice").text = str(1 + 4 * (staff - 1))
        if staves > 1:
            ET.SubElement(note, "staff").text = str(staff)
    return measure


def modify_score(xml: str, rng: random.Random) -> str:
    """
    Changes the time signatures, adds time signature changes and multi-measure rests. All
    parts get the same changes in the same measures.
    """
    root = ET.fromstring(xml.encode("utf-8"))  # noqa: S314
    parts = root.findall("part")
    if not parts:
        return xml
    count = min(len(p.findall("measure")) for p in parts)
    times: dict[int, tuple[int, int]] = {}
    multirests: dict[int, int] = {}
    for index in range(count):
        has_time = parts[0].findall("measure")[index].find("attributes/time") is not None
        if has_time and rng.random() < 0.8 or index > 0 and rng.random() < 0.08:  # noqa: PLR2004
            times[index] = _random_time(rng)
        elif index > 0 and rng.random() < 0.04:  # noqa: PLR2004
            roll = rng.random()
            if roll < 0.5:  # noqa: PLR2004
                multirests[index] = rng.randint(2, 9)
            elif roll < 0.8:  # noqa: PLR2004
                multirests[index] = rng.randint(10, 30)
            elif roll < 0.95:  # noqa: PLR2004
                multirests[index] = rng.randint(31, 99)
            else:
                multirests[index] = rng.randint(100, 199)
    for part in parts:
        divisions = 1
        staves = 1
        length = 1.0
        measures = part.findall("measure")[:count]
        for index, measure in enumerate(measures):
            attributes = measure.find("attributes")
            if attributes is not None:
                divisions = int(attributes.findtext("divisions", str(divisions)))
                staves = int(attributes.findtext("staves", str(staves)))
            if index in times:
                _set_time(measure, *times[index], rng)
            time = measure.find("attributes/time")
            if time is not None and time.findtext("beats", "").isdigit():
                length = 4 * int(time.findtext("beats", "4")) / int(time.findtext("beat-type", "4"))
            if index not in multirests:
                continue
            number = multirests[index]
            # The first measure keeps its attributes and gets the multirest, the others
            # follow as empty measures
            for child in list(measure):
                if child.tag not in ("attributes", "print"):
                    measure.remove(child)
            rest = _rest_measure(measure.get("number", ""), divisions, length, staves, number)
            attributes = _attributes(measure)
            style = rest.find("attributes")
            if style is not None:
                attributes.extend(list(style))
            _sort_attributes(attributes)
            for child in rest:
                if child.tag != "attributes":
                    measure.append(child)
            position = list(part).index(measure)
            for i in range(1, number):
                extra = _rest_measure(
                    f"{measure.get('number', '')}r{i}", divisions, length, staves, None
                )
                part.insert(position + i, extra)
    return ET.tostring(root, encoding="unicode")


def _class(element: ET.Element) -> str:
    return (element.get("class") or "").split(" ")[0]


def _is_bbox(element: ET.Element) -> bool:
    return "bounding-box" in (element.get("class") or "")


def _own_box(element: ET.Element) -> tuple[float, float, float, float] | None:
    for child in element:
        if child.tag == _SVG + "g" and _is_bbox(child):
            rect = child.find(_SVG + "rect")
            if rect is not None:
                return (
                    float(rect.get("x", 0)),
                    float(rect.get("y", 0)),
                    float(rect.get("width", 0)),
                    float(rect.get("height", 0)),
                )
    return None


def _glyphs(element: ET.Element) -> list[tuple[str, float, float]]:
    """(SMuFL code, x, y) of the glyphs in the element."""
    result = []
    for use in element.iter(_SVG + "use"):
        href = use.get("{http://www.w3.org/1999/xlink}href", use.get("href", ""))
        match = re.search(r"translate\(([\d.-]+),\s*([\d.-]+)", use.get("transform", ""))
        if match:
            result.append((href.lstrip("#").split("-")[0], float(match[1]), float(match[2])))
    return result


def _number(glyphs: list[tuple[str, float, float]]) -> int | None:
    digits = [(x, _DIGIT.fullmatch(code)) for code, x, _ in glyphs]
    if not digits or any(d is None for _, d in digits):
        return None
    return int("".join(d[1] for _, d in sorted(digits, key=lambda d: d[0])))  # type: ignore[index]


def _time_signature(
    glyphs: list[tuple[str, float, float]], middle: float
) -> tuple[int, int] | None:
    codes = [g[0] for g in glyphs]
    if codes == [_COMMON]:
        return 4, 4
    if codes == [_CUT]:
        return 2, 2
    numerator = _number([g for g in glyphs if g[2] < middle])
    denominator = _number([g for g in glyphs if g[2] > middle])
    if numerator is None or denominator not in DENOMINATORS or numerator > 99:  # noqa: PLR2004
        return None
    return numerator, denominator


def _staff_lines(staff: ET.Element) -> tuple[float, float, float, float] | None:
    lines = [
        [float(n) for n in re.findall(r"[\d.-]+", path.get("d", ""))]
        for path in staff
        if path.tag == _SVG + "path"
    ][:5]
    if len(lines) < 5 or any(len(line) < 4 for line in lines):  # noqa: PLR2004
        return None
    ys = sorted(line[1] for line in lines)
    return min(line[0] for line in lines), max(line[2] for line in lines), ys[0], ys[-1]


def parse_verovio_page(svg: str, k: float) -> list[StaffOnPage]:
    """Staffs per system and measure in raster pixels, k raster pixels per SVG unit."""
    margin = re.search(r'class="page-margin" transform="translate\(([\d.-]+), ([\d.-]+)\)', svg)
    if margin is None:
        return []
    mx, my = float(margin[1]), float(margin[2])
    staffs: list[StaffOnPage] = []
    for system in ET.fromstring(svg).iter(_SVG + "g"):  # noqa: S314
        if _class(system) != "system":
            continue
        by_index: dict[int, StaffOnPage] = {}
        for measure in (g for g in system if _class(g) == "measure"):
            for n, staff in enumerate(g for g in measure if _class(g) == "staff"):
                lines = _staff_lines(staff)
                if lines is None:
                    continue
                left, right, top, bottom = lines
                unit = (bottom - top) / 4
                middle = (top + bottom) / 2
                if n not in by_index:
                    by_index[n] = StaffOnPage((top + my) * k, unit * k, (left + mx) * k, 0, [], [])
                result = by_index[n]
                result.right = (right + mx) * k
                for element in staff.iter(_SVG + "g"):
                    cls = _class(element)
                    if _is_bbox(element):
                        continue
                    box = _own_box(element)
                    if box is None:
                        continue
                    x = (box[0] + box[2] / 2 + mx) * k
                    if cls == "meterSig":
                        time = _time_signature(_glyphs(element), middle)
                        if time is None:
                            # Unknown, it's neither a positive nor a negative
                            result.anchors.append(Anchor(x, KIND_TIME_SIGNATURE, IGNORE, IGNORE))
                        else:
                            result.anchors.append(Anchor(x, KIND_TIME_SIGNATURE, *time, IGNORE))
                    elif cls == "multiRest":
                        number = _number([g for g in _glyphs(element) if g[2] < top])
                        result.anchors.append(
                            Anchor(
                                x,
                                KIND_MULTIREST,
                                IGNORE,
                                IGNORE,
                                number if number is not None else IGNORE,
                            )
                        )
                    elif cls in _NEGATIVE_CLASSES:
                        result.negatives.append(x)
            for barline in (g for g in measure if _class(g) == "barLine"):
                box = _own_box(barline)
                if box is not None:
                    for staff_on_page in by_index.values():
                        staff_on_page.negatives.append((box[0] + box[2] / 2 + mx) * k)
        staffs.extend(by_index[n] for n in sorted(by_index))
    return staffs


def _render_pages(
    xml: str, rng: random.Random, max_pages: int
) -> list[tuple[NDArray, list[StaffOnPage]]]:
    import verovio  # noqa: PLC0415

    toolkit = verovio.toolkit()
    toolkit.setOptions(
        {
            "pageWidth": 2100,
            "pageHeight": 2970,
            "scale": rng.choice([30, 35, 40, 45, 50]),
            "font": rng.choice(FONTS),
            "header": "none",
            "footer": "none",
            "svgBoundingBoxes": True,
            "multiRestStyle": rng.choice(MULTIREST_STYLES),
            "spacingLinear": rng.uniform(0.15, 0.35),
            "spacingNonLinear": rng.uniform(0.45, 0.75),
        }
    )
    if not toolkit.loadData(xml):
        return []
    pages = list(range(1, toolkit.getPageCount() + 1))
    rng.shuffle(pages)
    result = []
    for page in pages[:max_pages]:
        svg = toolkit.renderToSVG(page)
        view = re.search(r'class="definition-scale"[^>]*viewBox="0 0 ([\d.]+)', svg)
        if view is None:
            continue
        staffs_svg = parse_verovio_page(svg, 1.0)
        if not staffs_svg:
            continue
        unit_svg = float(np.median([s.unit for s in staffs_svg]))
        # Raster with 9 to 20 pixels per staff space
        k = rng.uniform(9, 20) / unit_svg
        width = round(float(view[1]) * k)
        if width > 6000:  # noqa: PLR2004
            continue
        png = subprocess.run(  # noqa: S603
            ["rsvg-convert", "-b", "white", "-w", str(width)],  # noqa: S607
            input=svg.encode("utf-8"),
            capture_output=True,
            check=True,
        ).stdout
        image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        k_px = image.shape[1] / float(view[1])
        result.append((image, parse_verovio_page(svg, k_px)))
    return result


def excluded_songs() -> set[str]:
    """Songs of homr's photo test set, see training/omr_datasets/convert_photos.py."""
    if not photo_templates.exists():
        raise FileNotFoundError(
            f"{photo_templates} is missing, run training/omr_datasets/convert_photos.py once"
        )
    data = json.loads(photo_templates.read_text(encoding="utf-8"))
    return {m for t in data["templates"] for m in re.findall(r"lc\d+", t["source"])}


def is_validation(name: str) -> bool:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 20 == 0  # noqa: S324


def _verovio_job(args: tuple[str, int, int]) -> tuple[str, NDArray, NDArray] | None:
    path, seed, max_pages = args
    rng = random.Random(seed)
    try:
        xml = modify_score(_read_score(Path(path)), rng)
        pages = _render_pages(xml, rng, max_pages)
    except Exception as e:  # noqa: BLE001 - a broken score is skipped
        print("Skipped", path, type(e).__name__, str(e)[:200], flush=True)
        return None
    images: list[NDArray] = []
    labels: list[tuple[int, int, int, int]] = []
    for page, staffs in pages:
        i, lab = crops_of_page(page, staffs, rng, negatives_per_positive=1.0)
        images.extend(i)
        labels.extend(lab)
    if not images:
        return None
    return Path(path).stem, np.stack(images), np.array(labels, np.int16)


# --- DeepScores ------------------------------------------------------------------------


def _deepscores_pages(json_file: Path) -> list[tuple[str, list[StaffOnPage]]]:
    data = json.loads(json_file.read_text(encoding="utf-8"))
    names = {
        k: v["name"] for k, v in data["categories"].items() if v["annotation_set"] == "deepscores"
    }
    annotations = data["annotations"]
    result = []
    for image in data["images"]:
        boxes: list[tuple[str, list[float]]] = []
        for ann_id in image["ann_ids"]:
            ann = annotations.get(str(ann_id))
            if ann is None:
                continue
            name = next((names[c] for c in ann["cat_id"] if c in names), None)
            if name is not None:
                boxes.append((name, ann["a_bbox"]))
        staffs = []
        for _, (x0, y0, x1, y1) in (b for b in boxes if b[0] == "staff"):
            unit = (y1 - y0) / 4
            if unit < 4:  # noqa: PLR2004
                continue
            middle = (y0 + y1) / 2
            inside = [
                (name, box)
                for name, box in boxes
                if name != "staff" and y0 - 0.5 * unit <= (box[1] + box[3]) / 2 <= y1 + 0.5 * unit
            ]
            digits = [(name, box) for name, box in inside if name.startswith("timeSig")]
            staff = StaffOnPage(y0, unit, x0, x1, [], [])
            # Digits of one time signature overlap horizontally
            digits.sort(key=lambda d: d[1][0])
            groups: list[list[tuple[str, list[float]]]] = []
            for digit in digits:
                if groups and digit[1][0] <= max(d[1][2] for d in groups[-1]) + 0.3 * unit:
                    groups[-1].append(digit)
                else:
                    groups.append([digit])
            for group in groups:
                x = (min(d[1][0] for d in group) + max(d[1][2] for d in group)) / 2
                codes = [d[0] for d in group]
                time: tuple[int, int] | None = None
                if codes == ["timeSigCommon"]:
                    time = (4, 4)
                elif codes == ["timeSigCutCommon"]:
                    time = (2, 2)
                else:
                    upper = sorted(
                        (d for d in group if (d[1][1] + d[1][3]) / 2 < middle),
                        key=lambda d: d[1][0],
                    )
                    lower = sorted(
                        (d for d in group if (d[1][1] + d[1][3]) / 2 > middle),
                        key=lambda d: d[1][0],
                    )
                    if (
                        upper
                        and lower
                        and all(d[0][-1].isdigit() for d in upper + lower)
                        and len(upper) <= 2  # noqa: PLR2004
                    ):
                        numerator = int("".join(d[0][-1] for d in upper))
                        denominator = int("".join(d[0][-1] for d in lower))
                        if numerator > 0 and denominator in DENOMINATORS:
                            time = (numerator, denominator)
                if time is None:
                    staff.anchors.append(Anchor(x, KIND_TIME_SIGNATURE, IGNORE, IGNORE, IGNORE))
                else:
                    staff.anchors.append(Anchor(x, KIND_TIME_SIGNATURE, *time, IGNORE))
            for name, box in inside:
                x = (box[0] + box[2]) / 2
                if name in ("restHBar", "restHNr"):
                    # Multirests of unknown count
                    staff.anchors.append(Anchor(x, KIND_MULTIREST, IGNORE, IGNORE, IGNORE))
                elif not name.startswith(("timeSig", "staff", "ledger", "beam", "slur", "tie")):
                    staff.negatives.append(x)
            staffs.append(staff)
        result.append((image["filename"], staffs))
    return result


def _deepscores_job(
    args: tuple[str, list[StaffOnPage], int],
) -> tuple[str, NDArray, NDArray] | None:
    filename, staffs, seed = args
    page = cv2.imread(str(deepscores / "images" / filename), cv2.IMREAD_GRAYSCALE)
    if page is None:
        return None
    rng = random.Random(seed)
    images, labels = crops_of_page(page, staffs, rng, negatives_per_positive=1.0)
    # Only the time signatures: some glyphs, e.g. common time, lack an annotation and would
    # end up in negatives
    keep = [i for i, label in enumerate(labels) if label[0] == KIND_TIME_SIGNATURE]
    if not keep:
        return None
    return (
        filename,
        np.stack([images[i] for i in keep]),
        np.array([labels[i] for i in keep], np.int16),
    )


# --- Writing ---------------------------------------------------------------------------


def job_file(prefix: str, name: str, seed: int) -> Path:
    split = "val" if is_validation(name) else "train"
    return output_root / split / "jobs" / f"{prefix}_{name}_{seed}.npz"


def _run_and_save(job: tuple[Any, ...]) -> int:
    """Runs a job and saves its crops, returns their number."""
    prefix, function, args, name, seed = job
    target = job_file(prefix, name, seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Left behind if the job crashes the worker
    started = target.with_suffix(".started")
    started.touch()
    result = function(args)
    started.unlink()
    if result is None:
        np.savez(
            target,
            images=np.zeros((0, CONTEXT_HEIGHT, CONTEXT_WIDTH), np.uint8),
            labels=np.zeros((0, 4), np.int16),
        )
        return 0
    _, images, labels = result
    np.savez(target, images=images, labels=labels)
    return len(images)


def _run_jobs(jobs: list[tuple[Any, ...]], workers: int) -> None:
    """
    Runs the jobs whose file doesn't exist yet. A crashed worker, e.g. in Verovio, breaks
    the pool, the remaining jobs then run in a new one.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415
    from concurrent.futures.process import BrokenProcessPool  # noqa: PLC0415

    while True:
        todo = [job for job in jobs if not job_file(job[0], job[3], job[4]).exists()]
        print("Jobs to do", len(todo), "of", len(jobs), flush=True)
        if not todo:
            return
        crops = 0
        try:
            with ProcessPoolExecutor(workers) as executor:
                futures = {executor.submit(_run_and_save, job): job for job in todo}
                for done, future in enumerate(as_completed(futures)):
                    try:
                        crops += future.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as e:  # noqa: BLE001 - a broken score is skipped
                        job = futures[future]
                        print("Failed", job[3], type(e).__name__, str(e)[:200], flush=True)
                        # Marks it as done
                        _run_and_save((job[0], _nothing, None, job[3], job[4]))
                    if done % 100 == 0:
                        print("done", done, "of", len(todo), "crops", crops, flush=True)
            return
        except BrokenProcessPool:
            # The jobs which ran when a worker crashed are given up on
            for job in todo:
                target = job_file(job[0], job[3], job[4])
                if target.with_suffix(".started").exists() and not target.exists():
                    print("Gave up on", job[3], flush=True)
                    target.with_suffix(".started").unlink()
                    _run_and_save((job[0], _nothing, None, job[3], job[4]))


def _nothing(_: object) -> None:
    return None


def build_verovio(
    rounds: int, max_pages: int, workers: int, limit: int | None, split: str | None
) -> None:
    excluded = excluded_songs()
    paths = sorted(
        str(p)
        for p in lieder_scores.rglob("*.mxl")
        if not set(re.findall(r"lc\d+", p.stem)) & excluded
    )
    if limit:
        paths = paths[:limit]
    print("Scores", len(paths), "excluded", len(excluded))
    jobs = [
        ("verovio", _verovio_job, (p, 1000 * r + i, max_pages), Path(p).stem, 1000 * r + i)
        for r in range(rounds)
        for i, p in enumerate(paths)
    ]
    if split is not None:
        jobs = [j for j in jobs if (split == "val") == is_validation(j[3])]
    _run_jobs(jobs, workers)


def build_deepscores(workers: int, limit: int | None) -> None:
    jobs = []
    for json_name, split_seed in (("deepscores_train.json", 0), ("deepscores_test.json", 1)):
        for i, (filename, staffs) in enumerate(_deepscores_pages(deepscores / json_name)):
            seed = 10_000_000 * split_seed + i
            jobs.append(("deepscores", _deepscores_job, (filename, staffs, seed), filename, seed))
    if limit:
        jobs = jobs[:limit]
    _run_jobs(jobs, workers)


def write_examples(path: str, count: int = 200) -> None:
    """A sheet of context crops with their labels, to check the data by eye."""
    files = sorted((output_root / "train").rglob("*.npz"))
    data = np.load(files[0])
    images, labels = data["images"], data["labels"]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(images))[:count]
    tiles = []
    for i in order:
        tile = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)
        tile = cv2.copyMakeBorder(tile, 14, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        kind, num, den, cnt = labels[i]
        text = {0: "-", 1: f"{num}/{den}", 2: f"#{cnt}"}[int(kind)]
        cv2.putText(tile, text, (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
        cv2.line(tile, (tile.shape[1] // 2, 14), (tile.shape[1] // 2, 20), (255, 0, 0), 1)
        tiles.append(tile)
    columns = 20
    rows = [np.hstack(tiles[r : r + columns]) for r in range(0, len(tiles) - columns + 1, columns)]
    cv2.imwrite(path, np.vstack(rows))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["verovio", "deepscores", "all"], default="all")
    parser.add_argument("--rounds", type=int, default=2, help="Renders per Lieder score")
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Only the first n scores or pages")
    parser.add_argument("--split", choices=["train", "val"], help="Only the Lieder of a split")
    parser.add_argument("--examples", help="Only write example crops to this image")
    args = parser.parse_args()
    if args.examples:
        write_examples(args.examples)
    else:
        if args.source in ("verovio", "all"):
            build_verovio(args.rounds, args.max_pages, args.workers, args.limit, args.split)
        if args.source in ("deepscores", "all"):
            build_deepscores(args.workers, args.limit)
