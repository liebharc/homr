"""
Songs engraved by Verovio and drawn onto phone photos of template pages, through the photo's map
to the flat page (photo_maps.py), the way the photo's camera drew its own print (photo_look.py).
"""

import copy
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import cairosvg
import cv2
import numpy as np
import verovio
from numpy.typing import NDArray

from training.omr_datasets.photo_look import Look, draw_ink
from training.omr_datasets.photo_maps import FH, FLAT_MM, FW, PhotoConditions

FONTS = ["Leipzig", "Bravura", "Gootville", "Leland", "Petaluma"]
SIZES = [30, 40, 55]  # small, normal, large staves (Verovio's scale)
DRAWN_AT = 40  # Verovio's scale for every size: the size sets the page Verovio lays out on
# the music is drawn at this share of the width it takes on the page: 1 is never enlarged (an
# enlarged drawing would add blur of its own); tests use less, for speed
DETAIL = 1.0
SCALE = 0.86  # the music's share of the template page width, leaving room for the marks
MARK = round(10 * FLAT_MM)
MARK_CENTER = 10 * FLAT_MM
# what an <attributes> element carries on to later bars, in the order MusicXML wants them
CARRIED = ["divisions", "key", "time", "staves", "part-symbol", "instruments", "clef", "transpose"]


@dataclass
class EngravedPage:
    """A song on the flat template page, the MusicXML of the bars on it, and per system the
    number of bars and each staff's box (left, top, right, bottom of its lines) in page pixels."""

    flat: NDArray
    musicxml: str
    systems: list[dict[str, Any]]


def bar_count(xml: str) -> int:
    """The number of bars of the score's first part."""
    part = ET.fromstring(xml.encode("utf-8")).find("part")  # noqa: S314
    return len(part.findall("measure")) if part is not None else 0


def measures_from(xml: str, start: int, count: int) -> str:
    """The score with only `count` bars from bar `start` (0 = the first) in every part. The clef,
    key, time and the rest in force at `start` are put at the front of its first bar, so the
    excerpt reads as it sounds there."""
    root = ET.fromstring(xml.encode("utf-8"))  # noqa: S314
    for part in root.findall("part"):
        measures = part.findall("measure")
        if 0 < start < len(measures):
            in_force: dict[tuple[str, str | None], ET.Element] = {}
            for measure in measures[:start]:
                for attributes in measure.findall("attributes"):
                    for child in attributes:
                        in_force[child.tag, child.get("number")] = child
            carried = ET.Element("attributes")
            for tag in CARRIED:
                for (name, _), child in sorted(in_force.items(), key=lambda kv: str(kv[0][1])):
                    if name == tag:
                        carried.append(copy.deepcopy(child))
            measures[start].insert(0, carried)
        for measure in measures[:start] + measures[start + count :]:
            part.remove(measure)
    return ET.tostring(root, encoding="unicode")


def layout(svg: str, zoom: float, f: float, x0: float, y0: float) -> list[dict[str, Any]]:
    """Per system: the number of bars and each staff's box in flat pixels. Verovio draws in its
    own units: the page is the viewBox, the music is shifted by the page margin."""
    width = float(re.search(r'<svg width="([\d.]+)px"', svg).group(1))  # type: ignore[union-attr]
    view = float(
        re.search(r'class="definition-scale"[^>]*viewBox="0 0 ([\d.]+)', svg).group(1)  # type: ignore[union-attr]
    )
    margin = re.search(  # type: ignore[union-attr]
        r'class="page-margin" transform="translate\(([\d.-]+), ([\d.-]+)\)', svg
    ).groups()
    mx, my = float(margin[0]), float(margin[1])
    k = width / view * zoom * f

    def flat(x: float, y: float) -> tuple[float, float]:
        return x0 + (x + mx) * k, y0 + (y + my) * k

    systems = []
    for system in ET.fromstring(svg).iter("{http://www.w3.org/2000/svg}g"):  # noqa: S314
        if system.get("class") != "system":
            continue
        measures = [g for g in system if g.get("class", "").startswith("measure")]
        boxes: dict[int, list[float]] = {}
        for measure in measures:
            staffs = [g for g in measure if g.get("class") == "staff"]
            for n, staff in enumerate(staffs):
                lines = [
                    list(map(float, re.findall(r"[\d.-]+", p.get("d", ""))))
                    for p in staff
                    if p.tag.endswith("path")
                ][:5]
                left, top = flat(min(a[0] for a in lines), min(a[1] for a in lines))
                right, bottom = flat(max(a[2] for a in lines), max(a[1] for a in lines))
                box = boxes.get(n, [left, top, right, bottom])
                boxes[n] = [
                    min(box[0], left),
                    min(box[1], top),
                    max(box[2], right),
                    max(box[3], bottom),
                ]
        systems.append({"bars": len(measures), "staffs": [boxes[n] for n in sorted(boxes)]})
    return systems


def engrave_on_template(xml: str, font: str, size: int, start: int = 0) -> EngravedPage | None:
    """The most bars from bar `start` (0 = the first) that fit on one page, placed as on a
    template page (without the marks: the photo's map is in template pixels). None when fewer
    than 4 bars fit. A bigger size lays the music out on a smaller page, so the staves come out
    bigger once the page is fitted to the template (Verovio's scale only scales the drawing)."""
    k = 40 / size
    page_width = round(2100 * k)
    options = {
        "pageWidth": page_width,
        "pageHeight": round(2970 * k),
        "scale": DRAWN_AT,
        "adjustPageHeight": False,
        "footer": "none",
        "font": font,
        "pageMarginLeft": round(150 * k),
        "pageMarginRight": round(150 * k),
        "header": "none",
    }

    def toolkit(count: int) -> verovio.toolkit:
        tk = verovio.toolkit()
        tk.setOptions(options)
        tk.loadData(measures_from(xml, start, count))
        return tk

    low, high = 1, bar_count(xml) - start
    while low < high:
        mid = (low + high + 1) // 2
        if toolkit(mid).getPageCount() == 1:
            low = mid
        else:
            high = mid - 1
    if low < 4:
        return None
    svg = toolkit(low).renderToSVG(1)
    # drawn at least as wide as it sits on the page: enlarging it afterwards would blur it
    zoom = DETAIL * SCALE * FW / (page_width * DRAWN_AT / 100)
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), background_color="white", scale=zoom)
    music = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    if music is None:
        return None
    f = min(SCALE * FW / music.shape[1], (FH - 2 * (MARK_CENTER + MARK)) / music.shape[0])
    small = cv2.resize(
        music,
        (round(music.shape[1] * f), round(music.shape[0] * f)),
        interpolation=cv2.INTER_AREA,
    )
    flat = np.full((FH, FW), 255, np.uint8)
    y0, x0 = (FH - small.shape[0]) // 2, (FW - small.shape[1]) // 2
    flat[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
    return EngravedPage(flat, measures_from(xml, start, low), layout(svg, zoom, f, x0, y0))


def put_on_photo(photo: NDArray, cond: PhotoConditions, look: Look, flat: NDArray) -> NDArray:
    """The flat page drawn onto the photo's paper as its camera drew ink there (photo_look's
    draw_ink, the drawing the photo's look was checked with). Pixels off the page stay as they
    are."""
    drawn = draw_ink(flat, (cond.map_x, cond.map_y), look)
    made = np.clip(cond.paper * (1 - drawn[..., None]), 0, 255).astype(np.uint8)
    on = cond.on_page & (cond.map_x >= 0)
    result = photo.copy()
    result[on] = made[on]
    return result
