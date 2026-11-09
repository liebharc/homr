"""
Songs engraved by Verovio and drawn onto phone photos of template pages, through the photo's map
to the flat page (photo_maps.py), as dark and as blurred as the photo's own ink.
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

from training.omr_datasets.photo_maps import (
    FH,
    FLAT_MM,
    FW,
    PhotoConditions,
    edge_sharpness,
)

FONTS = ["Leipzig", "Bravura", "Gootville", "Leland", "Petaluma"]
SIZES = [30, 40, 55]  # Verovio scale: small, normal, large staves
SUPER = 2  # draw the page at twice the photo's size, then shrink: what a camera does
BLEED = 0.25  # printer ink spreads: strokes are widened by this many 300 dpi pixels each side
BLURS = [0, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]  # sigmas tried to match the photo's ink edges
ZOOM = 1.5  # Verovio's page drawn at this zoom
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
    than 4 bars fit."""
    options = {
        "pageWidth": 2100,
        "pageHeight": 2970,
        "scale": size,
        "adjustPageHeight": False,
        "footer": "none",
        "font": font,
        "pageMarginLeft": 150,
        "pageMarginRight": 150,
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
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), background_color="white", scale=ZOOM)
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
    return EngravedPage(flat, measures_from(xml, start, low), layout(svg, ZOOM, f, x0, y0))


def ink_strength(darkness: NDArray, page: NDArray, level: float) -> float:
    """How much a fully dark pixel darkens the paper, so the core of the drawn ink is as dark as
    the photo's (level, a share of the paper). Thin strokes never reach full darkness once shrunk
    and blurred."""
    strength = 1 - level
    for _ in range(5):
        core = (1 - strength * darkness < 0.6) & page
        if not core.any():
            break
        strength = min((1 - level) / max(float(np.median(darkness[core])), 1e-3), 1.0)
    return strength


def put_on_photo(
    photo: NDArray, cond: PhotoConditions, flat: NDArray, bleed: float = BLEED
) -> NDArray:
    """The flat page drawn onto the photo's paper, its ink widened by the printer's bleed and
    blurred as much as the photo's own ink. Pixels off the page stay as they are."""
    if bleed:  # a fraction blends the two whole widths around it
        low = int(bleed)
        wide = [
            (
                flat
                if r == 0
                else cv2.erode(
                    flat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                )
            )
            for r in (low, low + 1)
        ]
        part = bleed - low
        flat = np.asarray(
            wide[0] if part == 0 else cv2.addWeighted(wide[0], 1 - part, wide[1], part, 0)
        )
    h, w = cond.map_x.shape
    big = (w * SUPER, h * SUPER)
    big_x = cv2.resize(cond.map_x, big, interpolation=cv2.INTER_LINEAR)
    big_y = cv2.resize(cond.map_y, big, interpolation=cv2.INTER_LINEAR)
    lit = cv2.remap(flat, big_x, big_y, cv2.INTER_LINEAR, borderValue=255)
    darkness = 1 - cv2.resize(lit, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    tried: list[tuple[float, float, NDArray, float]] = []
    for s in BLURS:
        d = np.asarray(cv2.GaussianBlur(darkness, (0, 0), s) if s else darkness)
        strength = ink_strength(d, cond.on_page, cond.ink_level)
        off = abs(edge_sharpness(1 - d * strength, cond.on_page) - cond.sharp)
        tried.append((off, s, d, strength))
    _, _, blurred, strength = min(tried, key=lambda t: t[:2])
    made = cond.paper * (1 - blurred[..., None] * strength)
    result = photo.copy()
    result[cond.on_page] = np.clip(made, 0, 255).astype(np.uint8)[cond.on_page]
    return result
