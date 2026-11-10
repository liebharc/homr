# flake8: noqa: T201

"""
Creates the dataset for the staff relation model: Lieder pages rendered by MuseScore,
clean and degraded. The staffs are detected by homr and matched to the staffs of the
SVG, which give the label of each vertical pair.
Pieces of the staff structure benchmark are left out.
Usage: poetry run python -m training.staff_relations.create_dataset
"""

import argparse
import contextlib
import functools
import hashlib
import io
import json
import multiprocessing
import os
import random
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from homr.model import Staff, StaffPoint
from homr.staff_relations import (
    GRAND_STAFF,
    SAME_SYSTEM,
    SEPARATE,
    horizontal_end_width,
    horizontal_gap_width,
    horizontal_inputs,
    horizontal_pairs,
    page_channels,
    page_inputs,
    pair_crop,
    vertical_pairs,
)
from homr.type_definitions import NDArray
from validation.degradations import VARIANTS, degrade_with_points
from validation.staff_structure import default_directory as benchmark_directory
from validation.staff_structure import (
    gpu_workers,
    hidden_staffs_directory,
    lieder_directory,
    page_layout,
    page_width,
    render_svg,
)

default_directory = Path("datasets/staff_relations")
default_sources = [lieder_directory, hidden_staffs_directory]

# Points on the top and on the bottom line of a staff, to follow a curved staff
samples_per_line = 9
_validation_share = 0.1


def _hash(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def piece_of(svg: Path) -> str:
    """The piece of a page, the same for the renderings of single parts."""
    return re.sub(r"_(voice|piano|melody)$", "", svg.stem.rsplit("-", 1)[0])


def is_validation(piece: str) -> bool:
    return _hash(piece) % 100 < _validation_share * 100


def benchmark_pieces() -> set[str]:
    with open(benchmark_directory / "pages.json") as f:
        return {piece_of(Path(page["svg"])) for page in json.load(f)}


def _outline(staff: NDArray) -> NDArray:
    """Samples the top and the bottom line of a staff (corners) from left to right."""
    top_left, top_right, bottom_right, bottom_left = staff
    share = np.linspace(0, 1, samples_per_line)[:, np.newaxis]
    top = top_left + share * (top_right - top_left)
    bottom = bottom_left + share * (bottom_right - bottom_left)
    return np.concatenate([top, bottom])


def _truth(layout: list[list[NDArray]]) -> list[tuple[int, int]]:
    """System and grand staff group of each staff, from top to bottom."""
    result: list[tuple[int, int]] = []
    group = 0
    for system_index, system in enumerate(layout):
        for staffs in system:
            result.extend((system_index, group) for _ in staffs)
            group += 1
    return result


def _render_variants(directory: Path, variants_per_page: int, svg: Path) -> list[dict[str, Any]]:
    """Writes the degraded images of a page and the moved outlines of its staffs."""
    content = svg.read_text(encoding="utf-8")
    layout = page_layout(content)
    view_box = re.search(r'viewBox="0 0 ([\d.]+)', content)
    if len(layout) == 0 or view_box is None:
        return []
    page_id = f"{svg.parent.name}_{svg.stem}"
    png = directory / "images" / f"{page_id}.png"
    render_svg(svg, png)
    page = cv2.imread(str(png), cv2.IMREAD_COLOR)
    png.unlink()
    if page is None:
        return []
    scale = page_width / float(view_box.group(1))
    staffs = [staff for system in layout for staffs in system for staff in staffs]
    points = np.concatenate([_outline(staff) for staff in staffs]) * scale
    rng = random.Random(_hash(page_id))
    variants = rng.sample(sorted(set(VARIANTS) - {"clean"}), variants_per_page - 1)
    if rng.random() < 0.5:  # noqa: PLR2004
        variants.append("clean")
    else:
        variants.append(rng.choice(sorted(set(VARIANTS) - {"clean", *variants})))
    result = []
    for variant in variants:
        np_rng = np.random.default_rng(_hash(f"{page_id}/{variant}"))
        image, moved = degrade_with_points(VARIANTS[variant], page, np_rng, points)
        name = f"{page_id}_{variant}"
        path = directory / "images" / f"{name}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        result.append(
            {
                "name": name,
                "piece": piece_of(svg),
                "image": str(path),
                "outlines": moved.reshape(len(staffs), 2 * samples_per_line, 2).tolist(),
                "truth": _truth(layout),
            }
        )
    return result


def _config() -> Any:
    from homr.main import ProcessingConfig  # noqa: PLC0415

    return ProcessingConfig(
        enable_debug=False,
        enable_cache=True,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        transformer_use_gpu=False,
        segnet_use_gpu=False,
        coreml_encoder=False,
        title_detection=False,
    )


def match_staffs(staffs: list[Staff], to_input: Any, outlines: NDArray) -> list[int | None]:
    """
    The index of the true staff for each detected staff, None if it matches none. A
    detected staff matches the true staff whose center line is closest at most of its
    points, if it is within one unit size.
    """
    tops, bottoms = outlines[:, :samples_per_line], outlines[:, samples_per_line:]
    centers = (tops + bottoms) / 2
    units = np.linalg.norm(bottoms - tops, axis=2).mean(axis=1) / 4
    result: list[int | None] = []
    for staff in staffs:
        step = max(1, len(staff.grid) // 12)
        points = np.array([to_input((p.x, p.y[2])) for p in staff.grid[::step]])
        votes = []
        for x, y in points:
            distances = []
            for center in centers:
                if center[0, 0] - units[0] <= x <= center[-1, 0] + units[0]:
                    distances.append(abs(np.interp(x, center[:, 0], center[:, 1]) - y))
                else:
                    distances.append(np.inf)
            best = int(np.argmin(distances))
            if distances[best] < units[best]:
                votes.append(best)
        if len(votes) < 0.6 * len(points):  # noqa: PLR2004
            result.append(None)
        else:
            result.append(max(set(votes), key=votes.count))
    return result


def label(truth: list[tuple[int, int]], upper: int | None, lower: int | None) -> int | None:
    if upper is None or lower is None:
        return SEPARATE
    if upper == lower:
        return None
    (system_a, group_a), (system_b, group_b) = truth[upper], truth[lower]
    if system_a != system_b:
        return SEPARATE
    return GRAND_STAFF if group_a == group_b else SAME_SYSTEM


def _extract(directory: Path, task: dict[str, Any]) -> dict[str, Any]:
    from homr.main import detect_single_staffs  # noqa: PLC0415

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            single = detect_single_staffs(task["image"], _config())
    except Exception as e:
        return {"name": task["name"], "error": repr(e)[:200]}
    staffs = single.staffs
    matches = match_staffs(staffs, single.to_input_image, np.array(task["outlines"]))
    matched = len({m for m in matches if m is not None})
    status = {
        "name": task["name"],
        "detected": len(staffs),
        "matched": matched,
        "true": len(task["truth"]),
        "unmatched": sum(m is None for m in matches),
    }
    # Pages where the matching is unreliable would give wrong labels
    if matched < 0.7 * len(task["truth"]) or status["unmatched"] > 0.3 * len(
        staffs
    ):  # noqa: PLR2004
        return {**status, "skipped": True}
    if not (directory / "pairs_horizontal" / f"{task['name']}.npz").exists():
        status["horizontal"] = _write_horizontal(
            directory, task, single.predictions, staffs, matches
        )
    if not (directory / "pairs_truncated" / f"{task['name']}.npz").exists():
        status["truncated"] = _write_truncated(directory, task, single.predictions, staffs, matches)
    vertical = directory / "pairs" / f"{task['name']}.npz"
    if vertical.exists():
        return status
    pairs, crops = page_inputs(single.predictions, staffs)
    labels = [label(task["truth"], matches[i], matches[j]) for i, j in pairs]
    keep = [i for i, value in enumerate(labels) if value is not None]
    if len(keep) == 0:
        return {**status, "pairs": 0}
    np.savez_compressed(
        vertical,
        crops=(crops[keep] * 255).round().astype(np.uint8),
        labels=np.array([labels[i] for i in keep], dtype=np.int64),
    )
    return {**status, "pairs": len(keep)}


def truncate_staff(staff: Staff, rng: random.Random) -> Staff | None:
    """The staff without its first 2 to 20 units, like a staff whose start wasn't found."""
    unit = staff.average_unit_size
    cut = staff.min_x + rng.uniform(2, 20) * unit
    grid = [p for p in staff.grid if p.x >= cut]
    if len(grid) < 2 or grid[-1].x - grid[0].x < 20 * unit:  # noqa: PLR2004
        return None
    return Staff(grid)


def _write_truncated(
    directory: Path,
    task: dict[str, Any],
    predictions: Any,
    staffs: list[Staff],
    matches: list[int | None],
) -> int:
    """
    Vertical pairs where the start of one staff is missing. The staff detection misses
    the start of a staff on degraded images, the label doesn't change.
    """
    pairs = vertical_pairs(staffs)
    labels = [label(task["truth"], matches[i], matches[j]) for i, j in pairs]
    rng = random.Random(_hash(task["name"] + "/truncated"))
    candidates = [k for k, value in enumerate(labels) if value is not None]
    if len(candidates) == 0:
        return 0
    unit = float(np.median([staff.average_unit_size for staff in staffs]))
    page, scale = page_channels(predictions, unit)
    crops, kept = [], []
    for k in rng.sample(candidates, min(4, len(candidates))):
        upper, lower = staffs[pairs[k][0]], staffs[pairs[k][1]]
        if rng.random() < 0.5:  # noqa: PLR2004
            upper = truncate_staff(upper, rng)  # type: ignore[assignment]
        else:
            lower = truncate_staff(lower, rng)  # type: ignore[assignment]
        if upper is None or lower is None:
            continue
        crops.append(pair_crop(page, scale, upper, lower))
        kept.append(labels[k])
    if len(kept) == 0:
        return 0
    path = directory / "pairs_truncated" / f"{task['name']}.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        path,
        crops=(np.stack(crops) * 255).round().astype(np.uint8),
        labels=np.array(kept, dtype=np.int64),
    )
    return len(kept)


def split_staff(staff: Staff, rng: random.Random) -> tuple[Staff, Staff] | None:
    """
    Two pieces of a staff, like a staff which the line tracking lost under symbols. The
    gap is up to 150 units, as long as both pieces keep at least 15 units.
    """
    unit = staff.average_unit_size
    width = (staff.max_x - staff.min_x) / unit
    if width < 40 or len(staff.grid) < 6:  # noqa: PLR2004
        return None
    gap = rng.uniform(0, min(150, width - 30)) * unit
    start = rng.uniform(staff.min_x + 15 * unit, staff.max_x - 15 * unit - gap)
    left = [p for p in staff.grid if p.x < start]
    right = [p for p in staff.grid if p.x > start + gap]
    if len(left) < 2 or len(right) < 2:  # noqa: PLR2004
        return None
    return Staff(left), Staff(right)


def shift_staff(staff: Staff, offset: float) -> Staff:
    return Staff([StaffPoint(p.x, [y + offset for y in p.y], p.angle) for p in staff.grid])


def _write_horizontal(
    directory: Path,
    task: dict[str, Any],
    predictions: Any,
    staffs: list[Staff],
    matches: list[int | None],
) -> int:
    """
    Pairs of staffs in the same row: the pairs which homr detected, labeled by the true
    staffs, and pieces of single staffs, which are cut apart. The pieces also give
    negatives: with the gap blanked out, like the gutter between two pages, and with the
    right piece moved up or down, so that the lines don't continue.
    """
    path = directory / "pairs_horizontal" / f"{task['name']}.npz"
    pieces: list[Staff] = []
    pairs = []
    labels = []
    blank = []
    for i, j in horizontal_pairs(staffs):
        same = matches[i] is not None and matches[i] == matches[j]
        pairs.append((i, j))
        labels.append(int(same))
    rng = random.Random(_hash(task["name"]))
    matched = [i for i, m in enumerate(matches) if m is not None]
    for i in rng.sample(matched, min(3, len(matched))):
        split = split_staff(staffs[i], rng)
        if split is None:
            continue
        left, right = split
        unit = staffs[i].average_unit_size
        shifted = shift_staff(right, rng.choice([-1, 1]) * rng.uniform(1.5, 3.5) * unit)
        pieces.extend([left, right, shifted])
        first = len(staffs) + len(pieces) - 3
        pairs.extend([(first, first + 1), (first, first + 1), (first, first + 2)])
        labels.extend([1, 0, 0])
        blank.append(len(pairs) - 2)
    if len(pairs) == 0:
        return 0
    unit = float(np.median([staff.average_unit_size for staff in staffs]))
    page, scale = page_channels(predictions, unit)
    crops, feats = horizontal_inputs(page, scale, staffs + pieces, pairs)
    gap = slice(horizontal_end_width, horizontal_end_width + horizontal_gap_width)
    for k in blank:
        crops[k][:, :, gap] = 0
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        path,
        crops=(crops * 255).round().astype(np.uint8),
        features=feats,
        labels=np.array(labels, dtype=np.int64),
    )
    return len(pairs)


def create(  # noqa: PLR0913
    directory: Path,
    sources: list[Path],
    pages: int,
    variants_per_page: int,
    seed: int,
    workers: int,
) -> None:
    from validation.staff_structure import (  # noqa: PLC0415
        _init_worker,
        _segment_on_gpu,
    )

    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "pairs").mkdir(parents=True, exist_ok=True)
    excluded = benchmark_pieces()
    svgs = sorted(
        svg for source in sources for svg in source.glob("*.svg") if piece_of(svg) not in excluded
    )
    random.Random(seed).shuffle(svgs)
    svgs = svgs[:pages]
    context = multiprocessing.get_context("fork")

    tasks_file = directory / "tasks.jsonl"
    if not tasks_file.exists():
        render = functools.partial(_render_variants, directory, variants_per_page)
        with (
            context.Pool(workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool,
            open(tasks_file, "w") as f,
        ):
            for i, tasks in enumerate(pool.imap_unordered(render, svgs), 1):
                for task in tasks:
                    f.write(json.dumps(task) + "\n")
                if i % 100 == 0:
                    print(f"Rendered {i} / {len(svgs)}", flush=True)
    with open(tasks_file) as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    for task in tasks:
        task["validation"] = is_validation(task["piece"])

    uncached = [t["image"] for t in tasks if not Path(t["image"]).with_suffix(".npy").exists()]
    print(f"Segmenting {len(uncached)} images on the GPU", flush=True)
    with context.Pool(gpu_workers, initializer=_init_worker, initargs=(gpu_workers,)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_segment_on_gpu, uncached), 1):
            if i % 100 == 0:
                print(f"Segmented {i} / {len(uncached)}", flush=True)

    extract = functools.partial(_extract, directory)
    with (
        context.Pool(workers, initializer=_init_worker, initargs=(workers,)) as pool,
        open(directory / "status.jsonl", "w") as f,
    ):
        for i, status in enumerate(pool.imap_unordered(extract, tasks), 1):
            f.write(json.dumps(status) + "\n")
            if i % 100 == 0:
                print(f"Extracted {i} / {len(tasks)}", flush=True)
    split = {t["name"]: "val" if t["validation"] else "train" for t in tasks}
    (directory / "split.json").write_text(json.dumps(split))
    print(f"Wrote the pairs of {len(tasks)} images to {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=default_directory)
    parser.add_argument("--sources", type=Path, nargs="+", default=default_sources)
    parser.add_argument("--pages", type=int, default=2500)
    parser.add_argument("--variants-per-page", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    create(
        args.directory, args.sources, args.pages, args.variants_per_page, args.seed, args.workers
    )


if __name__ == "__main__":
    main()
