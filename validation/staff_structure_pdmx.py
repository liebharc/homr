# flake8: noqa: T201

"""
Staff structure benchmark on PDMX scores, for the layouts which Lieder doesn't have:
single parts, quartets, choirs, ensembles. A development set: its scores must not be
used for training. The expected layout comes from the parts of the MusicXML, a part with
two staffs is a grand staff.
Usage:
    poetry run python -m validation.staff_structure_pdmx prepare
    poetry run python -m validation.staff_structure run --directory datasets/staff_structure_pdmx
"""

import argparse
import csv
import functools
import json
import multiprocessing
import os
import random
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from validation.degradations import set_background_photos
from validation.staff_structure import (
    _write_variants,
    musescore,
    resolve_variants,
    staffs_per_system,
)

default_directory = Path("datasets/staff_structure_pdmx")
pdmx_root = Path("datasets/pdmx")
# Scores per number of parts
scores_per_parts = {"1": 60, "2": 80, "3": 50, "4": 70, "5+": 60}
max_complexity = 2


def _parts_group(parts: int) -> str:
    return "5+" if parts >= 5 else str(parts)  # noqa: PLR2004


def select_scores(seed: int) -> list[Path]:
    """Random PDMX scores without license conflicts, by number of parts."""
    by_group: dict[str, list[Path]] = {group: [] for group in scores_per_parts}
    with open(pdmx_root / "PDMX.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["subset:no_license_conflict"] != "True" or row["subset:all_valid"] != "True":
                continue
            try:
                parts, complexity = int(row["n_tracks"]), int(row["complexity"])
            except ValueError:
                continue
            if complexity <= max_complexity and parts >= 1:
                by_group[_parts_group(parts)].append(pdmx_root / row["mxl"].lstrip("./"))
    rng = random.Random(seed)
    result = []
    for group, count in scores_per_parts.items():
        candidates = sorted(by_group[group])
        rng.shuffle(candidates)
        result.extend(candidates[:count])
    return result


def _render(scores: list[Path], directory: Path) -> None:
    """Renders each score to SVG pages and MusicXML with MuseScore."""
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for score in scores:
        target = directory / score.name
        if any(directory.glob(f"{score.stem}-*.svg")):
            continue
        shutil.copyfile(score, target)
        jobs.append(
            {
                "in": str(target),
                "out": [str(target.with_suffix(".svg")), str(target.with_suffix(".musicxml"))],
            }
        )
    print(f"Rendering {len(jobs)} scores with MuseScore", flush=True)
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "QT_QUICK_BACKEND": "software",
        "MU_QT_QPA_PLATFORM": "offscreen",
    }
    for start in range(0, len(jobs), 25):
        job_file = directory / "jobs.json"
        job_file.write_text(json.dumps(jobs[start : start + 25]))
        result = subprocess.run(  # noqa: S603
            [str(musescore), "--force", "-j", str(job_file)],
            env=env,
            check=False,
            capture_output=True,
            timeout=1800,
        )
        if result.returncode != 0:
            print(f"MuseScore failed on jobs {start} to {start + 25}", flush=True)
        job_file.unlink()


def expected_layout(musicxml: str) -> list[bool]:
    """The staffs of a system from top to bottom, True for the upper staff of a grand staff."""
    layout = []
    for part in re.findall(r"<part id=.*?</part>", musicxml, re.DOTALL):
        staves = int(m.group(1)) if (m := re.search(r"<staves>(\d+)</staves>", part)) else 1
        layout.extend([True] if staves == 2 else [False] * staves)  # noqa: PLR2004
    return layout


def _select_pages(directory: Path, scores: list[Path], seed: int) -> list[dict[str, Any]]:
    """One page of each score whose systems show all staffs, so that the parts match."""
    rng = random.Random(seed)
    pages = []
    for score in scores:
        musicxml = directory / f"{score.stem}.musicxml"
        if not musicxml.exists():
            continue
        layout = expected_layout(musicxml.read_text(encoding="utf-8"))
        staffs = sum(2 if grand else 1 for grand in layout)
        svgs = sorted(directory.glob(f"{score.stem}-*.svg"))
        rng.shuffle(svgs)
        for svg in svgs:
            systems = staffs_per_system(svg.read_text(encoding="utf-8"))
            if len(systems) > 0 and all(n == staffs for n in systems):
                pages.append(
                    {
                        "id": svg.stem,
                        "svg": str(svg),
                        "systems": systems,
                        "layout": layout,
                        "parts": _parts_group(len(layout)),
                    }
                )
                break
    return sorted(pages, key=lambda page: page["id"])


def prepare(directory: Path, seed: int, variants: list[str], workers: int) -> None:
    scores = select_scores(seed)
    _render(scores, directory / "scores")
    pages = _select_pages(directory / "scores", scores, seed)
    for variant in variants:
        (directory / variant).mkdir(parents=True, exist_ok=True)
    write = functools.partial(_write_variants, directory, variants, False)
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(write, pages), 1):
            if i % 25 == 0:
                print(f"{i} / {len(pages)}", flush=True)
    with open(directory / "pages.json", "w") as f:
        json.dump(pages, f, indent=1)
    with open(directory / "scores.txt", "w") as f:
        f.write("".join(f"{score}\n" for score in scores))
    print(f"Wrote {len(pages)} pages to {directory}: {dict(Counter(p['parts'] for p in pages))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--directory", type=Path, default=default_directory)
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--workers", type=int, default=os.cpu_count())
    prepare_parser.add_argument("--variants", nargs="*")
    prepare_parser.add_argument("--backgrounds", type=Path)
    args = parser.parse_args()
    set_background_photos(args.backgrounds)
    prepare(args.directory, args.seed, resolve_variants(args.variants), args.workers)


if __name__ == "__main__":
    main()
