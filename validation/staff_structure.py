# flake8: noqa: T201

"""
Staff structure benchmark: checks the staffs of every system and that no staff was
dropped, on clean and degraded Lieder pages rendered by MuseScore.
Usage: see Benchmark.md.
"""

import argparse
import contextlib
import functools
import hashlib
import io
import itertools
import json
import lzma
import multiprocessing
import os
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from validation.degradations import GROUPS, VARIANTS, set_background_photos

default_directory = Path("datasets/staff_structure_lieder")
lieder_directory = Path("datasets/Lieder-main/flat")
lieder_scores = Path("datasets/Lieder-main/scores")
# Scores rendered with their own setting to hide empty staffs, for layout changes
hidden_staffs_directory = Path("datasets/Lieder-main/hidden_staffs")
musescore = Path("datasets/MuseScore")

# Pages per layout group, by staffs per system or "change" if systems differ.
# Lieder has no pieces with a single staff.
pages_per_layout = {"2": 15, "3": 30, "4+": 20, "change": 20}
pages_per_piece = 2

# Width of an A4 page at 300 dpi
page_width = 2480

CORRECT = "correct"


def _layout_group(systems: list[int]) -> str:
    if len(set(systems)) > 1:
        return "change"
    return "4+" if systems[0] >= 4 else str(systems[0])


def expected_staffs_per_system(musicxml: str) -> int:
    """Sums the staffs of all parts, a part without <staves> has one staff."""
    parts = re.findall(r"<part id=.*?</part>", musicxml, re.DOTALL)
    return sum(
        int(m.group(1)) if (m := re.search(r"<staves>(\d+)</staves>", part)) else 1
        for part in parts
    )


def _polylines(svg: str, class_name: str) -> list[list[tuple[float, float]]]:
    pattern = rf'<polyline class="{class_name}"[^>]*?points="([^"]+)"'
    return [
        [(float(x), float(y)) for x, y in (point.split(",") for point in m.group(1).split())]
        for m in re.finditer(pattern, svg)
    ]


def staffs_per_system(svg: str) -> list[int]:
    """
    Reads the staffs per system from a MuseScore SVG. MuseScore draws each staff line as
    a polyline and the bar line at the start of a system from each staff to the next.
    """
    lines = _polylines(svg, "StaffLines")
    if len(lines) == 0 or len(lines) % 5 != 0:
        return []
    # Left end, top and bottom of each staff
    staffs = sorted(
        ((lines[i][0][0], lines[i][0][1], lines[i + 4][0][1]) for i in range(0, len(lines), 5)),
        key=lambda staff: staff[1],
    )
    bar_lines = [
        (line[0][0], min(y for _, y in line), max(y for _, y in line))
        for line in _polylines(svg, "BarLine")
    ]
    systems = [1]
    for (x, top, bottom), (next_x, next_top, _) in itertools.pairwise(staffs):
        tolerance = (bottom - top) / 4
        joined = abs(x - next_x) < tolerance and any(
            abs(bar_x - x) < tolerance
            and bar_top < top + tolerance
            and bar_bottom > next_top - tolerance
            for bar_x, bar_top, bar_bottom in bar_lines
        )
        if joined:
            systems[-1] += 1
        else:
            systems.append(1)
    return systems


def resolve_variants(names: list[str] | None) -> list[str]:
    """
    Resolves variant and group names. Always includes "clean", the reference.
    """
    if not names:
        return list(VARIANTS)
    result = ["clean"]
    for name in names:
        if name in GROUPS:
            result.extend(GROUPS[name])
        elif name in VARIANTS:
            result.append(name)
        else:
            choices = ", ".join([*GROUPS, *VARIANTS])
            raise ValueError(f"Unknown variant or group {name}, choose from {choices}")
    return list(dict.fromkeys(result))


def _seed(page_id: str, variant: str) -> int:
    return int(hashlib.sha256(f"{page_id}/{variant}".encode()).hexdigest()[:8], 16)


def _render_hidden_staffs() -> None:
    """Renders the scores which hide empty staffs to SVG, as MuseScore lays them out."""
    hidden_staffs_directory.mkdir(exist_ok=True)
    jobs = []
    for score in sorted(lieder_scores.rglob("*.mscx")):
        target = hidden_staffs_directory / score.name
        rendered = any(hidden_staffs_directory.glob(f"{score.stem}-*.svg"))
        if rendered or "<hideEmptyStaves>1</hideEmptyStaves>" not in score.read_text(
            encoding="utf-8"
        ):
            continue
        shutil.copyfile(score, target)
        jobs.append({"in": str(target), "out": str(target.with_suffix(".svg"))})
    if len(jobs) == 0:
        return
    print(f"Rendering {len(jobs)} scores with MuseScore", flush=True)
    job_file = hidden_staffs_directory / "jobs.json"
    job_file.write_text(json.dumps(jobs))
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "QT_QUICK_BACKEND": "software"}
    subprocess.run(  # noqa: S603
        [str(musescore), "--force", "-j", str(job_file)],
        env=env,
        check=True,
        capture_output=True,
    )
    job_file.unlink()


def _svgs_by_piece(directory: Path) -> dict[str, list[str]]:
    svgs: dict[str, list[str]] = defaultdict(list)
    for name in sorted(os.listdir(directory)):
        if name.endswith(".svg"):
            svgs[name.rsplit("-", 1)[0]].append(name)
    return svgs


def _select_pages(seed: int) -> list[dict[str, Any]]:
    """
    Picks random Lieder pages, at most pages_per_piece per piece. Pages with a constant
    layout come from the SVGs with all staffs visible, their layout must match the
    staffs of the musicxml. Pages with a layout change come from the scores which hide
    empty staffs.
    """
    rng = random.Random(seed)
    visible = _svgs_by_piece(lieder_directory)
    hidden = _svgs_by_piece(hidden_staffs_directory)
    pieces = sorted(
        name.removesuffix(".musicxml")
        for name in os.listdir(lieder_directory)
        if name.endswith(".musicxml")
    )
    rng.shuffle(pieces)
    for by_piece in (visible, hidden):
        for piece in pieces:
            rng.shuffle(by_piece[piece])
    expected_by_piece: dict[str, int] = {}
    missing = dict(pages_per_layout)
    pages: list[dict[str, Any]] = []
    for round_index in range(pages_per_piece):
        for piece in pieces:
            if piece not in expected_by_piece:
                musicxml = (lieder_directory / f"{piece}.musicxml").read_text(encoding="utf-8")
                expected_by_piece[piece] = expected_staffs_per_system(musicxml)
            expected = expected_by_piece[piece]
            sources = [
                (lieder_directory, visible[piece], "", _layout_group([expected])),
                (hidden_staffs_directory, hidden[piece], "-hidden", "change"),
            ]
            for directory, svgs, suffix, group in sources:
                if not missing.get(group) or len(svgs) <= round_index:
                    continue
                svg = directory / svgs[round_index]
                systems = staffs_per_system(svg.read_text(encoding="utf-8"))
                if len(systems) == 0 or _layout_group(systems) != group or max(systems) > expected:
                    continue
                if group != "change" and systems[0] != expected:
                    continue
                missing[group] -= 1
                pages.append({"id": svg.stem + suffix, "svg": str(svg), "systems": systems})
    return sorted(pages, key=lambda page: page["id"])


def _render(svg: Path, png: Path) -> None:
    command = ["rsvg-convert", "-w", str(page_width), "-b", "white", "-o", str(png), str(svg)]
    subprocess.run(command, check=True)  # noqa: S603


def _write_variants(
    directory: Path, variants: list[str], overwrite: bool, page: dict[str, Any]
) -> None:
    page_id = page["id"]
    clean = directory / "clean" / f"{page_id}.png"
    if not clean.exists():
        _render(Path(page["svg"]), clean)
    image = cv2.imread(str(clean), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read {clean}")
    for variant in variants:
        path = directory / variant / f"{page_id}.png"
        if path.exists() and (not overwrite or variant == "clean"):
            continue
        rng = np.random.default_rng(_seed(page_id, variant))
        cv2.imwrite(str(path), VARIANTS[variant](image, rng))
        path.with_suffix(".npy").unlink(missing_ok=True)


def prepare(directory: Path, seed: int, variants: list[str], overwrite: bool, workers: int) -> None:
    _render_hidden_staffs()
    pages = _select_pages(seed)
    for variant in variants:
        (directory / variant).mkdir(parents=True, exist_ok=True)
    write = functools.partial(_write_variants, directory, variants, overwrite)
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(write, pages), 1):
            if i % 25 == 0:
                print(f"{i} / {len(pages)}", flush=True)

    with open(directory / "pages.json", "w") as f:
        json.dump(pages, f, indent=1)
    groups = Counter(_layout_group(page["systems"]) for page in pages)
    print(f"Wrote {len(pages)} pages x {len(variants)} variants to {directory}: {dict(groups)}")


def _detect_staffs_per_system(image_path: str) -> dict[str, Any]:
    from homr.main import ProcessingConfig, detect_staffs_in_image  # noqa: PLC0415
    from homr.system_repair import _ensure_same_number_of_staffs  # noqa: PLC0415

    config = ProcessingConfig(
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

    def staffs_per_system(multi_staffs: list[Any]) -> list[int]:
        return [sum(2 if s.is_grandstaff else 1 for s in m.staffs) for m in multi_staffs]

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            detected = detect_staffs_in_image(image_path, config)[0]
            final = _ensure_same_number_of_staffs(detected)
    except Exception as e:
        return {"error": repr(e)[:200]}
    return {
        "detected_staffs": sum(staffs_per_system(detected)),
        "systems": staffs_per_system(final),
    }


def classify(result: dict[str, Any], expected: list[int]) -> str:
    if "error" in result:
        return "error"
    systems = result["systems"]
    if len(systems) == 0:
        return "no staffs"
    if sum(systems) < result["detected_staffs"]:
        # Staffs which were detected but dropped, as they didn't fit the page layout
        return "staffs dropped"
    if systems == expected:
        return CORRECT
    if sum(systems) < sum(expected):
        return "missed staffs"
    if sum(systems) > sum(expected):
        return "extra staffs"
    if len(systems) > len(expected):
        return "systems split"
    if len(systems) < len(expected):
        return "systems merged"
    return "some systems wrong"


def _init_worker(workers: int) -> None:
    """Limits each worker to its share of the cores, for OpenCV and onnxruntime."""
    import onnxruntime  # noqa: PLC0415

    threads = max(1, (os.cpu_count() or 1) // workers)
    cv2.setNumThreads(threads)
    original = onnxruntime.InferenceSession

    def session(*args: Any, **kwargs: Any) -> Any:
        options = kwargs.get("sess_options") or onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        kwargs["sess_options"] = options
        return original(*args, **kwargs)

    onnxruntime.InferenceSession = session  # type: ignore


# A process on the GPU mostly waits for the CPU, while each process holds the model
# in GPU memory. 3 keep a 16GB GPU busy.
gpu_workers = 3


def _is_cached(image_path: str) -> bool:
    """Checks that the segmentation cache exists and matches the preprocessed image."""
    from homr.autocrop import autocrop_with_offset  # noqa: PLC0415
    from homr.color_adjust import apply_clahe  # noqa: PLC0415
    from homr.resize import resize_image  # noqa: PLC0415

    cache = Path(image_path).with_suffix(".npy")
    if not cache.exists():
        return False
    image = cv2.imread(image_path)
    if image is None:
        return False
    preprocessed = apply_clahe(resize_image(autocrop_with_offset(image)[0]))
    try:
        with lzma.open(cache, "rb") as f:
            for _ in range(5):
                np.load(f)
            cached_hash = f.readline().decode().strip()
    except (EOFError, lzma.LZMAError, ValueError):
        return False
    return cached_hash == hashlib.sha256(preprocessed.tobytes()).hexdigest()


def _segment_on_gpu(image_path: str) -> None:
    from homr.main import load_and_preprocess_predictions  # noqa: PLC0415

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        load_and_preprocess_predictions(image_path, False, True, True)


def _run_one(task: tuple[str, dict[str, Any], str]) -> tuple[str, str, dict[str, Any]]:
    variant, page, image_path = task
    result = _detect_staffs_per_system(image_path)
    result["expected"] = page["systems"]
    result["category"] = classify(result, page["systems"])
    page_id = page["id"]
    return variant, page_id, result


def run(directory: Path, workers: int, output: Path, variants: list[str], gpu: bool) -> None:
    with open(directory / "pages.json") as f:
        pages = json.load(f)
    missing = [v for v in variants if not (directory / v).exists()]
    if missing:
        raise ValueError(f"Run prepare first, these variants are missing: {', '.join(missing)}")
    tasks = [
        (variant, page, str(directory / variant / f"{page['id']}.png"))
        for variant in variants
        for page in pages
    ]
    context = multiprocessing.get_context("fork")
    if gpu:
        paths = [path for *_, path in tasks]
        with context.Pool(workers, initializer=_init_worker, initargs=(workers,)) as pool:
            cached = pool.map(_is_cached, paths)
        uncached = [path for path, is_cached in zip(paths, cached, strict=True) if not is_cached]
        print(f"Segmenting {len(uncached)} images on the GPU", flush=True)
        initargs = (gpu_workers,)
        with context.Pool(gpu_workers, initializer=_init_worker, initargs=initargs) as pool:
            for i, _ in enumerate(pool.imap_unordered(_segment_on_gpu, uncached), 1):
                if i % 50 == 0:
                    print(f"{i} / {len(uncached)}", flush=True)
    with (
        open(output, "w") as f,
        context.Pool(workers, initializer=_init_worker, initargs=(workers,)) as pool,
    ):
        for i, (variant, page_id, result) in enumerate(pool.imap_unordered(_run_one, tasks), 1):
            f.write(json.dumps({"variant": variant, "page": page_id, **result}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i} / {len(tasks)}", flush=True)
    print(f"Results were written to {output}\n")
    report(load_results(output))


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    """Reads the results of a finished or still running run, by variant and page."""
    results: dict[str, dict[str, Any]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                results[result.pop("variant")][result.pop("page")] = result
    order = [v for v in VARIANTS if v in results]
    return {variant: dict(sorted(results[variant].items())) for variant in order}


def report(results: dict[str, dict[str, Any]]) -> None:
    """
    "of clean correct" only counts pages which are correct on the clean variant,
    to show the effect of the degradation alone.
    """
    if not results:
        print("No results")
        return
    clean = results.get("clean", {})
    clean_correct = {p for p, r in clean.items() if r["category"] == CORRECT}
    groups = list(pages_per_layout)
    width = max(len(variant) for variant in results) + 2
    header = f"{'variant':<{width}}{'correct':>10}{'of clean correct':>18}"
    print(header + "".join(f"{'staffs ' + g if g != 'change' else g:>11}" for g in groups))
    for variant, pages in results.items():
        correct = {p for p, r in pages.items() if r["category"] == CORRECT}
        line = f"{variant:<{width}}{len(correct):>5}/{len(pages):<4}"
        line += f"{len(correct & clean_correct):>12}/{len(clean_correct):<5}"
        for group in groups:
            in_group = {p for p, r in pages.items() if _layout_group(r["expected"]) == group}
            line += f"{len(correct & in_group):>7}/{len(in_group):<3}"
        print(line)
    print("\nFailures by category:")
    for variant, pages in results.items():
        categories = Counter(r["category"] for r in pages.values() if r["category"] != CORRECT)
        print(f"  {variant:<{width}}" + ", ".join(f"{c}: {n}" for c, n in categories.most_common()))


def _outcome(result: dict[str, Any]) -> tuple[Any, str]:
    return result.get("systems"), result["category"]


def compare(before_path: Path, after_path: Path) -> None:
    before = load_results(before_path)
    after = load_results(after_path)
    changes: Counter[str] = Counter()
    for variant in after:
        for page_id, new in sorted(after[variant].items()):
            old = before.get(variant, {}).get(page_id)
            if old is None or _outcome(old) == _outcome(new):
                continue
            was_correct, is_correct = old["category"] == CORRECT, new["category"] == CORRECT
            if is_correct and not was_correct:
                change = "fixed"
            elif was_correct and not is_correct:
                change = "broken"
            else:
                change = "changed"
            changes[change] += 1
            print(f"{change:<8}{variant}/{page_id} expected {new['expected']}")
            print(f"  before {old['category']}: {old.get('systems', old.get('error'))}")
            print(f"  after  {new['category']}: {new.get('systems', new.get('error'))}")
    print(f"\n{changes['fixed']} fixed, {changes['broken']} broken, {changes['changed']} changed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Staff structure benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    variants_help = "Variants or groups of variants, default: all. See validation/degradations.py"
    prepare_parser = subparsers.add_parser("prepare", help="Select pages and write variants.")
    prepare_parser.add_argument("--directory", type=Path, default=default_directory)
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--workers", type=int, default=os.cpu_count())
    prepare_parser.add_argument("--variants", nargs="*", help=variants_help)
    prepare_parser.add_argument(
        "--overwrite", action="store_true", help="Write variants again which already exist."
    )
    prepare_parser.add_argument(
        "--backgrounds",
        type=Path,
        help="Directory with photos of desks, rooms, ... to use as surroundings of the sheet.",
    )
    run_parser = subparsers.add_parser("run", help="Run the staff detection on all variants.")
    run_parser.add_argument("--directory", type=Path, default=default_directory)
    run_parser.add_argument("--workers", type=int, default=os.cpu_count())
    run_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run the segmentation on the GPU (fp16 model, results differ slightly from the CPU).",
    )
    run_parser.add_argument("--output", type=Path, default=Path("staff_structure.jsonl"))
    run_parser.add_argument("--variants", nargs="*", help=variants_help)
    report_parser = subparsers.add_parser("report", help="Report the results of a run.")
    report_parser.add_argument("results", type=Path)
    compare_parser = subparsers.add_parser("compare", help="Compare the results of two runs.")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after", type=Path)
    args = parser.parse_args()

    if args.command == "prepare":
        set_background_photos(args.backgrounds)
        variants = resolve_variants(args.variants)
        prepare(args.directory, args.seed, variants, args.overwrite, args.workers)
    elif args.command == "run":
        run(args.directory, args.workers, args.output, resolve_variants(args.variants), args.gpu)
    elif args.command == "report":
        report(load_results(args.results))
    else:
        compare(args.before, args.after)


if __name__ == "__main__":
    main()
