# flake8: noqa: T201

"""
Compares the layout of each system on the real photos in testdata, single and grand
staffs, with the parts in each reference.musicxml. Evaluation only, nothing is tuned on
these photos.
Set HOMR_STAFF_RELATIONS=1 to use the staff relation model.
Usage: poetry run python -m training.staff_relations.evaluate_photos --output result.jsonl
"""

import argparse
import contextlib
import io
import json
import multiprocessing
import re
from pathlib import Path
from typing import Any

from validation.staff_structure import _init_worker

testdata = Path("testdata")


def expected_layout(musicxml: str) -> list[bool]:
    """
    The staffs of a system from top to bottom, True for a grand staff: a part with two
    staffs is a grand staff, homr writes grand staffs this way.
    """
    layout = []
    for part in re.findall(r"<part id=.*?</part>", musicxml, re.DOTALL):
        staves = int(m.group(1)) if (m := re.search(r"<staves>(\d+)</staves>", part)) else 1
        layout.extend([True] if staves == 2 else [False] * staves)  # noqa: PLR2004
    return layout


def photos() -> list[tuple[Path, list[bool]]]:
    """Each photo with the system layout of its reference, one per folder or photo."""
    result: list[tuple[Path, list[bool]]] = []
    for image in sorted(testdata.glob("*/*.jpg")):
        reference = image.parent / "reference.musicxml"
        if not reference.exists():
            reference = image.with_suffix(".musicxml")
        if reference.exists():
            musicxml = reference.read_text(encoding="utf-8")
            layout = expected_layout(musicxml)
            result.append((image, layout))
    return result


def _run(task: tuple[Path, list[bool]]) -> dict[str, Any]:
    from homr.main import ProcessingConfig, detect_staffs_in_image  # noqa: PLC0415
    from homr.system_repair import repair_systems  # noqa: PLC0415

    image, expected = task
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
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            multi_staffs, _, debug, _, _, _ = detect_staffs_in_image(str(image), config)
            detected = [[s.is_grandstaff for s in m.staffs] for m in multi_staffs]
            repaired = repair_systems(debug, multi_staffs)
    except Exception as e:
        return {"image": str(image), "expected": expected, "error": repr(e)[:200]}
    systems = [[s.is_grandstaff for s in m.staffs] for m in repaired]
    return {
        "image": str(image),
        "expected": expected,
        "detected": detected,
        "systems": systems,
        "correct_before_repair": len(detected) > 0 and all(lay == expected for lay in detected),
        "correct": len(systems) > 0 and all(layout == expected for layout in systems),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    tasks = photos()
    context = multiprocessing.get_context("fork")
    with (
        context.Pool(args.workers, initializer=_init_worker, initargs=(args.workers,)) as pool,
        open(args.output, "w") as f,
    ):
        results = list(pool.imap(_run, tasks))
        for result in results:
            f.write(json.dumps(result) + "\n")
    correct = sum(r.get("correct", False) for r in results)
    before = sum(r.get("correct_before_repair", False) for r in results)
    print(f"{correct} / {len(results)} photos with the expected layout in every system")
    print(f"{before} / {len(results)} before system_repair")
    for r in results:
        if not r.get("correct"):
            print(
                f"  {r['image']}: expected {r['expected']}, got {r.get('systems', r.get('error'))}"
            )


if __name__ == "__main__":
    main()
