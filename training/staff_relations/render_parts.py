# flake8: noqa: T201

"""
Renders single parts of Lieder pieces with MuseScore: the first part (voice, single
staffs) and the last part (piano, grand staffs). Lieder has no pages with only single
staffs, the staff relation model needs them.
Usage: poetry run python -m training.staff_relations.render_parts
"""

import argparse
import json
import os
import random
import re
import subprocess
from pathlib import Path

from training.omr_datasets.convert_lieder import (
    _ensure_music_font_style_files,
    _music_font_for_file,
    _music_font_style_file,
)
from validation.staff_structure import lieder_directory, musescore

default_directory = Path("datasets/staff_relations_parts")

# Index of the part, the melody is the voice without lyrics
parts = {"voice": 0, "piano": -1, "melody": 0}


def remove_lyrics(musicxml: str) -> str:
    """Without lyrics the staffs of a melody are as close as those of instrumental music."""
    return re.sub(r"<lyric(\s[^>]*)?>.*?</lyric>", "", musicxml, flags=re.DOTALL)


def keep_part(musicxml: str, index: int) -> str | None:
    """The score with only one of its parts, None if it has only one part."""
    ids = re.findall(r'<score-part id="([^"]+)"', musicxml)
    if len(ids) < 2:  # noqa: PLR2004
        return None
    keep = ids[index]
    for part_id in ids:
        if part_id == keep:
            continue
        musicxml = re.sub(
            rf'<score-part id="{re.escape(part_id)}">.*?</score-part>',
            "",
            musicxml,
            flags=re.DOTALL,
        )
        musicxml = re.sub(
            rf'<part id="{re.escape(part_id)}">.*?</part>', "", musicxml, flags=re.DOTALL
        )
    musicxml = re.sub(r"<part-group[^>]*/>", "", musicxml)
    musicxml = re.sub(r"<part-group .*?</part-group>", "", musicxml, flags=re.DOTALL)
    # The breaks of the full score would leave the pages of a single part half empty
    return re.sub(r' new-(system|page)="yes"', "", musicxml)


def style_with_start_bar_line(style: str, directory: Path) -> str:
    """
    A copy of a style which draws the bar line at the start of systems with one staff.
    MuseScore leaves it out by default, many other engravings have it.
    """
    target = directory / f"start_bar_line_{Path(style).name}"
    if not target.exists():
        content = Path(style).read_text(encoding="utf-8")
        content = content.replace(
            "</Style>", "  <startBarlineSingle>1</startBarlineSingle>\n  </Style>"
        )
        target.write_text(content, encoding="utf-8")
    return str(target)


def main() -> None:
    from training.staff_relations.create_dataset import (  # noqa: PLC0415
        benchmark_pieces,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=default_directory)
    parser.add_argument("--pieces", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--parts", nargs="+", choices=list(parts), default=list(parts))
    parser.add_argument(
        "--start-bar-line", action="store_true", help="Bar line at the start of single staffs"
    )
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    excluded = benchmark_pieces()
    scores = sorted(
        path for path in lieder_directory.glob("*.musicxml") if path.stem not in excluded
    )
    random.Random(args.seed).shuffle(scores)
    _ensure_music_font_style_files()
    jobs = []
    for score in scores[: args.pieces]:
        musicxml = score.read_text(encoding="utf-8")
        for name, index in parts.items():
            if name not in args.parts:
                continue
            part = keep_part(musicxml, index)
            if part is None:
                continue
            if name == "melody":
                part = remove_lyrics(part)
            source = args.directory / f"{score.stem}_{name}.musicxml"
            target = source.with_suffix(".svg")
            if any(args.directory.glob(f"{source.stem}-*.svg")):
                continue
            source.write_text(part, encoding="utf-8")
            style = _music_font_style_file(_music_font_for_file(str(score)))
            if args.start_bar_line:
                style = style_with_start_bar_line(style, args.directory)
            jobs.append({"in": str(source), "out": str(target), "style": style})
    print(f"Rendering {len(jobs)} parts with MuseScore", flush=True)
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "QT_QUICK_BACKEND": "software",
        "MU_QT_QPA_PLATFORM": "offscreen",
    }
    for start in range(0, len(jobs), 50):
        job_file = args.directory / "jobs.json"
        job_file.write_text(json.dumps(jobs[start : start + 50]))
        try:
            result = subprocess.run(  # noqa: S603
                [str(musescore), "--force", "-j", str(job_file)],
                env=env,
                capture_output=True,
                timeout=600,
                check=False,
            )
            if result.returncode != 0:
                print(f"MuseScore failed on batch {start}: {result.stderr[-300:]!r}")
        except subprocess.TimeoutExpired:
            # MuseScore sometimes hangs, the batch is lost
            print(f"MuseScore timed out on batch {start}")
        print(f"{min(start + 50, len(jobs))} / {len(jobs)}", flush=True)
    job_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
