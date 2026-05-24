import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
ARTIFACT_ROOT = WORKSPACE_DIR / "attacks" / "artifacts" / "spectral_noise_injection"
POISONED_ROOT = ARTIFACT_ROOT / "poisoned_images"
RENDERED_ROOT = ARTIFACT_ROOT / "rendered_musicxml"
CHECKPOINT_PATH = RENDERED_ROOT / "render_checkpoint.txt"

DEFAULT_RENDER_LIMIT = 10000
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))


def resolve_musescore_path() -> Path | None:
    env_path = os.environ.get("MUSESCORE_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate

    candidates = [
        Path(r"D:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files\MuseScore 3\bin\mscore3.exe"),
        Path(r"C:\Program Files (x86)\MuseScore 3\bin\mscore3.exe"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for executable_name in ("MuseScore4.exe", "mscore3.exe", "MuseScore4", "mscore3"):
        found = shutil.which(executable_name)
        if found:
            return Path(found)

    return None


def render_one(muse_path: Path, musicxml_file: Path, output_file: Path) -> tuple[bool, str]:
    cmd = [
        str(muse_path),
        "-T",
        "10",
        "-r",
        "300",
        "-o",
        str(output_file),
        str(musicxml_file),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True, ""
    except subprocess.CalledProcessError as error:
        return False, error.stderr.decode(errors="replace")
    except FileNotFoundError:
        return False, f"MuseScore executable not found at {muse_path}"


def load_completed_keys() -> set[str]:
    completed_keys: set[str] = set()
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open("r", encoding="utf-8") as checkpoint_file:
            for line in checkpoint_file:
                key = line.strip()
                if key:
                    completed_keys.add(key)
    return completed_keys


def record_completed_key(key: str) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as checkpoint_file:
        checkpoint_file.write(key + "\n")


def format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def build_speed_report(completed: int, total_jobs: int, elapsed_seconds: float) -> str:
    if completed <= 0 or elapsed_seconds <= 0:
        return ""

    average_per_second = completed / elapsed_seconds
    average_per_minute = average_per_second * 60
    report = f" | elapsed {format_elapsed(elapsed_seconds)} | avg {average_per_minute:.2f} scores/min"

    remaining = total_jobs - completed
    if remaining > 0 and average_per_second > 0:
        eta_seconds = remaining / average_per_second
        report += f" | eta {format_elapsed(eta_seconds)}"

    return report


def iter_poisoned_musicxml_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.musicxml") if path.is_file())


def render_poisoned_musicxml(limit: int = DEFAULT_RENDER_LIMIT, workers: int = DEFAULT_WORKERS) -> None:
    if not POISONED_ROOT.exists():
        print(f"Error: could not find poisoned image root at {POISONED_ROOT}")
        return

    musicxml_files = iter_poisoned_musicxml_files(POISONED_ROOT)
    if not musicxml_files:
        print(f"No .musicxml files found under {POISONED_ROOT}")
        return

    if limit > 0:
        musicxml_files = musicxml_files[:limit]
        print(f"Limiting rendering to the first {limit} MusicXML files.")
    else:
        print("Rendering all MusicXML files in the poisoned image tree.")

    muse_path = resolve_musescore_path()
    if muse_path is None:
        print(
            "Error: MuseScore executable not found. Set MUSESCORE_PATH to MuseScore4.exe or mscore3.exe, "
            "or install MuseScore in a standard location."
        )
        sys.exit(1)

    completed_keys = load_completed_keys()
    if completed_keys:
        print(f"Loaded {len(completed_keys)} completed render(s) from checkpoint.")

    jobs: list[tuple[int, str, Path, Path]] = []
    total = len(musicxml_files)
    for index, musicxml_file in enumerate(musicxml_files, start=1):
        relative_musicxml = musicxml_file.relative_to(POISONED_ROOT)
        key = str(relative_musicxml.with_suffix(""))
        output_file = RENDERED_ROOT / relative_musicxml.parent / f"{relative_musicxml.stem}_rendered.png"

        print(f"[{index}/{total}] Processing: {relative_musicxml}")

        if key in completed_keys:
            print(f"[{index}/{total}] Skipping (already completed): {output_file.name}")
            continue

        if output_file.exists():
            print(f"[{index}/{total}] Skipping (output already exists): {output_file.name}")
            completed_keys.add(key)
            continue

        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{total}] Queued rendering: {musicxml_file.name} -> {output_file.name}")
        jobs.append((index, key, musicxml_file, output_file))

    if not jobs:
        print("No renderable MusicXML files were found in the selected rows.")
        return

    if workers <= 1:
        start_time = os.times()
        for index, key, musicxml_file, output_file in jobs:
            print(f"[{index}/{total}] Rendering: {musicxml_file.name} -> {output_file.name}")
            success, message = render_one(muse_path, musicxml_file, output_file)
            elapsed_seconds = sum(os.times()[:2]) - sum(start_time[:2])
            if success:
                record_completed_key(key)
                completed_keys.add(key)
                print(
                    f"[{index}/{total}] Successfully rendered: {key}"
                    f"{build_speed_report(index, len(jobs), elapsed_seconds)}"
                )
            else:
                print(f"[{index}/{total}] Error rendering {musicxml_file.name}: {message}")
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(render_one, muse_path, musicxml_file, output_file): (index, key, musicxml_file, output_file)
            for index, key, musicxml_file, output_file in jobs
        }

        completed = 0
        for future in as_completed(future_to_job):
            index, key, musicxml_file, output_file = future_to_job[future]
            completed += 1
            success, message = future.result()
            elapsed_seconds = time.perf_counter() - start_time
            if success:
                record_completed_key(key)
                completed_keys.add(key)
                print(
                    f"[{index}/{total}] Done ({completed}/{len(jobs)}): {key}"
                    f"{build_speed_report(completed, len(jobs), elapsed_seconds)}"
                )
            else:
                print(f"[{index}/{total}] Error rendering {musicxml_file.name}: {message}")

    total_elapsed_seconds = time.perf_counter() - start_time
    print(
        f"Finished {len(jobs)} render(s) in {format_elapsed(total_elapsed_seconds)}"
        f" ({len(jobs) / max(total_elapsed_seconds, 0.000001) * 60:.2f} scores/min)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render generated MusicXML files from attacks/artifacts/spectral_noise_injection/poisoned_images."
    )
    parser.add_argument(
        "-N",
        "--limit",
        type=int,
        default=DEFAULT_RENDER_LIMIT,
        help=(
            f"Maximum number of MusicXML files to render (default: {DEFAULT_RENDER_LIMIT}). "
            "Use 0 or a negative value to render all files."
        ),
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"Number of parallel MuseScore processes to run (default: {DEFAULT_WORKERS}). "
            "Use 1 to disable parallel rendering."
        ),
    )
    args = parser.parse_args()
    render_poisoned_musicxml(limit=args.limit, workers=args.workers)