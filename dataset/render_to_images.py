import pandas as pd
import os
import subprocess
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
import sys

# 1. Setup paths relative to this script's location
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "PDMX.csv"
MXL_ROOT = BASE_DIR / "mxl"
OUTPUT_DIR = BASE_DIR / "images"
CHECKPOINT_PATH = OUTPUT_DIR / "render_checkpoint.txt"


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


def render_one(muse_path: Path, mxl_file: Path, output_file: Path) -> tuple[bool, str]:
    cmd = [
        str(muse_path),
        "-T", "10",
        "-r", "300",
        "-o", str(output_file),
        str(mxl_file),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode(errors="replace")
    except FileNotFoundError:
        return False, f"MuseScore executable not found at {muse_path}"


def load_completed_stems() -> set[str]:
    completed_stems: set[str] = set()
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open("r", encoding="utf-8") as checkpoint_file:
            for line in checkpoint_file:
                stem = line.strip()
                if stem:
                    completed_stems.add(stem)
    return completed_stems


def record_completed_stem(stem: str) -> None:
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as checkpoint_file:
        checkpoint_file.write(stem + "\n")


def resolve_score_file(row: pd.Series, actual_score_files: dict[str, Path]) -> Path | None:
    csv_mxl_value = row.get("mxl")
    if isinstance(csv_mxl_value, str) and csv_mxl_value.strip():
        candidate = (BASE_DIR / csv_mxl_value.lstrip("./")).resolve()
        if candidate.exists():
            return candidate

    csv_stem = Path(row["path"]).stem
    return actual_score_files.get(csv_stem)


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
    report = f" | elapsed {format_elapsed(elapsed_seconds)} | avg {average_per_minute:.2f} images/min"

    remaining = total_jobs - completed
    if remaining > 0 and average_per_second > 0:
        eta_seconds = remaining / average_per_second
        report += f" | eta {format_elapsed(eta_seconds)}"

    return report

def render_dataset(limit: int = DEFAULT_RENDER_LIMIT, workers: int = DEFAULT_WORKERS):
    # 2. Prepare output directory
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 3. Load dataset
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_PATH}")
        return

    # 4. Recursively map the actual score files on disk
    print("Scanning 'mxl' folder recursively for score files...")
    # This maps 'filename' -> Path object
    actual_score_files = {}
    for pattern in ("*.mxl", "*.xml"):
        for p in MXL_ROOT.rglob(pattern):
            # Store using just the stem (e.g., 'QmWiUDGmGLyK8...')
            actual_score_files[p.stem] = p

    print(f"Found {len(actual_score_files)} total score files in the directory tree.")

    muse_path = resolve_musescore_path()
    if muse_path is None:
        print(
            "Error: MuseScore executable not found. Set MUSESCORE_PATH to the full path "
            "of MuseScore4.exe (or mscore3.exe), or install MuseScore in a standard location."
        )
        sys.exit(1)

    completed_stems = load_completed_stems()
    if completed_stems:
        print(f"Loaded {len(completed_stems)} completed render(s) from checkpoint.")

    # 5. Process files by matching CSV references to our disk map
    if limit > 0:
        df = df.head(limit)
        print(f"Limiting rendering to the first {limit} rows.")
    else:
        print("Rendering all rows in the dataset.")

    total = len(df)
    print(f"Using {workers} worker(s) for rendering.")

    jobs = []
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        # Get the core filename stem from the CSV path (ignores .json, .xml, or folder strings)
        csv_stem = Path(row['path']).stem

        print(f"[{index}/{total}] Processing: {csv_stem}")

        output_file = OUTPUT_DIR / f"{csv_stem}.png"

        if csv_stem in completed_stems:
            print(f"[{index}/{total}] Skipping (already completed): {output_file.name}")
            continue

        if output_file.exists():
            print(f"[{index}/{total}] Skipping (output already exists): {output_file.name}")
            completed_stems.add(csv_stem)
            continue
        
        # Look up the exact score path from the CSV first, then fall back to the recursive scan.
        score_file = resolve_score_file(row, actual_score_files)
        if score_file is None:
            print(f"[{index}/{total}] Skipping (not found anywhere in mxl/): {csv_stem}")
            continue
            
        mxl_file = score_file
        print(f"[{index}/{total}] Queued rendering: {mxl_file.name} -> {output_file.name}")
        jobs.append((index, csv_stem, mxl_file, output_file))

    if not jobs:
        print("No renderable files were found in the selected rows.")
        return

    start_time = time.perf_counter()

    if workers <= 1:
        for index, csv_stem, mxl_file, output_file in jobs:
            print(f"[{index}/{total}] Rendering: {mxl_file.name} -> {output_file.name}")
            success, message = render_one(muse_path, mxl_file, output_file)
            elapsed_seconds = time.perf_counter() - start_time
            if success:
                record_completed_stem(csv_stem)
                completed_stems.add(csv_stem)
                print(
                    f"[{index}/{total}] Successfully rendered: {csv_stem}"
                    f"{build_speed_report(index, len(jobs), elapsed_seconds)}"
                )
            else:
                print(f"[{index}/{total}] Error rendering {mxl_file.name}: {message}")
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(render_one, muse_path, mxl_file, output_file): (index, csv_stem, mxl_file, output_file)
            for index, csv_stem, mxl_file, output_file in jobs
        }

        completed = 0
        for future in as_completed(future_to_job):
            index, csv_stem, mxl_file, output_file = future_to_job[future]
            completed += 1
            success, message = future.result()
            elapsed_seconds = time.perf_counter() - start_time
            if success:
                record_completed_stem(csv_stem)
                completed_stems.add(csv_stem)
                print(
                    f"[{index}/{total}] Done ({completed}/{len(jobs)}): {csv_stem}"
                    f"{build_speed_report(completed, len(jobs), elapsed_seconds)}"
                )
            else:
                print(f"[{index}/{total}] Error rendering {mxl_file.name}: {message}")

    total_elapsed_seconds = time.perf_counter() - start_time
    print(
        f"Finished {len(jobs)} render(s) in {format_elapsed(total_elapsed_seconds)}"
        f" ({len(jobs) / max(total_elapsed_seconds, 0.000001) * 60:.2f} images/min)."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render MusicXML/MXL files from the PDMX dataset.")
    parser.add_argument(
        "-N",
        "--limit",
        type=int,
        default=DEFAULT_RENDER_LIMIT,
        help=(
            f"Maximum number of images to generate (default: {DEFAULT_RENDER_LIMIT}). "
            "Use 0 or a negative value to render all rows."
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
    render_dataset(limit=args.limit, workers=args.workers)