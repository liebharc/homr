import csv
import multiprocessing
import os
import random
import zipfile
from itertools import zip_longest
from pathlib import Path

import cv2

from homr.circle_of_fifths import strip_naturals
from homr.download_utils import download_file, untar_file
from homr.simple_logging import eprint
from homr.transformer.configs import default_config
from training.omr_datasets.convert_lieder import (
    MeasureCutter,
    _count_staffs,
    contains_only_supported_clefs,
    is_grandstaff,
)
from training.omr_datasets.convert_musetrainer import (
    _N_WORKERS,
    _TIMEOUT_SECONDS,
    _WINDOW_SIZE,
    _context_at_measure,
    _svg_to_png,
    _tokens_to_svg,
)
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.transformer.training_vocabulary import (
    calc_ratio_of_tuplets,
    check_token_lines,
    token_lines_to_str,
)

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
pdmx_root = os.path.join(dataset_root, "pdmx")
pdmx_csv = os.path.join(pdmx_root, "PDMX.csv")
pdmx_mxl_root = os.path.join(pdmx_root, "mxl")
pdmx_out_root = os.path.join(pdmx_root, "out")
pdmx_train_index = os.path.join(pdmx_root, "index.txt")

_MAX_COMPLEXITY = 2
_MAX_TRACKS = 2
_TARGET_FILES = 10000  # ~50K images


def _read_mxl(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        xml_name = next(
            (n for n in names if n.endswith(".xml") and "/" not in n),
            next((n for n in names if n.endswith(".xml")), None),
        )
        if xml_name is None:
            raise ValueError(f"No XML found in {path}")
        return zf.read(xml_name).decode("utf-8")


def _convert_file_impl(mxl_path: Path) -> list[str]:
    try:
        xml_str = _read_mxl(mxl_path)
    except Exception as e:
        eprint("Failed to read", mxl_path, e)
        return []

    try:
        voices = music_xml_string_to_tokens(xml_str)
    except Exception as e:
        eprint("Failed to parse", mxl_path, e)
        return []

    if not voices:
        return []

    voices_to_process = [v for v in voices if _count_staffs(v) >= 1]
    if not voices_to_process:
        return []

    stem = mxl_path.stem

    rel_parts = mxl_path.relative_to(pdmx_mxl_root).parts
    out_dir = os.path.join(pdmx_out_root, *rel_parts[:-1])
    os.makedirs(out_dir, exist_ok=True)

    results: list[str] = []

    for voice_idx, voice in enumerate(voices_to_process):
        n_measures = len(voice)
        if n_measures < 2:
            continue

        n_staffs = 2 if is_grandstaff(voice) else 1

        window_start = 0
        window_idx = 0
        while window_start < n_measures:
            end = min(window_start + _WINDOW_SIZE, n_measures)
            window_measures = voice[window_start:end]

            clefs, key, time_sym = _context_at_measure(voice, window_start, n_staffs)
            cutter = MeasureCutter(list(window_measures))
            cutter.clefs = clefs
            cutter.key = key
            cutter.time = time_sym

            tokens = cutter.extract_measures(len(window_measures), always_include_time=True)

            if calc_ratio_of_tuplets(tokens) <= 0.2 and contains_only_supported_clefs(tokens):
                tokens = strip_naturals(tokens)
                try:
                    if len(tokens) > default_config.max_seq_len - 2:
                        raise ValueError("Sequence too long")
                    check_token_lines(tokens)
                except ValueError:
                    pass
                else:
                    svg_str = _tokens_to_svg(tokens)
                    if svg_str is not None:
                        img = _svg_to_png(svg_str)
                        if img is not None:
                            basename = f"{stem}-v{voice_idx}-w{window_idx}"
                            img_path = os.path.join(out_dir, basename + ".jpg")
                            tok_path = os.path.join(out_dir, basename + ".tokens")
                            cv2.imwrite(img_path, img)
                            with open(tok_path, "w") as f:
                                f.write(token_lines_to_str(tokens))
                            rel_img = str(Path(img_path).relative_to(git_root))
                            rel_tok = str(Path(tok_path).relative_to(git_root))
                            results.append(rel_img + "," + rel_tok + "\n")

            window_start = end
            window_idx += 1

    return results


def _load_filtered_paths() -> list[Path]:
    buckets: dict[tuple[int, int], list[Path]] = {}
    with open(pdmx_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["subset:no_license_conflict"] != "True":
                continue
            if row["subset:all_valid"] != "True":
                continue
            try:
                n_tracks = int(row["n_tracks"])
                if n_tracks > _MAX_TRACKS:
                    continue
            except (ValueError, KeyError):
                continue
            try:
                complexity = int(row["complexity"])
                if complexity > _MAX_COMPLEXITY:
                    continue
            except (ValueError, KeyError):
                continue
            rel_mxl = row["mxl"].lstrip("./")
            buckets.setdefault((n_tracks, complexity), []).append(Path(pdmx_root) / rel_mxl)

    rng = random.Random(42)
    for paths in buckets.values():
        rng.shuffle(paths)

    for key in sorted(buckets):
        n_tracks, complexity = key
        eprint(f"  tracks={n_tracks} complexity={complexity}: {len(buckets[key]):,} files")

    sentinel = object()
    result: list[Path] = []
    for group in zip_longest(*[buckets[k] for k in sorted(buckets)], fillvalue=sentinel):
        for item in group:
            if item is not sentinel:
                result.append(item)  # type: ignore[arg-type]
            if len(result) >= _TARGET_FILES:
                return result
    return result


def convert_pdmx() -> None:
    os.makedirs(pdmx_root, exist_ok=True)

    if not os.path.exists(pdmx_csv):
        eprint("Downloading PDMX.csv (~214 MB)")
        download_file(
            "https://zenodo.org/api/records/15571083/files/PDMX.csv/content",
            pdmx_csv,
        )

    if not os.path.exists(pdmx_mxl_root):
        eprint("Downloading PDMX mxl.tar.gz (~1.8 GB)")
        mxl_archive = os.path.join(pdmx_root, "mxl.tar.gz")
        download_file(
            "https://zenodo.org/api/records/15571083/files/mxl.tar.gz/content",
            mxl_archive,
        )
        eprint("Extracting mxl.tar.gz")
        untar_file(mxl_archive, pdmx_root)

    os.makedirs(pdmx_out_root, exist_ok=True)

    eprint("Reading CSV and applying filters")
    mxl_paths = _load_filtered_paths()
    eprint(f"{len(mxl_paths)} files pass pre-filters (c<={_MAX_COMPLEXITY}, tracks<={_MAX_TRACKS})")

    with open(pdmx_train_index, "w") as index_f:
        file_number = 0
        skipped_files = 0
        with multiprocessing.Pool(processes=_N_WORKERS, maxtasksperchild=2) as p:
            async_results = [
                (path, p.apply_async(_convert_file_impl, (path,))) for path in mxl_paths
            ]
            for path, ar in async_results:
                try:
                    result = ar.get(timeout=_TIMEOUT_SECONDS)
                except multiprocessing.TimeoutError:
                    eprint("Timeout processing", path, "skipping")
                    result = []
                if result:
                    for line in result:
                        index_f.write(line)
                    index_f.flush()
                else:
                    skipped_files += 1
                file_number += 1
                if file_number % 10 == 0:
                    eprint(
                        f"Processed {file_number}/{len(mxl_paths)} files,",
                        f"skipped {skipped_files} files",
                    )

    eprint("Done - index written to", pdmx_train_index)


if __name__ == "__main__":
    convert_pdmx()
