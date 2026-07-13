import io
import json
import multiprocessing
import os
import random
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from homr.circle_of_fifths import strip_naturals
from homr.download_utils import download_file, unzip_file
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml, xml_to_string
from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.transformer.configs import default_config
from homr.transformer.vocabulary import EncodedSymbol, empty
from homr.type_definitions import NDArray
from training.omr_datasets.convert_lieder import (
    MeasureCutter,
    _count_staffs,
    contains_only_supported_clefs,
    is_grandstaff,
)
from training.omr_datasets.music_xml_parser import Measure, music_xml_string_to_tokens
from training.transformer.training_vocabulary import (
    calc_ratio_of_tuplets,
    check_token_lines,
    token_lines_to_str,
)

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
musetrainer_root = os.path.join(dataset_root, "musetrainer")
musetrainer_train_index = os.path.join(musetrainer_root, "index.txt")

_WINDOW_SIZE = 8
_VEROVIO_FONTS = ["Leipzig", "Bravura", "Gootville", "Leland", "Petaluma"]
_N_WORKERS = max(1, os.cpu_count() or 4)
_TIMEOUT_SECONDS = 20
_RENDER_TIMEOUT_SECONDS = _TIMEOUT_SECONDS


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


def _context_at_measure(
    voice: list[Measure], measure_idx: int, n_staffs: int
) -> tuple[list[EncodedSymbol], EncodedSymbol, EncodedSymbol]:
    if n_staffs >= 2:
        clefs: list[EncodedSymbol] = [
            EncodedSymbol("clef_G2", empty, empty, empty, empty, "upper"),
            EncodedSymbol("clef_F4", empty, empty, empty, empty, "lower"),
        ]
    else:
        clefs = [EncodedSymbol("clef_G2", empty, empty, empty, empty, "upper")]
    key: EncodedSymbol = EncodedSymbol("keySignature_0")
    time_sym: EncodedSymbol = EncodedSymbol("timeSignature/4")

    for i in range(measure_idx):
        for symbol in voice[i]:
            if "clef" in symbol.rhythm:
                idx = 1 if symbol.position == "lower" and len(clefs) > 1 else 0
                clefs[idx] = symbol
            elif "keySignature" in symbol.rhythm:
                key = symbol
            elif "timeSignature" in symbol.rhythm:
                time_sym = symbol

    return clefs, key, time_sym


_DYNAMICS = ("pp", "p", "mp", "mf", "f", "ff", "fff", "sf", "sfz", "fp")
_TEMPO_MARKS = (
    ("Largo", 50),
    ("Adagio", 70),
    ("Andante", 90),
    ("Moderato", 100),
    ("Allegro", 120),
    ("Vivace", 140),
    ("Presto", 170),
)


def _insert_direction(measure: ET.Element, direction: ET.Element) -> None:
    note_idx = next((i for i, child in enumerate(measure) if child.tag == "note"), len(measure))
    measure.insert(note_idx, direction)


def inject_musicxml_markings(xml_str: str) -> str:
    try:
        root = ET.fromstring(xml_str)  # noqa: S314
    except ET.ParseError:
        return xml_str

    measures = root.findall(".//measure")
    if not measures:
        return xml_str

    if random.random() < 0.40:
        tempo_text, tempo_bpm = random.choice(_TEMPO_MARKS)
        direction = ET.Element("direction", attrib={"placement": "above"})
        dtype = ET.SubElement(direction, "direction-type")
        ET.SubElement(dtype, "words").text = tempo_text
        ET.SubElement(direction, "sound").set("tempo", str(tempo_bpm))
        _insert_direction(measures[0], direction)

    if random.random() < 0.60:
        n = random.randint(1, min(3, len(measures)))
        for measure in random.sample(measures, n):
            dyn_tag = random.choice(_DYNAMICS)
            direction = ET.Element("direction", attrib={"placement": "below"})
            dtype = ET.SubElement(direction, "direction-type")
            ET.SubElement(ET.SubElement(dtype, "dynamics"), dyn_tag)
            _insert_direction(measure, direction)

    return ET.tostring(root, encoding="unicode")


def _render_svg_in_subprocess(
    xml_str: str, scale: int, font: str, mnum_interval: int
) -> str | None:
    # Verovio can hang or crash on pathological input (observed empirically). Running it in a
    # disposable subprocess with a hard timeout means a stuck render just gets killed - it can
    # never leak a worker in the parent Pool the way an in-process hang or a signal-based
    # timeout (which can crash mid-C-call) would.
    request = json.dumps(
        {"xml_str": xml_str, "scale": scale, "font": font, "mnum_interval": mnum_interval}
    )
    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "training.omr_datasets.verovio_render_worker"],
            input=request.encode(),
            capture_output=True,
            timeout=_RENDER_TIMEOUT_SECONDS,
            cwd=git_root,
            check=True,
        )
    except subprocess.TimeoutExpired:
        eprint("Verovio rendering timed out")
        return None
    except subprocess.CalledProcessError as e:
        # A segfault (the common case for pathological input) kills the process before it
        # can write anything, so stderr is often empty - fall back to reporting the signal
        # or exit code instead of staying silent.
        stderr = e.stderr.decode(errors="replace").strip()
        if not stderr:
            stderr = (
                f"killed by signal {signal.Signals(-e.returncode).name}"
                if e.returncode < 0
                else f"exit code {e.returncode}"
            )
        eprint("Verovio rendering failed:", stderr)
        return None

    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        eprint("Verovio rendering returned invalid output:", e)
        return None

    svg: str | None = response.get("svg")
    return svg


def _tokens_to_svg(tokens: list[EncodedSymbol]) -> str | None:
    try:
        xml_el = generate_xml(XmlGeneratorArguments(), [tokens], "")
        xml_str = inject_musicxml_markings(xml_to_string(xml_el))
    except Exception as e:
        eprint("XML generation failed:", e)
        return None

    try:
        root = ET.fromstring(xml_str)  # noqa: S314
        start = random.randint(1, 150)
        for i, measure in enumerate(root.findall(".//measure")):
            measure.set("number", str(start + i))
        xml_str = ET.tostring(root, encoding="unicode")
    except ET.ParseError:
        pass

    scale = random.randint(40, 80)
    font = random.choice(_VEROVIO_FONTS)
    mnum_interval = 1 if random.random() < 0.20 else 0

    return _render_svg_in_subprocess(xml_str, scale, font, mnum_interval)


def _svg_to_png(svg_str: str) -> NDArray | None:
    try:
        result = subprocess.run(  # noqa: S603
            ["rsvg-convert"],  # noqa: S607
            input=svg_str.encode(),
            capture_output=True,
            timeout=_RENDER_TIMEOUT_SECONDS,
            check=True,
        )
        rgba = Image.open(io.BytesIO(result.stdout)).convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        img: NDArray = np.array(bg.convert("L"))
    except subprocess.TimeoutExpired:
        eprint("SVG rasterization timed out")
        return None
    except Exception as e:
        eprint("SVG rasterization failed:", e)
        return None

    return add_image_into_tr_omr_canvas(img)


def _convert_file_impl(path: Path) -> list[str]:
    try:
        xml_str = _read_mxl(path)
    except Exception as e:
        eprint("Failed to read", path, e)
        return []

    try:
        voices = music_xml_string_to_tokens(xml_str)
    except Exception as e:
        eprint("Failed to parse", path, e)
        return []

    if not voices:
        return []

    voices_to_process = [v for v in voices if _count_staffs(v) >= 1]
    if not voices_to_process:
        return []

    stem = path.stem
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
                            img_path = os.path.join(musetrainer_root, basename + ".jpg")
                            tok_path = os.path.join(musetrainer_root, basename + ".tokens")
                            cv2.imwrite(img_path, img)
                            with open(tok_path, "w") as f:
                                f.write(token_lines_to_str(tokens))
                            rel_img = str(Path(img_path).relative_to(git_root))
                            rel_tok = str(Path(tok_path).relative_to(git_root))
                            results.append(rel_img + "," + rel_tok + "\n")

            window_start = end
            window_idx += 1

    return results


def convert_musetrainer() -> None:
    if not os.path.exists(musetrainer_root):
        eprint("Downloading MuseTrainer from https://github.com/musetrainer/library")
        archive = os.path.join(dataset_root, "musetrainer.zip")
        download_file(
            "https://github.com/musetrainer/library/archive/refs/heads/master.zip",
            archive,
        )
        unzip_file(archive, dataset_root)
        extracted = os.path.join(dataset_root, "library-master")
        if os.path.exists(extracted):
            os.rename(extracted, musetrainer_root)

    os.makedirs(musetrainer_root, exist_ok=True)
    mxl_files = list(Path(musetrainer_root).rglob("*.mxl"))
    if not mxl_files:
        eprint("No .mxl files found in", musetrainer_root)
        return

    eprint(f"Processing {len(mxl_files)} MuseTrainer files")

    with open(musetrainer_train_index, "w") as index_f:
        file_number = 0
        skipped_files = 0
        with multiprocessing.Pool(processes=_N_WORKERS, maxtasksperchild=2) as p:
            async_results = [
                (path, p.apply_async(_convert_file_impl, (path,))) for path in mxl_files
            ]
            for path, ar in async_results:
                try:
                    result = ar.get(timeout=_TIMEOUT_SECONDS)
                except multiprocessing.TimeoutError:
                    eprint("Timeout processing", path, "skipping")
                    result = []
                except Exception as e:
                    eprint("Error", e)
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
                        f"Processed {file_number}/{len(mxl_files)} files,",
                        f"skipped {skipped_files} files",
                    )

    eprint("Done — index written to", musetrainer_train_index)


if __name__ == "__main__":
    convert_musetrainer()
