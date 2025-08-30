import json
import multiprocessing
import os
import platform
import random
import shutil
import stat
import sys
from io import BytesIO
from pathlib import Path

import cairosvg
import cv2
import numpy as np
from PIL import Image

from homr.download_utils import download_file, unzip_file
from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.transformer.vocabulary import SplitSymbol, calc_duration
from training.convert_grandstaff import distort_image
from training.musescore_svg import SvgMusicFile, get_position_from_multiple_svg_files
from training.music_xml_to_tokens import music_xml_file_to_tokens
from training.transformer.training_vocabulary import token_lines_to_str

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
lieder = os.path.join(dataset_root, "Lieder-main")
lieder_train_index = os.path.join(lieder, "index.txt")
musescore_path = os.path.join(dataset_root, "MuseScore")


if platform.system() == "Windows":
    eprint("Transformer training is only implemented for Linux")
    eprint("Feel free to submit a PR to support Windows")
    eprint("Running MuseScore with the -j parameter on Windows doesn't work")
    eprint("https://github.com/musescore/MuseScore/issues/16221")
    sys.exit(1)

if not os.path.exists(musescore_path):
    eprint("Downloading MuseScore from https://musescore.org/")
    download_file(
        "https://cdn.jsdelivr.net/musescore/v4.2.1/MuseScore-4.2.1.240230938-x86_64.AppImage",
        musescore_path,
    )
    os.chmod(musescore_path, stat.S_IXUSR)

if not os.path.exists(lieder):
    eprint("Downloading Lieder from https://github.com/OpenScore/Lieder")
    lieder_archive = os.path.join(dataset_root, "Lieder.zip")
    download_file("https://github.com/OpenScore/Lieder/archive/refs/heads/main.zip", lieder_archive)
    unzip_file(lieder_archive, dataset_root)


def copy_all_mscx_files(working_dir: str, dest: str) -> None:
    for root, _dirs, files in os.walk(working_dir):
        for file in files:
            if file.endswith(".mscx"):
                source = os.path.join(root, file)
                shutil.copyfile(source, os.path.join(dest, file))


def create_formats(source_file: str, formats: list[str]) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []

    # List of files where MuseScore seems to hang up
    files_with_known_issues = [
        "lc6264558",
        "lc5712131",
        "lc5995407",
        "lc5712146",
        "lc6001354",
        "lc6248307",
        "lc5935864",
    ]
    if any(issue in source_file for issue in files_with_known_issues):
        return jobs
    for target_format in formats:
        dirname = os.path.dirname(source_file)
        basename = os.path.basename(source_file)
        out_name = dirname + "/" + basename.replace(".mscx", f".{target_format}")
        out_name_alt = dirname + "/" + basename.replace(".mscx", f"-1.{target_format}")
        if os.path.exists(out_name) or os.path.exists(out_name_alt):
            eprint(out_name, "already exists")
            continue
        job = {
            "in": source_file,
            "out": out_name,
        }
        jobs.append(job)
    return jobs


def _create_musicxml_and_svg_files() -> None:
    dest = os.path.join(lieder, "flat")
    os.makedirs(dest, exist_ok=True)
    copy_all_mscx_files(os.path.join(lieder, "scores"), dest)

    mscx_files = list(Path(dest).rglob("*.mscx"))

    MuseScore = os.path.join(dataset_root, "MuseScore")

    all_jobs = []

    for file in mscx_files:
        jobs = create_formats(str(file), ["musicxml", "svg"])
        all_jobs.extend(jobs)

    if len(all_jobs) == 0:
        eprint("All musicxml were already created, going on with the next step")
        return

    with open("job.json", "w") as f:
        json.dump(all_jobs, f)

    eprint("Starting with", len(all_jobs), "jobs, with the first being", all_jobs[0])

    if os.system(MuseScore + " --force -j job.json") != 0:  # noqa: S605
        eprint("Error running MuseScore")
        os.remove("job.json")
        sys.exit(1)
    os.remove("job.json")


def write_text_to_file(text: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def _split_file_into_staffs(
    voices: list[list[list[SplitSymbol]]], svg_files: list[SvgMusicFile], just_token_files: bool
) -> list[str]:
    voice = 0

    clefs = [SplitSymbol("clef_G2") for _ in range(len(voices))]
    keys = [SplitSymbol("keySignature_0") for _ in range(len(voices))]
    result: list[str] = []
    for svg_file in svg_files:
        png_file = svg_file.filename.replace(".svg", ".png")
        if not just_token_files:
            target_width = 1400
            scale = target_width / svg_file.width
            png_data = cairosvg.svg2png(url=svg_file.filename, scale=scale)
            pil_img = Image.open(BytesIO(png_data))
            image = np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()
        for staff_number, staff in enumerate(svg_file.staffs):
            staff_image_file_name = png_file.replace(".png", f"-{staff_number}.png")
            if not just_token_files:
                y_offset = int(1.5 * staff.height)
                x_offset = 50
                x = staff.x - x_offset
                y = staff.y - y_offset
                width = staff.width + 2 * x_offset
                height = staff.height + 2 * y_offset
                x = int(x * scale)
                y = int(y * scale)
                width = int(width * scale)
                height = int(height * scale)

                staff_image = image[y : y + height, x : x + width]
                margin_top = random.randint(5, 10)
                margin_bottom = random.randint(5, 10)
                preprocessed = add_image_into_tr_omr_canvas(staff_image, margin_top, margin_bottom)
                cv2.imwrite(staff_image_file_name, preprocessed)
                staff_image_file_name = distort_image(staff_image_file_name)
            elif not os.path.exists(staff_image_file_name):
                raise ValueError(f"File {staff_image_file_name} not found")

            token_file_name = png_file.replace(".png", f"-{staff_number}.tokens")
            selected_measures: list[SplitSymbol] = []
            clef = clefs[voice]
            key = keys[voice]
            for i in range(staff.number_of_measures):
                selected_measure = voices[voice].pop(0)
                start_with_clef = False
                has_key = False
                for j, symbol in enumerate(selected_measure):
                    if "clef" in symbol.rhythm:
                        clefs[voice] = symbol
                        start_with_clef = j == 0
                    if "keySignature" in symbol.rhythm:
                        has_key = True
                        keys[voice] = symbol

                if i == 0:
                    if not start_with_clef:
                        if not has_key:
                            selected_measure.insert(0, key)
                        selected_measure.insert(0, clef)
                selected_measures.extend(selected_measure)
            tokens_content = token_lines_to_str(selected_measures)
            write_text_to_file(tokens_content, token_file_name)
            # Only add files to the index which have valid triplets
            if check_triplets(selected_measures):
                result.append(staff_image_file_name + "," + token_file_name + "\n")
            else:
                eprint("Incomplete triplets", staff_image_file_name)
            voice = (voice + 1) % len(voices)
    if any(len(measure) > 0 for measure in voices):
        raise ValueError("Warning: Not all measures were processed")

    return result


def check_triplets(measure: list[SplitSymbol]) -> bool:
    last_symbol_was_chord = False
    number_of_triplets = 0
    for symbol in measure:
        if "chord" in symbol.rhythm:
            last_symbol_was_chord = True
            continue

        if last_symbol_was_chord:
            last_symbol_was_chord = False
            continue

        if "note" in symbol.rhythm or "rest" in symbol.rhythm:
            duration = symbol.rhythm.split("_")[1]
            if "G" in duration:
                continue
            duration = duration.replace(".", "")
            dur = int(duration)
            is_triplet = dur > 0 and dur % 3 == 0
            if is_triplet:
                number_of_triplets += 1

    return number_of_triplets % 3 == 0


def check_duration(measure: list[SplitSymbol]) -> bool:
    last_symbol_was_chord = False
    duration_sum = 0.0
    timeSigBase = 4
    for symbol in measure:
        if symbol.rhythm.startswith("timeSignature"):
            timeSigBase = int(symbol.rhythm.split("/")[1])
            continue
        if "chord" in symbol.rhythm:
            last_symbol_was_chord = True
            continue

        if last_symbol_was_chord:
            last_symbol_was_chord = False
            continue

        if "note" in symbol.rhythm or "rest" in symbol.rhythm:
            duration = symbol.rhythm.split("_")[1]
            if "G" in duration:
                continue
            dots = duration.count(".")
            duration = duration.replace(".", "")
            dur = int(duration)
            if not dur:
                continue
            duration_sum += calc_duration(1.0 / dur, dots) * timeSigBase

    eps = 1e-6
    is_integer_duration = abs(duration_sum - round(duration_sum)) < eps

    return is_integer_duration


def _convert_file(file: Path, just_token_files: bool) -> list[str]:
    try:
        voices = music_xml_file_to_tokens(str(file))
        number_of_voices = len(voices)
        number_of_measures = len(voices[0])
        svg_files = get_position_from_multiple_svg_files(str(file))
        measures_in_svg = [sum(s.number_of_measures for s in file.staffs) for file in svg_files]
        sum_of_measures_in_xml = number_of_measures * number_of_voices
        if sum(measures_in_svg) != sum_of_measures_in_xml:
            # This happens for:
            # The voices have a different number of measures, e.g.
            # because singing starts after a piano intro
            # Special layout decisions, e.g. a key or time change
            # might add a bar at the end of one line,
            # that looks like an extra measure in our SVG parsing
            # (example: second page of lc6570092)
            eprint(
                file,
                "INFO: Number of measures in SVG files",
                sum(measures_in_svg),
                "does not match number of measures in XML",
                sum_of_measures_in_xml,
            )

            return []
        return _split_file_into_staffs(voices, svg_files, just_token_files)

    except Exception as e:
        eprint("Error while processing", file, e)
        return []


def _convert_file_only_token(path: Path) -> list[str]:
    return _convert_file(path, True)


def _convert_token_and_image(path: Path) -> list[str]:
    return _convert_file(path, False)


def convert_lieder(only_recreate_token_files: bool = False) -> None:
    eprint("Indexing Lieder dataset, this can up to several hours.")
    _create_musicxml_and_svg_files()
    music_xml_files = list(Path(os.path.join(lieder, "flat")).rglob("*.musicxml"))
    with open(lieder_train_index, "w") as f:
        file_number = 0
        skipped_files = 0
        with multiprocessing.Pool() as p:
            for result in p.imap_unordered(
                (
                    _convert_file_only_token
                    if only_recreate_token_files
                    else _convert_token_and_image
                ),
                music_xml_files,
            ):
                if len(result) > 0:
                    for line in result:
                        f.write(line)
                    f.flush()
                else:
                    skipped_files += 1
                file_number += 1
                if file_number % 10 == 0:
                    eprint(
                        f"Processed {file_number}/{len(music_xml_files)} files,",
                        f"skipped {skipped_files} files",
                    )
    eprint("Done indexing")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    only_recreate_token_files = False
    if "--only-tokens" in sys.argv:
        only_recreate_token_files = True
    elif len(sys.argv) > 1:
        _convert_token_and_image(Path(sys.argv[1]))
        sys.exit(0)
    convert_lieder(only_recreate_token_files)
