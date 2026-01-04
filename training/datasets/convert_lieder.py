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

from homr.circle_of_fifths import strip_naturals
from homr.download_utils import download_file, unzip_file
from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.transformer.vocabulary import EncodedSymbol, empty
from training.datasets.convert_grandstaff import distort_image
from training.datasets.musescore_svg import (
    SvgMusicFile,
    SvgStaff,
    get_position_from_multiple_svg_files,
)
from training.datasets.music_xml_parser import music_xml_file_to_tokens
from training.transformer.training_vocabulary import (
    calc_ratio_of_tuplets,
    token_lines_to_str,
)

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
lieder = os.path.join(dataset_root, "Lieder-main")
lieder_train_index = os.path.join(lieder, "index.txt")
musescore_path = os.path.join(dataset_root, "MuseScore")


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


class MeasureCutter:
    def __init__(self, voice: list[list[EncodedSymbol]]) -> None:
        self.voice = voice
        self.number_of_staffs = _count_staffs(voice)
        if self.number_of_staffs == 1:
            self.clefs = [EncodedSymbol("clef_G2", empty, empty, empty, "upper")]
        else:
            self.clefs = [
                EncodedSymbol("clef_G2", empty, empty, empty, "upper"),
                EncodedSymbol("clef_F4", empty, empty, empty, "lower"),
            ]
        self.key = EncodedSymbol("keySignature_0")
        self.time = EncodedSymbol("timeSignature/4")

    def _position_to_staff_no(self, symbol: EncodedSymbol) -> int:
        if symbol.position == "lower":
            return 1
        return 0

    def extract_measures(self, count: int) -> list[EncodedSymbol]:
        clefs = self.clefs.copy()
        key = self.key
        time = self.time
        has_time = False
        result: list[EncodedSymbol] = []
        for i in range(count):
            selected_measure = self.voice.pop(0)
            is_first_measure = i == 0
            first_measure_before_any_non_key_or_clef = is_first_measure
            measure_result: list[EncodedSymbol] = []
            for symbol in selected_measure:
                if "clef" in symbol.rhythm:
                    self.clefs[self._position_to_staff_no(symbol)] = symbol
                    if not first_measure_before_any_non_key_or_clef:
                        measure_result.append(symbol)
                    else:
                        clefs[self._position_to_staff_no(symbol)] = symbol
                elif "keySignature" in symbol.rhythm:
                    self.key = symbol
                    if not first_measure_before_any_non_key_or_clef:
                        measure_result.append(symbol)
                    else:
                        key = symbol
                elif "chord" in symbol.rhythm:
                    if not first_measure_before_any_non_key_or_clef:
                        measure_result.append(symbol)
                elif "timeSignature" in symbol.rhythm:
                    self.time = symbol
                    if not first_measure_before_any_non_key_or_clef:
                        measure_result.append(symbol)
                    else:
                        has_time = True
                        time = symbol
                else:
                    first_measure_before_any_non_key_or_clef = False
                    measure_result.append(symbol)

            if is_first_measure:
                if has_time:
                    measure_result.insert(0, time)
                measure_result.insert(0, key)
                for j, clef in enumerate(reversed(clefs)):
                    if j > 0:
                        measure_result.insert(0, EncodedSymbol("chord"))
                    measure_result.insert(0, clef)
            result.extend(measure_result)
        return result


def _split_file_into_staffs(
    voices: list[list[list[EncodedSymbol]]],
    svg_files: list[SvgMusicFile],
    just_token_files: bool,
    fail_if_image_is_missing: bool,
) -> list[str]:
    result: list[str] = []
    splitter = [MeasureCutter(v) for v in voices]
    for svg_file in svg_files:
        png_file = svg_file.filename.replace(".svg", ".png")
        if not just_token_files:
            target_width = 1400
            scale = target_width / svg_file.width
            png_data = cairosvg.svg2png(url=svg_file.filename, scale=scale)
            pil_img = Image.open(BytesIO(png_data))
            image = np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()
        # alternate through voices
        staffs: list[SvgStaff] = sorted(svg_file.staffs.copy(), key=lambda x: x.y)
        current_voice = 0
        staff_number = 0
        while len(staffs) > 0:
            staff_number += 1
            total_staff_area = staffs.pop(0)
            first_staff_height = total_staff_area.height
            measures = splitter[current_voice]
            for _ in range(measures.number_of_staffs - 1):
                total_staff_area = total_staff_area.merge_staff(staffs.pop(0))
            staff_image_file_name = png_file.replace(".png", f"-{staff_number}.png")
            if not just_token_files:
                y_offset = int(random.uniform(1.5, 2.5) * first_staff_height)
                x_offset = 50
                x = total_staff_area.x - x_offset
                y = total_staff_area.y - y_offset
                width = total_staff_area.width + 2 * x_offset
                height = total_staff_area.height + 2 * y_offset
                x = int(x * scale)
                y = int(y * scale)
                width = int(width * scale)
                height = int(height * scale)

                staff_image = image[y : y + height, x : x + width]
                preprocessed = distort_image(staff_image)
                preprocessed = add_image_into_tr_omr_canvas(preprocessed)
                cv2.imwrite(staff_image_file_name, preprocessed)
            elif not os.path.exists(staff_image_file_name) and fail_if_image_is_missing:
                raise ValueError(f"File {staff_image_file_name} not found")

            token_file_name = png_file.replace(".png", f"-{staff_number}.tokens")
            selected_measures: list[EncodedSymbol] = measures.extract_measures(
                total_staff_area.number_of_measures
            )

            if calc_ratio_of_tuplets(selected_measures) <= 0.2:
                selected_measures = strip_naturals(selected_measures)
                tokens_content = token_lines_to_str(selected_measures)
                write_text_to_file(tokens_content, token_file_name)
                result.append(
                    str(Path(staff_image_file_name).relative_to(git_root))
                    + ","
                    + str(Path(token_file_name).relative_to(git_root))
                    + "\n"
                )
            current_voice = (current_voice + 1) % len(voices)
    if any(len(measure) > 0 for measure in voices):
        raise ValueError("Warning: Not all measures were processed")

    return result


def _count_staffs(voice: list[list[EncodedSymbol]]) -> int:
    if len(voice) == 0:
        return 0
    first_measure = voice[0]
    if len(first_measure) == 0:
        return 0
    if len(first_measure) < 3:
        return 0
    third_symbol = first_measure[2]
    if third_symbol.rhythm.startswith("clef"):
        return 2
    return 1


def convert_xml_and_svg_file(
    file: Path, just_token_files: bool, fail_if_image_is_missing: bool = True
) -> list[str]:
    try:
        voices = music_xml_file_to_tokens(str(file))
        number_of_voices = sum([_count_staffs(v) for v in voices])
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
        return _split_file_into_staffs(
            voices, svg_files, just_token_files, fail_if_image_is_missing
        )

    except Exception as e:
        eprint("Error while processing", file, e)
        return []


def _convert_file_only_token(path: Path) -> list[str]:
    return convert_xml_and_svg_file(path, True)


def _convert_token_and_image(path: Path) -> list[str]:
    return convert_xml_and_svg_file(path, False)


def convert_lieder(only_recreate_token_files: bool = False) -> None:
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
        download_file(
            "https://github.com/OpenScore/Lieder/archive/refs/heads/main.zip", lieder_archive
        )
        unzip_file(lieder_archive, dataset_root)

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
