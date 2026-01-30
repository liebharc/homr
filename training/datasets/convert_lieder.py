import json
import multiprocessing
import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from homr.circle_of_fifths import strip_naturals
from homr.download_utils import download_file, unzip_file
from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.transformer.vocabulary import EncodedSymbol, empty
from training.datasets.musescore_svg import (
    SvgMusicFile,
    SvgStaff,
    get_position_from_multiple_svg_files,
)
from training.datasets.music_xml_parser import Measure, music_xml_file_to_tokens
from training.transformer.training_vocabulary import (
    calc_ratio_of_tuplets,
    token_lines_to_str,
)

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
lieder = os.path.join(dataset_root, "Lieder-main")
quartets = os.path.join(dataset_root, "StringQuartets-main")
lieder_train_index = os.path.join(lieder, "index.txt")
musescore_path = os.path.join(dataset_root, "MuseScore")


class MusicXmlPage:
    def __init__(self, voices: list[list[Measure]], number_of_measures: int = 0) -> None:
        if len(voices) > 0:
            self.number_of_measures = len(voices[0])
        else:
            self.number_of_measures = number_of_measures


def split_into_pages(voices: list[list[Measure]]) -> list[MusicXmlPage]:
    """Split voices into pages based on the new_page flag in measures.

    When a measure has new_page=True, it starts a new page.
    All voices must be synchronized - they must split at the same measure indices.

    Raises:
        ValueError: If voices have different lengths or page breaks don't align.
    """
    if not voices:
        return []

    if not voices[0]:
        return []

    # Validate all voices have the same length
    first_voice_len = len(voices[0])
    for i, voice in enumerate(voices[1:], start=1):
        if len(voice) != first_voice_len:
            raise ValueError(
                f"Voice {i} has {len(voice)} measures, but voice 0 has {first_voice_len} measures. "
                "All voices must have the same number of measures."
            )

    # Find page break positions for each voice
    voice_page_breaks: list[list[int]] = []

    for voice in voices:
        page_breaks = [0]  # Start of first page

        for measure_idx, measure in enumerate(voice):
            if hasattr(measure, "new_page") and measure.new_page and measure_idx > 0:
                page_breaks.append(measure_idx)

        page_breaks.append(len(voice))  # End position
        voice_page_breaks.append(page_breaks)

    # Validate all voices have the same page break positions
    reference_breaks = voice_page_breaks[0]
    for voice_idx, breaks in enumerate(voice_page_breaks[1:], start=1):
        if breaks != reference_breaks:
            raise ValueError(
                f"Voice {voice_idx} has page breaks at {breaks[1:-1]}, "
                f"but voice 0 has page breaks at {reference_breaks[1:-1]}. "
                "All voices must have page breaks at the same measure indices."
            )

    # Split all voices at the validated page break positions
    pages: list[MusicXmlPage] = []

    for i in range(len(reference_breaks) - 1):
        start_idx = reference_breaks[i]
        end_idx = reference_breaks[i + 1]

        # Extract measures for this page from all voices
        page_voices: list[list[Measure]] = []
        for voice in voices:
            page_voices.append(voice[start_idx:end_idx])

        pages.append(MusicXmlPage(page_voices))

    return pages


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
    def __init__(self, voice: list[Measure]) -> None:
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


def contains_only_supported_clefs(symbols: list[EncodedSymbol]) -> float:
    for symbol in symbols:
        if symbol.rhythm.startswith("clef_percussion"):
            return False
    return True


def _split_file_into_staffs(
    number_of_voices: int,
    svg_file: SvgMusicFile,
    splitter: list[MeasureCutter],
    just_token_files: bool,
    fail_if_image_is_missing: bool,
) -> list[str]:
    result: list[str] = []
    png_file = svg_file.filename.replace(".svg", ".png")
    if not just_token_files:
        target_width = 1400
        scale = target_width / svg_file.width
        subprocess.run(  # noqa: S603
            [  # noqa: S607
                "rsvg-convert",
                "-w",
                "1400",
                "-o",
                png_file,
                svg_file.filename,
            ],
            check=True,
        )
        pil_img = Image.open(png_file).convert("L")
        image = np.array(pil_img)
    # alternate through voices
    staffs: list[SvgStaff] = sorted(svg_file.staffs.copy(), key=lambda x: x.y)
    current_voice = 0
    staff_number = 0
    while len(staffs) > 0:
        staff_number += 1
        total_staff_area = staffs.pop(0)
        measures = splitter[current_voice]
        staff_image_file_name = png_file.replace(".png", f"-{staff_number}.png")
        if not just_token_files:
            y_offset = 50
            x_offset_right = 10
            x_offset_left = 40
            x = total_staff_area.x - x_offset_left
            y = total_staff_area.y - y_offset
            width = total_staff_area.width + x_offset_right + x_offset_left
            height = total_staff_area.height + 2 * y_offset
            x = int(x * scale)
            y = int(y * scale)
            width = int(width * scale)
            height = int(height * scale)

            staff_image = image[y : y + height, x : x + width]
            preprocessed = add_image_into_tr_omr_canvas(staff_image)
            cv2.imwrite(staff_image_file_name, preprocessed)
        elif not os.path.exists(staff_image_file_name) and fail_if_image_is_missing:
            raise ValueError(f"File {staff_image_file_name} not found")

        token_file_name = png_file.replace(".png", f"-{staff_number}.tokens")
        selected_measures: list[EncodedSymbol] = measures.extract_measures(
            total_staff_area.number_of_measures
        )

        if calc_ratio_of_tuplets(selected_measures) <= 0.2 and contains_only_supported_clefs(
            selected_measures
        ):
            selected_measures = strip_naturals(selected_measures)
            tokens_content = token_lines_to_str(selected_measures)
            write_text_to_file(tokens_content, token_file_name)
            result.append(
                str(Path(staff_image_file_name).relative_to(git_root))
                + ","
                + str(Path(token_file_name).relative_to(git_root))
                + "\n"
            )
        current_voice = (current_voice + 1) % number_of_voices

    return result


def _count_staffs(voice: list[Measure]) -> int:
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


def is_grandstaff(voice: list[Measure]) -> bool:
    if len(voice) == 0:
        return False
    first_measure = voice[0]
    if len(first_measure) < 3:
        return False
    return (
        first_measure[0].rhythm.startswith("clef")
        and first_measure[1].rhythm == "chord"
        and first_measure[2].rhythm.startswith("clef")
    )


def get_svg_voice_count(voice: list[Measure]) -> int:
    """
    The concepts get confusing here: The SVG treats
    a grandstaff as two voices. While in MusicXML it's a
    single voice.
    """
    if is_grandstaff(voice):
        return 2
    return 1


def convert_xml_and_svg_file(
    file: Path, just_token_files: bool, fail_if_image_is_missing: bool = True
) -> list[str]:
    try:
        voices = music_xml_file_to_tokens(str(file))
        splitter = [MeasureCutter(v) for v in voices]
        pages = split_into_pages(voices)
        svg_files = get_position_from_multiple_svg_files(str(file))
        number_of_voices = sum([get_svg_voice_count(voice) for voice in voices])
        for voice_idx, voice in enumerate(voices):
            if is_grandstaff(voice):
                for svg_file in svg_files:
                    svg_file.merge_voice_with_next_one(voice_idx, number_of_voices)
                number_of_voices -= 1

        result: list[str] = []
        if len(pages) != len(svg_files):
            total_svg_measures = sum(svg.number_of_measures for svg in svg_files) / number_of_voices
            total_xml_measures = sum(page.number_of_measures for page in pages)
            if total_xml_measures == total_svg_measures:
                # This happens if the layout required extra pages
                pages = [
                    MusicXmlPage([], svg.number_of_measures // number_of_voices)
                    for svg in svg_files
                ]
            else:
                eprint(
                    file,
                    "INFO: Number of pages in SVG files",
                    len(svg_files),
                    "does not match number of pages in XML",
                    len(pages),
                )

                return []
        for i, page in enumerate(pages):
            svg_file = svg_files[i]
            number_of_measures_per_voice_svg = svg_file.number_of_measures / number_of_voices
            if page.number_of_measures != number_of_measures_per_voice_svg:
                eprint(
                    file,
                    "Page",
                    i + 1,
                    "INFO: Number of measures in SVG files",
                    number_of_measures_per_voice_svg,
                    "does not match number of measures in XML",
                    page.number_of_measures,
                )
                # Remove the measures from the cutter
                for cutter in splitter:
                    cutter.extract_measures(page.number_of_measures)
                continue
            result.extend(
                _split_file_into_staffs(
                    number_of_voices, svg_file, splitter, just_token_files, fail_if_image_is_missing
                )
            )
        return result

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

        eprint("Downloading StringQuartets from https://github.com/OpenScore/StringQuartets")
        quartets_archive = os.path.join(dataset_root, "StringQuartets.zip")
        download_file(
            "https://github.com/OpenScore/StringQuartets/archive/refs/heads/main.zip",
            quartets_archive,
        )
        unzip_file(quartets_archive, dataset_root)
        shutil.copytree(
            os.path.join(quartets, "scores"), os.path.join(lieder, "scores"), dirs_exist_ok=True
        )

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
        eprint(str.join("", _convert_token_and_image(Path(sys.argv[1]))))
        sys.exit(0)
    convert_lieder(only_recreate_token_files)
