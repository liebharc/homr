# ruff: noqa: E402

import json
import multiprocessing
import os
import platform
import re
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
from homr.transformer.vocabulary import EncodedSymbol, empty
from training.omr_datasets.musescore_svg import (
    SvgMusicFile,
    SvgStaff,
    get_position_from_multiple_svg_files,
)
from training.omr_datasets.music_xml_parser import Measure, music_xml_file_to_tokens
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

    # sq8940236: MuseScore seems to hang up
    # lc5001945: nested tuplet, not good for training
    # lc6209608, lc6236149: empty&invisible staff in the very first system
    # lc6162644: irregular staff in the last svg page
    files_with_known_issues = ["sq8940236", "lc5001945", "lc6162644", "lc6209608", "lc6236149"]
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


"""
In mscx file, tuplet can be set invisible via the following ways:
1. <Tuplet( id="...")>
     <visible>0</visible>
   </Tuplet>
    this is equal to <notations print-object="no"> in musicXML
2. <numberType>2</numberType>
    this is equal to <tuplet show-number="none" ...> in musicXML
3. <bracketType>2</bracketType>
    this is equal to <tuplet bracket="no" ...> in musicXML
4. numberType & bracketType can be set globally via <tupletNumberType> & <tupletBracketType>

recommend reading lc5033057 from Lieder to better understand the rules about tuplet visibility.
"""


def _make_tuplet_visible(mscx_file: str) -> None:
    with open(mscx_file, encoding="utf-8") as f:
        content = f.read()

    # match: <visible>0</visible> or <numberType>2</numberType>
    # or <bracketType>2</bracketType>
    strip_re = re.compile(
        r"[ \t]*<(?:visible>0|numberType>2|bracketType>2)"
        r"</(?:visible|numberType|bracketType)>[ \t]*\r?\n"
    )

    def strip_hidden(match: "re.Match[str]") -> str:
        return strip_re.sub("", match.group(0))

    # match within <Tuplet> ... </Tuplet>
    new_content = re.sub(
        r"<Tuplet(?:\s[^>]*)?>.*?</Tuplet>", strip_hidden, content, flags=re.DOTALL
    )

    # match globally: <tupletNumberType>2</tupletNumberType>
    # or <tupletBracketType>2</tupletBracketType>
    new_content = re.sub(
        r"[ \t]*<(?:tupletNumberType|tupletBracketType)>2"
        r"</(?:tupletNumberType|tupletBracketType)>[ \t]*\r?\n",
        "",
        new_content,
    )

    if new_content != content:
        with open(mscx_file, "w", encoding="utf-8") as f:
            f.write(new_content)


def _make_staff_visible(mscx_file: str) -> None:
    """
    When rendering SVG, MuseScore can hide empty staff.
    This will cause problem in SvgMusicFile.merge_voice_with_next_one(), because the merge procedure
    assumes that all staffs are visible and deals with the staff one by one.

    To avoid hiding empty staff, we need to change the following 3 settings:
      * the global setting: <hideEmptyStaves>1</hideEmptyStaves>
      * the per-staff setting: <hideWhenEmpty>1</hideWhenEmpty>
      * <cutaway>1</cutaway>: this hides part of the staff, e.g. some empty measures.

    Note: unfortunatelly, this does not cover every case.
    lc6236149, lc6209608 still fails, so we just skip them.
    """
    with open(mscx_file, encoding="utf-8") as f:
        content = f.read()

    new_content = re.sub(
        r"<hideEmptyStaves>1</hideEmptyStaves>",
        "<hideEmptyStaves>0</hideEmptyStaves>",
        content,
    )
    new_content = re.sub(
        r"<hideWhenEmpty>1</hideWhenEmpty>",
        "<hideWhenEmpty>0</hideWhenEmpty>",
        new_content,
    )
    new_content = re.sub(
        r"<cutaway>1</cutaway>",
        "<cutaway>0</cutaway>",
        new_content,
    )

    if new_content != content:
        with open(mscx_file, "w", encoding="utf-8") as f:
            f.write(new_content)


def _create_musicxml_and_svg_files() -> None:
    dest = os.path.join(lieder, "flat")
    os.makedirs(dest, exist_ok=True)
    copy_all_mscx_files(os.path.join(lieder, "scores"), dest)

    mscx_files = list(Path(dest).rglob("*.mscx"))

    MuseScore = os.path.join(dataset_root, "MuseScore")

    all_jobs = []

    for file in mscx_files:
        _make_tuplet_visible(str(file))
        _make_staff_visible(str(file))
        jobs = create_formats(str(file), ["musicxml", "svg"])
        all_jobs.extend(jobs)

    if len(all_jobs) == 0:
        eprint("All musicxml were already created, going on with the next step")
        return

    eprint("Starting with", len(all_jobs), "jobs")

    BATCH_SIZE = 50
    failed_files: list[str] = []

    batches = [all_jobs[i : i + BATCH_SIZE] for i in range(0, len(all_jobs), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        eprint(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} jobs)")

        with open("job.json", "w") as f:
            json.dump(batch, f)

        if os.system(MuseScore + " --force -j job.json") == 0:  # noqa: S605
            os.remove("job.json")
            continue

        env = os.environ.copy()
        # No need to run GUI, so we can use offscreen backend
        env["QT_QUICK_BACKEND"] = "software"
        env["QT_QPA_PLATFORM"] = "offscreen"

        # Batch failed - retry each job individually
        eprint(f"Batch {batch_idx + 1} failed, retrying individually")
        os.remove("job.json")

        for job in batch:
            with open("job.json", "w") as f:
                json.dump([job], f)

            if os.system(MuseScore + " --force -j job.json") != 0:  # noqa: S605
                eprint("Failed:", job["in"])
                failed_files.append(job["in"])

            if os.path.exists("job.json"):
                os.remove("job.json")

    if failed_files:
        eprint(f"\nMuseScore export finished with {len(failed_files)} failed file(s):")
        for path in failed_files:
            eprint(" ", path)
    else:
        eprint("MuseScore export completed with no failures.")


def write_text_to_file(text: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(text)


class MeasureCutter:
    def __init__(self, voice: list[Measure]) -> None:
        self.voice = voice
        self.number_of_staffs = _count_staffs(voice)
        if self.number_of_staffs == 1:
            self.clefs = [EncodedSymbol("clef_G2", empty, empty, empty, empty, "upper")]
        else:
            self.clefs = [
                EncodedSymbol("clef_G2", empty, empty, empty, empty, "upper"),
                EncodedSymbol("clef_F4", empty, empty, empty, empty, "lower"),
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
    image = None
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

            staff_image = image[y : y + height, x : x + width]  # type: ignore
            cv2.imwrite(staff_image_file_name, staff_image)
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

    if image is not None:
        del image

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
        assert len(pages) == len(svg_files)  # noqa: S101

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
            "https://github.com/musescore/MuseScore/releases/download/v4.6.5/MuseScore-Studio-4.6.5.253511702-x86_64.AppImage",
            musescore_path,
        )

        perms = (
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        )

        os.chmod(musescore_path, perms)  # chmod 755

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
        with multiprocessing.Pool(processes=8, maxtasksperchild=2) as p:
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
