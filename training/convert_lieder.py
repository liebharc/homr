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

import cairosvg  # type: ignore
import cv2
import numpy as np
from PIL import Image

import homr.notation_conversions as conversions
from homr.download_utils import download_file, unzip_file
from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from training.convert_grandstaff import distort_image
from training.musescore_svg import (
    SvgMusicFile,
    SvgStaff,
    get_position_from_multiple_svg_files,
)
from training.transformer.kern_tokens import filter_for_kern

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


def _split_file_into_staffs(
    time_sig: str, number_of_staffs: int, measures: list[str], svg_files: list[SvgMusicFile]
) -> list[str]:
    result: list[str] = []
    start_measure = 0
    for svg_file in svg_files:
        png_file = svg_file.filename.replace(".svg", ".png")
        target_width = 1400
        scale = target_width / svg_file.width
        png_data = cairosvg.svg2png(url=svg_file.filename, scale=scale)
        pil_img = Image.open(BytesIO(png_data))
        image = np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()
        for staff_number, staff_group in enumerate(group_array(svg_file.staffs, number_of_staffs)):
            for staff in staff_group:
                if staff.number_of_measures != staff_group[0].number_of_measures:
                    raise ValueError("Different number of measures in the same group")
            staff_image_file_name = png_file.replace(".png", f"-{staff_number}.png")
            max_staff_height = max([s.height for s in staff_group])
            min_staff_x = min([s.x for s in staff_group])
            max_staff_x = max([s.x + s.width for s in staff_group])
            min_staff_y = min([s.y for s in staff_group])
            max_staff_y = max([s.y + s.height for s in staff_group])
            y_offset = max_staff_height
            x_offset = 50
            x = min_staff_x - x_offset
            y = min_staff_y - y_offset
            width = (max_staff_x - min_staff_x) + 2 * x_offset
            height = (max_staff_y - min_staff_y) + 2 * y_offset
            x = int(x * scale)
            y = int(y * scale)
            width = int(width * scale)
            height = int(height * scale)

            staff_image = image[y : y + height, x : x + width]
            margin_top = random.randint(0, 10)
            margin_bottom = random.randint(0, 10)
            preprocessed = add_image_into_tr_omr_canvas(staff_image, margin_top, margin_bottom)
            cv2.imwrite(staff_image_file_name, preprocessed)
            staff_image_file_name = distort_image(staff_image_file_name)

            end_measure = start_measure + staff_group[0].number_of_measures
            kern_lines = [time_sig]
            kern_lines += measures[start_measure:end_measure]
            start_measure = end_measure

            kern_file = staff_image_file_name.replace(".png", ".krn")

            with open(kern_file, "w") as file:
                file.writelines(kern_lines)

            result.append(staff_image_file_name + "," + kern_file + "\n")

    if len(measures) != start_measure:
        raise ValueError(
            "Warning: Not all measures were processed "
            + str(len(measures))
            + " "
            + str(start_measure)
        )

    return result


def group_array(arr: list[SvgStaff], group_size: int) -> list[list[SvgStaff]]:
    return [arr[i : i + group_size] for i in range(0, len(arr), group_size)]


def _split_file_into_pages(
    time_sig: str, number_of_staffs: int, measures: list[str], svg_files: list[SvgMusicFile]
) -> list[str]:
    result: list[str] = []
    start_measure = 0
    for svg_file in svg_files:
        target_width = 1400
        scale = target_width / svg_file.width
        png_data = cairosvg.svg2png(url=svg_file.filename, scale=scale)
        pil_img = Image.open(BytesIO(png_data))
        image = np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()

        margin_top = random.randint(0, 10)
        margin_bottom = random.randint(0, 10)
        preprocessed = add_image_into_tr_omr_canvas(image, margin_top, margin_bottom)
        preprocessed_path = Path(svg_file.filename.replace(".svg", "-pre.png"))
        cv2.imwrite(str(preprocessed_path.absolute()), preprocessed)
        distort_image(str(preprocessed_path.absolute()))
        kern_file = svg_file.filename.replace(".svg", ".krn")
        number_of_kern_measures = len(svg_file.staffs) // number_of_staffs
        kern_lines = [time_sig]
        kern_lines += measures[start_measure : start_measure + number_of_kern_measures]
        start_measure += number_of_kern_measures
        with open(kern_file, "w") as file:
            file.writelines(kern_lines)
        result.append(str(preprocessed_path) + "," + kern_file + "\n")

    return result


def _convert_file(file: Path) -> list[str]:
    try:
        kern_file = conversions.musicxml_to_kern(str(file))
        number_of_staffs, time_sig, measures = split_kern_file_into_measures(kern_file)
        number_of_measures = len(measures)
        svg_files = get_position_from_multiple_svg_files(str(file))
        measures_in_svg = [sum(s.number_of_measures for s in file.staffs) for file in svg_files]
        sum_of_measures_in_kern = number_of_measures * number_of_staffs
        if sum(measures_in_svg) != sum_of_measures_in_kern:
            eprint(
                kern_file,
                "INFO: Number of measures in SVG files",
                sum(measures_in_svg),
                "does not match number of measures in KRN",
                sum_of_measures_in_kern,
            )

            return []
        return _split_file_into_staffs(time_sig, number_of_staffs, measures, svg_files)

    except Exception as e:
        eprint("Error while processing", file, e)
        return []


def split_kern_file_into_measures(kern_file: str) -> tuple[int, str, list[str]]:
    # Return: Number of staffs, key and time sig, measures
    measures = []
    number_of_staffs = 0
    current_measure: list[str] = []
    before_first_measure = ""

    with open(kern_file) as kern:
        lines = kern.readlines()
        lines = filter_for_kern(lines)
        for line in lines:
            if line.startswith("*staff"):
                number_of_staffs = len(line.split())

            if line.startswith("="):
                if before_first_measure == "":
                    before_first_measure = str.join("\n", current_measure) + "\n"
                else:
                    measures.append(str.join("\n", current_measure))
                current_measure = [line]
            else:
                current_measure.append(line)

    return (number_of_staffs, before_first_measure, measures)


def convert_lieder() -> None:
    eprint("Indexing Lieder dataset, this can up to several hours.")
    _create_musicxml_and_svg_files()
    music_xml_files = list(Path(os.path.join(lieder, "flat")).rglob("*.musicxml"))
    with open(lieder_train_index, "w") as f:
        file_number = 0
        skipped_files = 0
        with multiprocessing.Pool() as p:
            for result in p.imap_unordered(
                (_convert_file),
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
    convert_lieder()
