# ruff: noqa: E402

import multiprocessing
import os
import sys
from pathlib import Path

from homr.download_utils import download_file, unzip_file
from homr.simple_logging import eprint

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
musetrainer_root = os.path.join(dataset_root, "musetrainer")
musetrainer_mxl_root = os.path.join(musetrainer_root, "scores")
flat_musetrainer = os.path.join(musetrainer_root, "flat")
rendered_scores = os.path.join(musetrainer_root, "rendered_scores")
musetrainer_train_index = os.path.join(musetrainer_root, "index.txt")

from training.omr_datasets.convert_lieder import (
    convert_file_only_token_musetrainer,
    convert_token_and_image_musetrainer,
    create_musicxml_and_svg_files_from_mxl,
)


def convert_musetrainer(only_recreate_token_files: bool = False) -> None:
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
    os.makedirs(flat_musetrainer, exist_ok=True)
    os.makedirs(rendered_scores, exist_ok=True)
    mxl_files = list(Path(musetrainer_mxl_root).rglob("*.mxl"))
    if not mxl_files:
        eprint("No .mxl files found in", musetrainer_root)
        return

    eprint(f"Processing {len(mxl_files)} MuseTrainer files")

    create_musicxml_and_svg_files_from_mxl(mxl_files, rendered_scores)
    music_xml_files = list(Path(rendered_scores).rglob("*.musicxml"))
    with open(musetrainer_train_index, "w") as f:
        file_number = 0
        skipped_files = 0
        with multiprocessing.Pool(processes=8, maxtasksperchild=2) as p:
            for result in p.imap_unordered(
                (
                    convert_file_only_token_musetrainer
                    if only_recreate_token_files
                    else convert_token_and_image_musetrainer
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

    eprint("Done — index written to", musetrainer_train_index)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    only_recreate_token_files = False
    if "--only-tokens" in sys.argv:
        only_recreate_token_files = True
    convert_musetrainer(only_recreate_token_files)
