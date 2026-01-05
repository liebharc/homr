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
from fractions import Fraction

import cairosvg
import cv2
import numpy as np
from PIL import Image

from homr.circle_of_fifths import strip_naturals
from homr.download_utils import download_file, untar_file, unzip_file
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
from staff_detection_ideal import main as detect_staff

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
musixqa_root = os.path.join(dataset_root, "musixqa")
musixqa_images = os.path.join(musixqa_root, "images")
musixqa_meta_data = os.path.join(musixqa_root, "metadata.json")
musixqa_train_index = os.path.join(musixqa_root, "index.txt")
musescore_path = os.path.join(dataset_root, "MuseScore")

# Mapping of major keys to circle-of-fifths values
KEY_TO_FIFTHS = {
    "C Major": 0,
    "G Major": 1,
    "D Major": 2,
    "A Major": 3,
    "E Major": 4,
    "B Major": 5,
    "F# Major": 6,
    "C# Major": 7,
    "F Major": -1,
    "Bb Major": -2,
    "Eb Major": -3,
    "Ab Major": -4,
    "Db Major": -5,
    "Gb Major": -6,
    "Cb Major": -7,
}

def duration_to_token(duration_str: str) -> str:
    """
    Convert a fractional duration (e.g. '1/16', '3/16') to a transformer token.
    Base rule:
      - 1/n  -> note_n
      - dotted value multiplies base by 1.5 and is encoded with '.'
    """
    frac = Fraction(duration_str)
    base = frac.denominator // frac.numerator  # e.g. 1/16 -> 16

    # undotted
    if frac == Fraction(1, base):
        return str(base)

    # dotted: base * 1.5 == frac denominator/numerator
    dotted = Fraction(1, base) * Fraction(3, 2)
    if frac == dotted:
        return f"{base}."

    raise ValueError(f"Unsupported duration: {duration_str}")

def convert_piece(piece):
    tokens = []

    # Clef
    tokens.append("clef_G2 . . .")

    # Key signature (circle of fifths)
    fifths = KEY_TO_FIFTHS[piece["key"]]
    tokens.append(f"keySignature_{fifths} . . .")

    # Time signature (denominator only)
    denominator = piece["time_signature"].split("/")[-1]
    tokens.append(f"timeSignature/{denominator} . . .")

    # Notes
    for bar in piece["bars"]:
        for note in bar["staves"]["treble"]:
            dur_token = duration_to_token(note["duration"])
            pitch = note["pitch"]
            tokens.append(f"note_{dur_token} {pitch} _ _ upper")
        tokens.append("barline . . .")

    return ", ".join(tokens)

def convert_musixqa(only_recreate_token_files: bool = False) -> None:
    if not os.path.exists(musixqa_root):
        eprint(
            "Downloading MusiXQA from https://huggingface.co/datasets/puar-playground/MusiXQA"
        )
        os.mkdir(musixqa_root)
        musixqa_image_archive = os.path.join(musixqa_root, "images.tar")
        download_file("https://huggingface.co/datasets/puar-playground/MusiXQA/resolve/main/images.tar?download=true", musixqa_root)
        untar_file(musixqa_image_archive, musixqa_root, zipped=False)

        download_file("https://huggingface.co/datasets/puar-playground/MusiXQA/resolve/main/metadata.json?download=true", musixqa_root)

    with open(musixqa_meta_data, 'r') as f:
        metadata = json.load(f)

    for image in os.listdir(musixqa_images)[0:1000]:
        r = detect_staff(os.path.join(musixqa_images, image))
        p = convert_piece(metadata[image.replace(".png", "")])
        print(r)


if __name__ == "__main__":
    import sys

    multiprocessing.set_start_method("spawn")
    only_recreate_token_files = False
    if "--only-tokens" in sys.argv:
        only_recreate_token_files = True
    convert_musixqa(only_recreate_token_files)
