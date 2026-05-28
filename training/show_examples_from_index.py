# ruff: noqa: T201

import argparse
from typing import Any

import cv2
import numpy as np
from termcolor import colored

from homr.staff_parsing import add_image_into_tr_omr_canvas
from training.transformer.image_utils import distort_image

parser = argparse.ArgumentParser(description="Show examples from a dataset index")
parser.add_argument("index", type=str, help="Index file name")
parser.add_argument("number_of_images", type=int, help="Number of images to show at once")
parser.add_argument(
    "--sorted",
    action="store_true",
    help="Show the images in the order they are in the index file",
)
parser.add_argument(
    "--no-augmentation",
    action="store_true",
    help="Skip augmentation/distortion",
)
parser.add_argument(
    "--search",
    type=str,
    help="Only token files which contain this search term will be displayed",
)
args = parser.parse_args()

index_file_name = args.index
search = args.search
number_of_samples_per_iteration = int(args.number_of_images)

index_file = open(index_file_name)
index_lines = index_file.readlines()
index_file.close()


if not args.sorted:
    np.random.shuffle(index_lines)


def print_color(text: str, highlights: list[str], color: Any) -> None:
    for line in text.splitlines():
        words = line.split()
        for word in words:
            if any(highlight in word for highlight in highlights):
                print(colored(word, color), end=" ")
            else:
                print(word, end=" ")
        print()


done = False
highlight = ["barline", "#", "N", "b"]
if search:
    highlight.append(search)
while not done:
    batch: list[tuple[str, str]] = []
    while len(batch) < number_of_samples_per_iteration:
        if len(index_lines) == 0:
            break

        line = index_lines.pop()
        cells = line.strip().split(",")

        with open(cells[1]) as file:
            tokens = str.join("", file.readlines())

        if search and search not in tokens:
            continue

        batch.append((cells[0], tokens))

    if len(batch) == 0:
        break

    images = None
    print()
    print()
    print()
    print("==========================================")
    print()
    for image_path, tokens in batch:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read " + image_path)
        if not args.no_augmentation:
            image = distort_image(image)
            image = add_image_into_tr_omr_canvas(image)
        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)
        print()
        print(">>> " + image_path)
        print_color(tokens, highlight, "green")
        if search:
            print("Count:", tokens.count(search))
    cv2.imshow("Images", images)  # type: ignore
    escKey = 27
    spaceKey = 32
    enterKey = 13
    while True:
        key = cv2.waitKey(0)
        if key == escKey:
            done = True
            break
        if key in (spaceKey, enterKey):
            break
