# ruff: noqa: T201

import argparse
import os
from typing import Any

import cv2
import numpy as np
from termcolor import colored

parser = argparse.ArgumentParser(description="Show examples from a dataset index")
parser.add_argument("index", type=str, help="Index file name")
parser.add_argument("number_of_images", type=int, help="Number of images to show at once")
parser.add_argument(
    "--sorted",
    action="store_true",
    help="Show the images in the order they are in the index file",
)
parser.add_argument(
    "--min-ser",
    type=int,
    help="Minimum SER to show",
)
parser.add_argument(
    "--max-ser",
    type=int,
    help="Maximum SER to show",
)
args = parser.parse_args()

index_file_name = args.index
number_of_samples_per_iteration = int(args.number_of_images)

index_file = open(index_file_name)
index_lines = index_file.readlines()
index_file.close()

ser_position = 2

if not args.sorted:
    np.random.shuffle(index_lines)

if args.min_ser is not None:
    index_lines = [
        line for line in index_lines if int(line.split(",")[ser_position]) >= args.min_ser
    ]

if args.max_ser is not None:
    index_lines = [
        line for line in index_lines if int(line.split(",")[ser_position]) <= args.max_ser
    ]


def print_color(text: str, highlights: list[str], color: Any) -> None:
    words = text.split()
    for word in words:
        if any(highlight in word for highlight in highlights):
            print(colored(word, color), end=" ")
        else:
            print(word, end=" ")
    print()


while True:
    batch: list[str] = []
    while len(batch) < number_of_samples_per_iteration:
        if len(index_lines) == 0:
            break

        batch.append(index_lines.pop())

    if len(batch) == 0:
        break

    images = None
    print()
    print()
    print()
    print("==========================================")
    print()
    for line in batch:
        cells = line.strip().split(",")
        image_path = cells[0]
        semantic_path = cells[1]
        ser: None | int = None
        if len(cells) > ser_position:
            ser = int(cells[ser_position])
        agnostic_path = semantic_path.replace(".semantic", ".agnostic")
        image = cv2.imread(image_path)
        with open(semantic_path) as file:
            semantic = file.readline().strip().replace("+", " ")
        if os.path.exists(agnostic_path):
            with open(agnostic_path) as file:
                original_agnostic = file.readline().strip().replace("+", " ")
        else:
            original_agnostic = agnostic_path
        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)
        print()
        if ser is not None:
            print(">>> " + image_path + f" SER: {ser}%")
        else:
            print(">>> " + image_path)
        print_color(semantic, ["barline", "#", "N", "b"], "green")
    cv2.imshow("Images", images)  # type: ignore
    escKey = 27
    if cv2.waitKey(0) == escKey:
        break
