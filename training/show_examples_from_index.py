# ruff: noqa: T201

import os
import sys
from typing import Any

import cv2
import numpy as np
from termcolor import colored

index_file_name = sys.argv[1]
number_of_samples_per_iteration = int(sys.argv[2])

index_file = open(index_file_name)
index_lines = index_file.readlines()
index_file.close()

np.random.shuffle(index_lines)


def print_color(text: str, highlights: list[str], color: Any) -> None:
    words = text.split()
    for word in words:
        if any(highlight in word for highlight in highlights):
            print(colored(word, color), end=" ")
        else:
            print(word, end=" ")
    print()


while True:
    batch = []
    for _ in range(number_of_samples_per_iteration):
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
        image_path, semantic_path = line.strip().split(",")
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
        print(">>> " + image_path)
        print_color(semantic, ["barline", "#", "N", "b"], "green")
    cv2.imshow("Images", images)  # type: ignore
    escKey = 27
    if cv2.waitKey(0) == escKey:
        break
