# ruff: noqa: T201

import difflib
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.staff2score import Staff2Score
from training.transformer.training_vocabulary import read_tokens, token_lines_to_str


def _normalize(token: str) -> str:
    token = token.replace("..", ".")
    token = (
        token.replace(" tieStart ", " _ ")
        .replace(" tieStop ", " _ ")
        .replace(" tieStart_tieStop ", " _ ")
    )
    token = (
        token.replace(" slurStart ", " _ ")
        .replace(" slurStop ", " _ ")
        .replace(" slurStart_slurStop ", " _ ")
    )
    return token.strip() + "\n"


def diff_index(index_path: str) -> None:
    git_root = Path(__file__).parent.parent.absolute()

    if not os.path.exists(index_path):
        eprint(f"Error: Index file not found: {index_path}")
        sys.exit(1)

    # Initialize model
    config = Config()
    model = Staff2Score(config)

    with open(index_path, "r") as f:
        lines = f.readlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) != 2:
            eprint(f"Skipping malformed line: {line}")
            continue

        img_rel_path, tokens_rel_path = parts
        img_abs_path = os.path.join(git_root, img_rel_path)
        tokens_abs_path = os.path.join(git_root, tokens_rel_path)

        if not os.path.exists(img_abs_path):
            eprint(f"Image not found: {img_abs_path}")
            continue
        if not os.path.exists(tokens_abs_path):
            eprint(f"Tokens not found: {tokens_abs_path}")
            continue

        # Load ground truth
        gt_symbols = read_tokens(tokens_abs_path)
        gt_str = token_lines_to_str(gt_symbols)

        # Predict
        image = Image.open(img_abs_path).convert("L")
        pred_symbols = model.predict(np.array(image))
        pred_str = token_lines_to_str(pred_symbols)

        gt = gt_str.splitlines(keepends=True)
        pred = pred_str.splitlines(keepends=True)

        gt = [_normalize(g) for g in gt]
        pred = [_normalize(p) for p in pred]

        if gt != pred:
            print(f"--- {img_rel_path} ---")
            diff = difflib.unified_diff(gt, pred, fromfile="Ground Truth", tofile="Prediction")
            sys.stdout.writelines(diff)
            print("\n")
        else:
            eprint(f"OK: {img_rel_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python training/diff_index.py <index_file>")
        sys.exit(1)

    diff_index(sys.argv[1])
