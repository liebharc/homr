# flake8: noqa: T201

"""
Trains variants of the vertical staff relation model on the cached crops of train.py,
to find out which inputs make the model generalize beyond Lieder. A variant masks parts
of the crop; the mask is part of the exported model, so homr needs no change.
Usage: poetry run python -m training.staff_relations.train_variants --variant left_stems
"""

import argparse
import re
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from homr.staff_relations import (
    CLASSES,
    channels,
    crop_height,
    crop_width,
    features,
    left_width,
    margin_units,
    outer_rows,
    pixels_per_unit,
    staff_rows,
)
from homr.type_definitions import NDArray
from training.staff_relations.create_dataset import is_validation
from training.staff_relations.train import StaffRelationNet, _augment, export

# Order in which train.py wrote the cache
cache_directories = [
    "datasets/staff_relations",
    "datasets/staff_relations_singles",
    "datasets/staff_relations_melody_pairs",
    "datasets/staff_relations_melody_sb_pairs",
]
IMAGE, STAFF, SYMBOLS, STEMS_RESTS, CLEFS_KEYS = range(channels)
gap_start = outer_rows + staff_rows
gap_end = crop_height - outer_rows - staff_rows
margin_columns = margin_units * pixels_per_unit


def mask_for(variant: str) -> NDArray:
    """1 where the model sees the crop, 0 where it is blanked."""
    mask = np.ones((channels, crop_height, crop_width), dtype=np.float32)
    full = slice(left_width, crop_width)
    if variant == "all":
        pass
    elif variant == "left_only":
        mask[:, :, full] = 0
    elif variant == "left_stems":
        # Full width: only staff lines and bar lines, no notes, text or clefs
        mask[[IMAGE, SYMBOLS, CLEFS_KEYS], :, full] = 0
    elif variant == "no_gap_text":
        # The gap right of the margin without image and symbols: no lyrics
        mask[[IMAGE, SYMBOLS, CLEFS_KEYS], gap_start:gap_end, margin_columns:] = 0
    elif variant == "margin_stems":
        # Margin with all channels, everything else only staff lines and bar lines
        mask[[IMAGE, SYMBOLS, CLEFS_KEYS], :, margin_columns:] = 0
    else:
        raise ValueError(variant)
    return mask


def _erase_clefs(crop: torch.Tensor, probability: float) -> torch.Tensor:
    """
    Erases the clefs and keys of each staff with the given probability, old parts often
    print them only on the first staff. The staff lines stay.
    """
    crop = crop.clone()
    clefs = nn.functional.max_pool2d(crop[:, CLEFS_KEYS : CLEFS_KEYS + 1], 7, 1, 3)[:, 0] > 0.2
    half = crop_height // 2
    chosen = torch.rand(crop.shape[0], 2, device=crop.device) < probability
    rows = torch.arange(crop_height, device=crop.device)
    upper = (rows < half)[None, :, None] & chosen[:, 0, None, None]
    lower = (rows >= half)[None, :, None] & chosen[:, 1, None, None]
    # Only where the staff is, so that no staff lines appear left of its start
    staff_columns = nn.functional.max_pool2d(crop[:, STAFF : STAFF + 1], (crop_height, 1), 1)
    has_staff = staff_columns[:, 0] > 0.3  # noqa: PLR2004
    erase = clefs & (upper | lower) & has_staff
    image = crop[:, IMAGE]
    # The median of each row is the staff line on staff line rows, else the background
    lines = image.median(2, keepdim=True).values.expand_as(image)
    crop[:, IMAGE] = torch.where(erase, lines, image)
    for channel in (SYMBOLS, CLEFS_KEYS):
        crop[:, channel] = torch.where(erase, 0, crop[:, channel])
    return crop


def _add_edge_lines(crop: torch.Tensor, probability: float) -> torch.Tensor:
    """
    Draws a vertical line over the full height left of the staffs with the given
    probability, like the page edge of a scan. It connects nothing, the label stays.
    """
    crop = crop.clone()
    staff_columns = nn.functional.max_pool2d(crop[:, STAFF : STAFF + 1], (crop_height, 1), 1)
    has_staff = staff_columns[:, 0, 0] > 0.3  # noqa: PLR2004
    for i in torch.nonzero(torch.rand(crop.shape[0]) < probability)[:, 0].tolist():
        columns = torch.nonzero(has_staff[i])[:, 0]
        # At least one unit left of the staff start, scans have it close to the brace
        space = (int(columns[0]) if len(columns) > 0 else left_width) - pixels_per_unit
        width = int(torch.randint(1, 2 * pixels_per_unit + 1, ()))
        if space < width:
            continue
        x = int(torch.randint(0, space - width + 1, ()))
        ink = float(torch.empty(()).uniform_(0.6, 1.0))
        crop[i, IMAGE, :, x : x + width] = crop[i, IMAGE, :, x : x + width].clamp(min=ink)
    return crop


class MaskedNet(nn.Module):
    def __init__(self, mask: NDArray, use_features: bool) -> None:
        super().__init__()
        self.register_buffer("mask", torch.from_numpy(mask)[None])
        self.use_features = use_features
        self.net = StaffRelationNet(crop_height, crop_width, features, len(CLASSES))

    def forward(self, crop: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        if not self.use_features:
            feats = feats * 0
        mask = cast(torch.Tensor, self.mask)
        return self.net(crop * mask, feats)


def _labels(split: str, with_truncated: bool) -> tuple[NDArray, NDArray, NDArray]:
    """Features, labels and the rows in the cache to use, in the order of the cache."""
    feats, labels, keep = [], [], []
    for directory in cache_directories:
        for folder in ["pairs", "pairs_truncated"]:
            for file in sorted((Path(directory) / folder).glob("*.npz")):
                piece = re.search(r"lc\d+|sq\d+", file.stem)
                validation = piece is not None and is_validation(piece.group(0))
                if validation != (split == "val"):
                    continue
                with np.load(file) as d:
                    feats.append(d["features"])
                    labels.append(d["labels"])
                    keep.append(np.full(len(d["labels"]), with_truncated or folder == "pairs"))
    return np.concatenate(feats), np.concatenate(labels), np.concatenate(keep)


def load(with_truncated: bool, cache: Path) -> dict[str, tuple[NDArray, NDArray, NDArray, NDArray]]:
    result = {}
    for split in ["train", "val"]:
        crops = np.load(cache / f"vertical_{split}.npy", mmap_mode="r")
        feats, labels, keep = _labels(split, with_truncated)
        if len(labels) != len(crops):
            raise ValueError("The cache doesn't match the pairs, run train.py first")
        rows = np.flatnonzero(keep)
        result[split] = (crops, feats, labels, rows)
    return result


def _batches(data, size: int, shuffle: bool, device: str):  # type: ignore[no-untyped-def]
    crops, feats, labels, rows = data
    order = np.random.permutation(rows) if shuffle else rows
    for start in range(0, len(order), size):
        index = np.sort(order[start : start + size])
        yield (
            torch.from_numpy(crops[index]).to(device).float() / 255,
            torch.from_numpy(feats[index]).to(device),
            torch.from_numpy(labels[index]).to(device),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="all")
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--without-truncated", action="store_true")
    parser.add_argument("--erase-clefs", type=float, default=0.0)
    parser.add_argument("--edge-lines", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", type=Path, default=Path("datasets/staff_relations_cache"))
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load(not args.without_truncated, args.cache)
    model = MaskedNet(mask_for(args.variant), not args.no_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * ((len(data["train"][3]) + 63) // 64)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, total_steps=steps)
    loss_function = nn.CrossEntropyLoss()
    val_data = data["val"]
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for crop, feats, labels in _batches(data["train"], 64, True, device):
            optimizer.zero_grad()
            augmented = _erase_clefs(crop, args.erase_clefs) if args.erase_clefs > 0 else crop
            if args.edge_lines > 0:
                augmented = _add_edge_lines(augmented, args.edge_lines)
            loss = loss_function(model(_augment(augmented), feats), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(labels)
        confusion = _evaluate(model, val_data, device)
        print(
            f"epoch {epoch}: loss {total / len(data['train'][3]):.4f}, "
            f"val accuracy {np.trace(confusion) / confusion.sum():.4f}",
            flush=True,
        )
    print("Confusion (true x predicted):\n", _evaluate(model, val_data, device))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(args.output).with_suffix(".pth"))
    export(model, args.output, (crop_height, crop_width, features))
    print(f"Wrote {args.output}")


def _evaluate(model: nn.Module, data, device: str) -> NDArray:  # type: ignore[no-untyped-def]
    model.eval()
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    with torch.no_grad():
        for crop, feats, labels in _batches(data, 256, False, device):
            predicted = model(crop, feats).argmax(1)
            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy(), strict=True):
                confusion[t, p] += 1
    return confusion


if __name__ == "__main__":
    main()
