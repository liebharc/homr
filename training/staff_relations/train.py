# flake8: noqa: T201

"""
Trains the staff relation model on the pairs of create_dataset and exports it to ONNX.
Usage: poetry run python -m training.staff_relations.train
"""

import argparse
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import nn

from homr.staff_relations import (
    CLASSES,
    channels,
    crop_height,
    crop_width,
    features,
    horizontal_crop_height,
    horizontal_crop_width,
    horizontal_features,
    horizontal_model_path,
    model_path,
)
from homr.type_definitions import NDArray
from training.staff_relations.create_dataset import default_directory, is_validation


class StaffRelationNet(nn.Module):
    def __init__(self, height: int, width: int, n_features: int, n_classes: int) -> None:
        super().__init__()

        def block(inputs: int, outputs: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, padding=1),
                nn.BatchNorm2d(outputs),
                nn.ReLU(),
                nn.Conv2d(outputs, outputs, 3, padding=1),
                nn.BatchNorm2d(outputs),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.convolutions = nn.Sequential(
            block(channels, 24), block(24, 48), block(48, 64), block(64, 96), nn.Conv2d(96, 32, 1)
        )
        flat = 32 * (height // 16) * (width // 16)
        self.head = nn.Sequential(
            nn.Linear(flat + n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, crop: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(crop).flatten(1)
        return self.head(torch.cat([x, feats], dim=1))


def load(
    directories: list[Path], kind: str, cache: Path
) -> dict[str, tuple[NDArray, NDArray, NDArray]]:
    """
    Reads the pairs by split. The crops don't fit into memory, they are written once to
    memory mapped files in the cache directory.
    """
    folders = ["pairs", "pairs_truncated"] if kind == "vertical" else ["pairs_horizontal"]
    files = [
        file
        for directory in directories
        for folder in folders
        for file in sorted((directory / folder).glob("*.npz"))
    ]
    by_split: dict[str, list[Path]] = {"train": [], "val": []}
    for file in files:
        piece = re.search(r"lc\d+|sq\d+", file.stem)
        validation = piece is not None and is_validation(piece.group(0))
        by_split["val" if validation else "train"].append(file)
    cache.mkdir(parents=True, exist_ok=True)
    result: dict[str, tuple[NDArray, NDArray, NDArray]] = {}
    for split, split_files in by_split.items():
        feats, labels = [], []
        for file in split_files:
            with np.load(file) as d:
                feats.append(d["features"])
                labels.append(d["labels"])
        count = sum(len(label) for label in labels)
        with np.load(split_files[0]) as d:
            shape = (count, *d["crops"].shape[1:])
        crops = np.lib.format.open_memmap(
            cache / f"{kind}_{split}.npy", mode="w+", dtype=np.uint8, shape=shape
        )
        start = 0
        for i, file in enumerate(split_files):
            with np.load(file) as d:
                crops[start : start + len(d["labels"])] = d["crops"]
                start += len(d["labels"])
            if i % 1000 == 0:
                print(f"Loaded {split} {i} / {len(split_files)}", flush=True)
        crops.flush()
        result[split] = (crops, np.concatenate(feats), np.concatenate(labels))
    return result


def _batches(
    data: tuple[NDArray, NDArray, NDArray], size: int, shuffle: bool, device: str
) -> Iterator[tuple[torch.Tensor, ...]]:
    crops, feats, labels = data
    order = np.random.permutation(len(labels)) if shuffle else np.arange(len(labels))
    for start in range(0, len(order), size):
        index = np.sort(order[start : start + size])
        yield (
            torch.from_numpy(crops[index]).to(device).float() / 255,
            torch.from_numpy(feats[index]).to(device),
            torch.from_numpy(labels[index]).to(device),
        )


def _augment(crop: torch.Tensor) -> torch.Tensor:
    """Changes the contrast of the image channel, the segmentation stays."""
    gain = torch.empty(crop.shape[0], 1, 1, device=crop.device).uniform_(0.6, 1.4)
    crop = crop.clone()
    crop[:, 0] = (crop[:, 0] * gain).clamp(0, 1)
    return crop


def evaluate(
    model: nn.Module, data: tuple[NDArray, NDArray, NDArray], device: str, n_classes: int
) -> NDArray:
    """Confusion matrix, true class x predicted class."""
    model.eval()
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for crop, feats, labels in _batches(data, 256, False, device):
            predicted = model(crop, feats).argmax(1)
            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy(), strict=True):
                confusion[t, p] += 1
    return confusion


def export(model: nn.Module, path: str, shape: tuple[int, int, int]) -> None:
    height, width, n_features = shape
    model.eval().cpu()
    crop = torch.zeros(1, channels, height, width)
    feats = torch.zeros(1, n_features)
    torch.onnx.export(
        model,
        (crop, feats),
        path,
        input_names=["crop", "features"],
        output_names=["logits"],
        dynamic_axes={"crop": {0: "batch"}, "features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


# Crop height, crop width, features and classes of each kind of pair
shapes = {
    "vertical": (crop_height, crop_width, features, len(CLASSES)),
    "horizontal": (horizontal_crop_height, horizontal_crop_width, horizontal_features, 2),
}
model_paths = {"vertical": model_path, "horizontal": horizontal_model_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", type=Path, nargs="+", default=[default_directory])
    parser.add_argument("--kind", choices=list(shapes), default="vertical")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output")
    parser.add_argument("--cache", type=Path, default=Path("datasets/staff_relations_cache"))
    args = parser.parse_args()
    output = args.output or model_paths[args.kind]
    height, width, n_features, n_classes = shapes[args.kind]
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load(args.directories, args.kind, args.cache)
    for name, (_, _, split_labels) in data.items():
        print(name, len(split_labels), np.bincount(split_labels, minlength=n_classes))
    model = StaffRelationNet(height, width, n_features, n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * ((len(data["train"][2]) + 63) // 64)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, total_steps=steps)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for crop, feats, labels in _batches(data["train"], 64, True, device):
            optimizer.zero_grad()
            loss = loss_function(model(_augment(crop), feats), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(labels)
        confusion = evaluate(model, data["val"], device, n_classes)
        accuracy = np.trace(confusion) / confusion.sum()
        print(
            f"epoch {epoch}: loss {total / len(data['train'][2]):.4f}, "
            f"val accuracy {accuracy:.4f}",
            flush=True,
        )
    print("Confusion (true x predicted):\n", evaluate(model, data["val"], device, n_classes))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(output).with_suffix(".pth"))
    export(model, output, (height, width, n_features))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
