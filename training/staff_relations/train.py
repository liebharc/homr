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
    horizontal_crop_height,
    horizontal_crop_width,
    horizontal_features,
    horizontal_model_path,
    model_path,
    pixels_per_unit,
)
from homr.type_definitions import NDArray
from training.staff_relations.create_dataset import default_directory, is_validation

IMAGE, STAFF, SYMBOLS, _, CLEFS_KEYS = range(channels)


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

    def forward(self, crop: torch.Tensor, feats: torch.Tensor | None = None) -> torch.Tensor:
        x = self.convolutions(crop).flatten(1)
        if feats is not None:
            x = torch.cat([x, feats], dim=1)
        return self.head(x)


def load(
    directories: list[Path], kind: str, cache: Path
) -> dict[str, tuple[NDArray, NDArray, NDArray]]:
    """
    Reads the pairs by split. The crops don't fit into memory, they are written once to
    memory mapped files in the cache directory. Vertical pairs have no features.
    """
    height, width, n_features, _ = shapes[kind]
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
                labels.append(d["labels"])
                feats.append(d["features"] if n_features > 0 else np.zeros((len(d["labels"]), 0)))
        count = sum(len(label) for label in labels)
        crops = np.lib.format.open_memmap(
            cache / f"{kind}_{split}.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(count, channels, height, width),
        )
        start = 0
        for i, file in enumerate(split_files):
            with np.load(file) as d:
                # Older datasets have wider crops, the model uses the left part
                crops[start : start + len(d["labels"])] = d["crops"][..., :width]
                start += len(d["labels"])
            if i % 1000 == 0:
                print(f"Loaded {split} {i} / {len(split_files)}", flush=True)
        crops.flush()
        result[split] = (crops, np.concatenate(feats).astype(np.float32), np.concatenate(labels))
    return result


def _cut_image_border(crop: torch.Tensor, probability: float) -> torch.Tensor:
    """
    Blanks the crop left of a random column with the given probability, like a photo or
    scan whose border cuts the brace off. The bar line at the start of the staffs stays.
    """
    crop = crop.clone()
    staff_columns = nn.functional.max_pool2d(crop[:, STAFF : STAFF + 1], (crop_height, 1), 1)
    has_staff = staff_columns[:, 0, 0] > 0.3  # noqa: PLR2004
    for i in torch.nonzero(torch.rand(crop.shape[0]) < probability)[:, 0].tolist():
        columns = torch.nonzero(has_staff[i])[:, 0]
        if len(columns) == 0:
            continue
        end = int(columns[0]) - pixels_per_unit // 2
        if end > 0:
            crop[i, :, :, : int(torch.randint(0, end + 1, ()))] = 0
    return crop


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
    crop[:, IMAGE] = (crop[:, IMAGE] * gain).clamp(0, 1)
    return crop


def _erase_clefs(crop: torch.Tensor, probability: float) -> torch.Tensor:
    """
    Erases the clefs and keys of each staff of a vertical pair with the given probability,
    old parts often print them only on the first staff. The staff lines stay.
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
    Draws a vertical line over the full height left of the staffs of a vertical pair with
    the given probability, like the page edge of a scan. It connects nothing, the label
    stays.
    """
    crop = crop.clone()
    staff_columns = nn.functional.max_pool2d(crop[:, STAFF : STAFF + 1], (crop_height, 1), 1)
    has_staff = staff_columns[:, 0, 0] > 0.3  # noqa: PLR2004
    for i in torch.nonzero(torch.rand(crop.shape[0]) < probability)[:, 0].tolist():
        columns = torch.nonzero(has_staff[i])[:, 0]
        # At least two units left of the staff start, a line closer to it is a bar line
        space = (int(columns[0]) if len(columns) > 0 else crop_width) - 2 * pixels_per_unit
        width = int(torch.randint(1, 2 * pixels_per_unit + 1, ()))
        if space < width:
            continue
        x = int(torch.randint(0, space - width + 1, ()))
        ink = float(torch.empty(()).uniform_(0.6, 1.0))
        crop[i, IMAGE, :, x : x + width] = crop[i, IMAGE, :, x : x + width].clamp(min=ink)
    return crop


def evaluate(
    model: nn.Module, data: tuple[NDArray, NDArray, NDArray], device: str, n_classes: int
) -> NDArray:
    """Confusion matrix, true class x predicted class."""
    model.eval()
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for crop, feats, labels in _batches(data, 256, False, device):
            predicted = model(crop, feats if feats.shape[1] > 0 else None).argmax(1)
            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy(), strict=True):
                confusion[t, p] += 1
    return confusion


def export(model: nn.Module, path: str, shape: tuple[int, int, int]) -> None:
    """Exports the model, without the features input if it has no features."""
    height, width, n_features = shape
    model.eval().cpu()
    inputs: tuple[torch.Tensor, ...] = (torch.zeros(1, channels, height, width),)
    names = ["crop"]
    if n_features > 0:
        inputs = (*inputs, torch.zeros(1, n_features))
        names.append("features")
    torch.onnx.export(
        model,
        inputs,
        path,
        input_names=names,
        output_names=["logits"],
        dynamic_axes={name: {0: "batch"} for name in [*names, "logits"]},
        opset_version=17,
        dynamo=False,
    )


# Crop height, crop width, features and classes of each kind of pair
shapes = {
    "vertical": (crop_height, crop_width, 0, len(CLASSES)),
    "horizontal": (horizontal_crop_height, horizontal_crop_width, horizontal_features, 2),
}
model_paths = {"vertical": model_path, "horizontal": horizontal_model_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", type=Path, nargs="+", default=[default_directory])
    parser.add_argument("--kind", choices=list(shapes), default="vertical")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--erase-clefs", type=float, default=0.3, help="Vertical pairs only")
    parser.add_argument("--edge-lines", type=float, default=0.3, help="Vertical pairs only")
    parser.add_argument("--image-border", type=float, default=0.2, help="Vertical pairs only")
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
            if args.kind == "vertical":
                augmented = _add_edge_lines(_erase_clefs(crop, args.erase_clefs), args.edge_lines)
                augmented = _cut_image_border(augmented, args.image_border)
                loss = loss_function(model(_augment(augmented)), labels)
            else:
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
