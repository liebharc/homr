# flake8: noqa: T201

"""
Trains the number reader of homr.number_recognition on the crops of
training/numbers/dataset.py and exports it to ONNX. Each step cuts the model input from
a context crop with random errors of the position, the staff lines and the scale, as the
attention and the staff detection make them, and degrades it like a scan or a photo.
Usage: poetry run python -m training.numbers.train
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from homr.number_recognition import (
    COUNT_DIGITS,
    CROP_ABOVE,
    CROP_HALF_WIDTH,
    CROP_HEIGHT,
    CROP_PX,
    CROP_WIDTH,
    DENOMINATORS,
    DIGIT_CLASSES,
    NUMERATOR_DIGITS,
    model_path,
)
from homr.type_definitions import NDArray
from training.numbers.dataset import (
    CONTEXT_PX,
    CONTEXT_TOP,
    CONTEXT_WIDTH,
    IGNORE,
    KIND_MULTIREST,
    KIND_NEGATIVE,
    KIND_TIME_SIGNATURE,
    output_root,
)
from training.transformer import image_utils
from training.transformer.augraphy_augment import apply_augraphy

IGNORED = -100  # ignore_index of the loss
# Largest x error of the attention which a positive is trained with, in staff units
MAX_X_ERROR = 2.4


def load(split: str, pattern: str = "**/*.npz") -> tuple[NDArray, NDArray]:
    files = sorted(glob.glob(str(output_root / split / pattern), recursive=True))
    images, labels = [], []
    for file in files:
        data = np.load(file)
        images.append(data["images"])
        labels.append(data["labels"])
    images_array, labels_array = np.concatenate(images), np.concatenate(labels)
    # Time signatures and multirests without a label train nothing
    unusable = ((labels_array[:, 0] == KIND_TIME_SIGNATURE) & (labels_array[:, 1] == IGNORE)) | (
        (labels_array[:, 0] == KIND_MULTIREST) & (labels_array[:, 3] == IGNORE)
    )
    return images_array[~unusable], labels_array[~unusable]


def digit_targets(number: int, slots: int) -> list[int]:
    """Right aligned digit slots, 0 for an empty slot, d + 1 for digit d."""
    if number in (IGNORE, 0):
        return [IGNORED if number == IGNORE else 0] * slots
    text = str(number)[-slots:].rjust(slots)
    return [0 if c == " " else int(c) + 1 for c in text]


def targets(label: NDArray) -> NDArray:
    kind, numerator, denominator, count = (int(v) for v in label)
    if kind == KIND_NEGATIVE:
        numerator = denominator = count = 0
    result = digit_targets(numerator, NUMERATOR_DIGITS)
    result.append(IGNORED if denominator == IGNORE else DENOMINATORS.index(denominator))
    result.extend(digit_targets(count, COUNT_DIGITS))
    return np.array(result, np.int64)


_geometric = image_utils._make_geometric_transforms(allow_occlusions=False)
MOSAIC = 8  # contexts per row and column of a mosaic


def distort_contexts(contexts: list[NDArray]) -> list[NDArray]:
    """
    The augmentations of the transformer training (image_utils.distort_image), without
    its staff warping: it's made for whole staves and would move the digits away from
    the anchor on a crop. Rotation, perspective and text are drawn per context. Ink,
    paper and Augraphy run on mosaics of MOSAIC x MOSAIC contexts, about the size of a
    staff image: their cost is mostly per call.
    """
    images = []
    for context in contexts:
        image = _geometric(image=cv2.cvtColor(context, cv2.COLOR_GRAY2BGR))["image"]
        if np.random.random() < 0.2:  # noqa: PLR2004
            image = image_utils.apply_text_injection(image)
        images.append(image)
    height, width = images[0].shape[:2]
    per_mosaic = MOSAIC * MOSAIC
    result = []
    for start in range(0, len(images), per_mosaic):
        group = images[start : start + per_mosaic]
        tiles = group + [np.full_like(group[0], 255)] * (per_mosaic - len(group))
        mosaic = np.vstack([np.hstack(tiles[r * MOSAIC : (r + 1) * MOSAIC]) for r in range(MOSAIC)])
        ink = image_utils._find_ink(cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY))
        mosaic = image_utils.apply_ink_color_shift(mosaic)
        mosaic = image_utils.apply_background_augmentation(mosaic)
        mosaic = apply_augraphy(mosaic)
        mosaic = image_utils.apply_clahe(mosaic, p=0.1)
        mosaic = image_utils._enforce_min_staff_contrast(mosaic, ink)
        gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
        for i in range(len(group)):
            r, c = divmod(i, MOSAIC)
            result.append(gray[r * height : (r + 1) * height, c * width : (c + 1) * width])
    return result


class Augmenter:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def x_error(self, kind: int) -> float:
        rng = self.rng
        if kind == KIND_NEGATIVE:
            return rng.uniform(-0.5, 0.5)
        sd = 0.6 if kind == KIND_TIME_SIGNATURE else 0.8
        return float(np.clip(rng.normal(0, sd), -MAX_X_ERROR, MAX_X_ERROR))

    def crop(self, context: NDArray, kind: int, augment: bool) -> NDArray:
        rng = self.rng
        dx = self.x_error(kind)
        dy = rng.normal(0, 0.12) if augment else 0.0
        scale = float(np.exp(rng.normal(0, 0.06))) if augment else 1.0
        angle = np.radians(rng.uniform(-2, 2)) if augment else 0.0
        shear = rng.uniform(-0.06, 0.06) if augment else 0.0
        # Crop pixels to staff units around the anchor, units to context pixels
        to_units = np.array(
            [[1 / CROP_PX, 0, -CROP_HALF_WIDTH], [0, 1 / CROP_PX, -CROP_ABOVE], [0, 0, 1]]
        )
        cos, sin = np.cos(angle), np.sin(angle)
        distortion = np.array(
            [[scale * cos, scale * (-sin + shear), dx], [scale * sin, scale * cos, dy], [0, 0, 1]]
        )
        to_context = np.array(
            [[CONTEXT_PX, 0, CONTEXT_WIDTH / 2], [0, CONTEXT_PX, CONTEXT_TOP], [0, 0, 1]]
        )
        matrix = (to_context @ distortion @ to_units)[:2].astype(np.float32)
        if augment and rng.random() < 0.3:  # noqa: PLR2004
            k = 2
            kernel = np.ones((k, k), np.uint8)
            # Thicker or thinner print, ink is dark
            context = (
                cv2.erode(context, kernel) if rng.random() < 0.6 else cv2.dilate(context, kernel)
            )  # noqa: PLR2004
        crop = cv2.warpAffine(
            context,
            matrix,
            (CROP_WIDTH, CROP_HEIGHT),
            flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        return self.degrade(crop) if augment else crop

    def degrade(self, crop: NDArray) -> NDArray:
        rng = self.rng
        image: NDArray = crop.astype(np.float32) / 255
        if rng.random() < 0.5:  # noqa: PLR2004
            # Low resolution
            factor = rng.uniform(0.45, 0.9)
            small = cv2.resize(
                image,
                (max(1, round(CROP_WIDTH * factor)), max(1, round(CROP_HEIGHT * factor))),
                interpolation=cv2.INTER_AREA,
            )
            image = cv2.resize(small, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.3:  # noqa: PLR2004
            image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.0))
        result: NDArray = np.clip(image * 255, 0, 255).astype(np.uint8)
        if rng.random() < 0.15:  # noqa: PLR2004
            # Binarized, like a black and white scan
            _, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if rng.random() < 0.2:  # noqa: PLR2004
            quality = int(rng.integers(30, 90))
            _, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
            result = np.asarray(cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE))
        return result


def _seed_worker(worker_id: int) -> None:
    # Each worker gets its own augmentations and one thread, the workers fill the cores
    np.random.seed(torch.initial_seed() % 2**32)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class CropDataset(Dataset):
    def __init__(self, images: NDArray, labels: NDArray, augment: bool) -> None:
        self.images = images
        self.labels = labels
        self.augment = augment
        self.targets = np.stack([targets(label) for label in labels])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """A whole batch, the mosaics need many contexts at once."""
        if not self.augment:
            # Validation crops are augmented too, but the same in every epoch. The
            # transformer's augmentations draw from numpy's global generator.
            np.random.seed(indices[0])
        augmenter = Augmenter(np.random.default_rng(None if self.augment else indices[0]))
        contexts = distort_contexts([self.images[i] for i in indices])
        crops = [
            augmenter.crop(context, int(self.labels[i][0]), augment=True)
            for i, context in zip(indices, contexts, strict=True)
        ]
        x = torch.from_numpy(np.stack(crops)).float().div(255).unsqueeze(1)
        return x, torch.from_numpy(self.targets[indices])


class NumberNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(inputs: int, outputs: int, pool: bool = True) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(inputs, outputs, 3, padding=1),
                nn.BatchNorm2d(outputs),
                nn.ReLU(),
                nn.Conv2d(outputs, outputs, 3, padding=1),
                nn.BatchNorm2d(outputs),
                nn.ReLU(),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(1, 32), block(32, 64), block(64, 128), block(128, 128, pool=False)
        )
        flat = 128 * (CROP_HEIGHT // 8) * (CROP_WIDTH // 8)
        self.trunk = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(flat, 384), nn.ReLU(), nn.Dropout(0.2)
        )
        self.numerator = nn.Linear(384, NUMERATOR_DIGITS * DIGIT_CLASSES)
        self.denominator = nn.Linear(384, len(DENOMINATORS))
        self.count = nn.Linear(384, COUNT_DIGITS * DIGIT_CLASSES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(self.features(x))
        return (
            self.numerator(h).view(-1, NUMERATOR_DIGITS, DIGIT_CLASSES),
            self.denominator(h),
            self.count(h).view(-1, COUNT_DIGITS, DIGIT_CLASSES),
        )


def losses(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], y: torch.Tensor
) -> torch.Tensor:
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORED)
    numerator, denominator, count = outputs
    n = NUMERATOR_DIGITS
    loss = loss_fn(numerator.reshape(-1, DIGIT_CLASSES), y[:, :n].reshape(-1))
    loss = loss + loss_fn(denominator, y[:, n])
    loss = loss + loss_fn(count.reshape(-1, DIGIT_CLASSES), y[:, n + 1 :].reshape(-1))
    return loss


def evaluate(model: nn.Module, loader: DataLoader, labels: NDArray, device: str) -> dict:
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in loader:
            numerator, denominator, count = model(x.to(device))
            predictions.append(
                torch.cat(
                    [numerator.argmax(-1), denominator.argmax(-1)[:, None], count.argmax(-1)], 1
                ).cpu()
            )
    predicted = torch.cat(predictions).numpy()
    truth = np.stack([targets(label) for label in labels])
    n = NUMERATOR_DIGITS
    kind = labels[:, 0]
    time_signature = kind == KIND_TIME_SIGNATURE
    multirest = kind == KIND_MULTIREST
    negative = kind == KIND_NEGATIVE
    time_correct = (predicted[:, : n + 1] == truth[:, : n + 1]).all(1)
    count_correct = (predicted[:, n + 1 :] == truth[:, n + 1 :]).all(1)
    return {
        "time_signature": float(time_correct[time_signature].mean()),
        "multirest": float(count_correct[multirest].mean()),
        "negative_time": float((predicted[negative, : n + 1] != 0).any(1).mean()),
        "negative_count": float((predicted[negative, n + 1 :] != 0).any(1).mean()),
    }


def train(epochs: int, output: str, workers: int) -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_images, train_labels = load("train")
    val_images, val_labels = load("val")
    print("Samples", len(train_images), "validation", len(val_images), flush=True)
    train_set = CropDataset(train_images, train_labels, augment=True)
    train_loader = DataLoader(
        train_set,
        batch_size=None,
        sampler=BatchSampler(RandomSampler(train_set), 256, drop_last=True),
        num_workers=workers,
        persistent_workers=True,
        worker_init_fn=_seed_worker,
    )
    val_set = CropDataset(val_images, val_labels, augment=False)
    val_loader = DataLoader(
        val_set,
        batch_size=None,
        sampler=BatchSampler(SequentialSampler(val_set), 512, drop_last=False),
        num_workers=workers,
        worker_init_fn=_seed_worker,
    )
    model = NumberNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, total_steps=epochs * len(train_loader)
    )
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            loss = losses(model(x.to(device)), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        metrics = evaluate(model, val_loader, val_labels, device)
        print(
            f"epoch {epoch} loss {total / len(train_loader):.4f}",
            " ".join(f"{k} {v:.4f}" for k, v in metrics.items()),
            flush=True,
        )

    model.eval().cpu()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(1, 1, CROP_HEIGHT, CROP_WIDTH),),
        output,
        input_names=["crops"],
        output_names=["numerator", "denominator", "count"],
        dynamic_axes={
            "crops": {0: "batch"},
            "numerator": {0: "batch"},
            "denominator": {0: "batch"},
            "count": {0: "batch"},
        },
        dynamo=False,
    )
    print("Wrote", output)


def write_examples(path: str, count: int = 50) -> None:
    """Augmented model inputs with their labels, to check them by eye."""
    images, labels = load("train", "verovio_000.npz")
    rng = np.random.default_rng(3)
    order = rng.permutation(len(images))[:count]
    tiles = []
    contexts = distort_contexts([images[i] for i in order])
    for i, context in zip(order, contexts, strict=True):
        crop = Augmenter(rng).crop(context, int(labels[i][0]), augment=True)
        tile = cv2.copyMakeBorder(crop, 12, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
        kind, numerator, denominator, number = (int(v) for v in labels[i])
        text = {0: "-", 1: f"{numerator}/{denominator}", 2: f"#{number}"}[kind]
        cv2.putText(tile, text, (1, 9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 0, 1)
        tiles.append(tile)
    columns = 25
    rows = [np.hstack(tiles[r : r + columns]) for r in range(0, len(tiles) - columns + 1, columns)]
    cv2.imwrite(path, np.vstack(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--output", default=model_path)
    parser.add_argument("--examples", help="Only write example inputs to this image")
    args = parser.parse_args()
    if args.examples:
        write_examples(args.examples)
    else:
        train(args.epochs, args.output, args.workers)
