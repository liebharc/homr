# flake8: noqa: T201

"""
Trains the digit classifier of homr.number_recognition on synthetic crops and exports
it to ONNX. Digits come from the SMuFL fonts of verovio and from the system fonts,
the "other" class from all other music glyphs and letters.
Usage: poetry run python -m training.numbers.train
"""

import argparse
import io
import os
import random
import re
import subprocess

import cairosvg
import cv2
import numpy as np
import torch
import verovio
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from homr.number_recognition import CLASSES, OTHER, crop_size, crop_units, model_path
from homr.type_definitions import NDArray

# Pixels per staff unit while composing a crop, it's scaled down to crop_size afterwards
UNIT = 24
SMUFL_UNIT = 250  # font units per staff space
SMUFL_FONTS = ["Bravura", "Leland", "Leipzig", "Gootville", "Petaluma"]
# SMuFL glyphs of digits by name, e.g. timeSig3, tuplet3, fingering3, figbass3
DIGIT_GLYPH = re.compile(r"^(timeSig|tuplet|fingering|figbass)(\d)(Large|Narrow|Small)?$")
# Glyphs which contain digits or are too large for a single symbol
SKIPPED_GLYPH = re.compile(
    r"^(timeSig|tuplet|fingering|figbass|functionNumber|ottava|fretboard|string|brace|bracket)"
)
EXCLUDED_FONTS = re.compile(
    r"(?i)unifont|wqy|ipa|droid|emoji|StandardSymbols|D050000L|fonts-japanese|Math"
)
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()[].,:;-!?&%+="


def _ink_crop(image: NDArray) -> NDArray:
    """Ink as 255, cropped to its bounding box."""
    ys, xs = np.where(image > 20)  # noqa: PLR2004
    if len(xs) == 0:
        return image[:0, :0]
    return image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _smufl_glyph(font_dir: str, code: str, box: dict[str, float]) -> NDArray | None:
    path = os.path.join(font_dir, code + ".xml")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read()
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    if w <= 0 or h <= 0:
        return None
    scale = UNIT / SMUFL_UNIT
    pad = 10
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w * scale + 2 * pad}" '
        f'height="{h * scale + 2 * pad}" viewBox="{x - pad / scale} {-(y + h) - pad / scale} '
        f'{w + 2 * pad / scale} {h + 2 * pad / scale}"><rect width="100%" height="100%" '
        f'fill="black" x="{x - pad / scale}" y="{-(y + h) - pad / scale}"/>'
        f'<g fill="white">{content}</g></svg>'
    )
    png = cairosvg.svg2png(bytestring=svg.encode())
    image = np.array(Image.open(io.BytesIO(png)).convert("L"))
    crop = _ink_crop(image)
    return crop if crop.size else None


def load_smufl() -> tuple[list[tuple[int, NDArray]], list[NDArray]]:
    """Digits as (digit, image) and other glyphs, at UNIT pixels per staff space."""
    data = os.path.join(os.path.dirname(verovio.__file__), "data")
    digits: list[tuple[int, NDArray]] = []
    others: list[NDArray] = []
    for font in SMUFL_FONTS:
        with open(os.path.join(data, font + ".xml")) as f:
            metadata = f.read()
        boxes = {
            m[0]: (m[5], {"x": float(m[1]), "y": float(m[2]), "w": float(m[3]), "h": float(m[4])})
            for m in re.findall(
                r'<g c="([0-9A-F]+)" x="([-\d.]+)" y="([-\d.]+)" w="([-\d.]+)" h="([-\d.]+)"'
                r'[^>]* n="(\w+)"',
                metadata,
            )
        }
        for code, (name, box) in boxes.items():
            digit = DIGIT_GLYPH.match(name)
            if not digit and SKIPPED_GLYPH.match(name):
                continue
            glyph = _smufl_glyph(os.path.join(data, font), code, box)
            if glyph is None or max(glyph.shape) > 12 * UNIT:
                continue
            if digit:
                digits.append((int(digit[2]), glyph))
            else:
                others.append(glyph)
    return digits, others


def _system_fonts() -> list[str]:
    output = subprocess.run(
        ["fc-list", ":", "file"], capture_output=True, text=True, check=True  # noqa: S607
    ).stdout
    files = sorted({line.split(":")[0].strip() for line in output.splitlines()})
    return [f for f in files if f.endswith((".ttf", ".otf")) and not EXCLUDED_FONTS.search(f)]


def _text_glyph(font: ImageFont.FreeTypeFont, char: str) -> NDArray | None:
    image = Image.new("L", (300, 300), 0)
    ImageDraw.Draw(image).text((50, 50), char, font=font, fill=255)
    crop = _ink_crop(np.array(image))
    return crop if crop.size else None


def load_text() -> tuple[list[dict[int, NDArray]], list[NDArray]]:
    """The digits of each system font and the letters of all of them."""
    fonts: list[dict[int, NDArray]] = []
    others: list[NDArray] = []
    for path in _system_fonts():
        try:
            font = ImageFont.truetype(path, 140)
        except OSError:
            continue
        digits = {}
        for d in range(10):
            glyph = _text_glyph(font, str(d))
            if glyph is not None:
                digits[d] = glyph
        if len(digits) == 10:  # noqa: PLR2004
            fonts.append(digits)
        for char in LETTERS:
            glyph = _text_glyph(font, char)
            if glyph is not None:
                others.append(glyph)
    return fonts, others


class Glyphs:
    def __init__(self) -> None:
        self.music_digits, self.music_others = load_smufl()
        self.text_fonts, self.text_others = load_text()
        self.text_digits = [(d, g) for font in self.text_fonts for d, g in font.items()]
        print(
            "Glyphs: music digits",
            len(self.music_digits),
            "music others",
            len(self.music_others),
            "text digits",
            len(self.text_digits),
            "letters",
            len(self.text_others),
        )


def _paste(canvas: NDArray, glyph: NDArray, cx: float, cy: float) -> tuple[int, int, int, int]:
    """Draws the glyph with its center at cx, cy and returns its box (x0, y0, x1, y1)."""
    h, w = glyph.shape
    x0 = round(cx - w / 2)
    y0 = round(cy - h / 2)
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x0 + w), min(canvas.shape[0], y0 + h)
    if cx1 > cx0 and cy1 > cy0:
        region = canvas[cy0:cy1, cx0:cx1]
        np.maximum(region, glyph[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0], out=region)
    return x0, y0, x0 + w, y0 + h


def _scaled(glyph: NDArray, height: float) -> NDArray:
    factor = max(height, 2) / glyph.shape[0]
    size = (max(1, round(glyph.shape[1] * factor)), max(1, round(height)))
    return cv2.resize(glyph, size, interpolation=cv2.INTER_AREA)


class Generator:
    def __init__(self, glyphs: Glyphs, rng: random.Random) -> None:
        self.glyphs = glyphs
        self.rng = rng

    def _digit(self, prefer_music: bool) -> tuple[int, NDArray, bool]:
        """A digit, its image and whether it's a music font digit in its own size."""
        if prefer_music and self.rng.random() < 0.6:  # noqa: PLR2004
            digit, glyph = self.rng.choice(self.glyphs.music_digits)
            return digit, glyph, True
        digit, glyph = self.rng.choice(self.glyphs.text_digits)
        return digit, glyph, False

    def _other(self) -> tuple[NDArray, bool]:
        if self.rng.random() < 0.55:  # noqa: PLR2004
            return self.rng.choice(self.glyphs.music_others), True
        return self.rng.choice(self.glyphs.text_others), False

    def _stroke(self, canvas: NDArray, cx: float, cy: float) -> None:
        """Lines like stems, beams, slurs or bar lines."""
        rng = self.rng
        thickness = max(1, round(rng.uniform(0.1, 0.5) * UNIT))
        angle = rng.choice([0, 90, rng.uniform(0, 180)])
        length = rng.uniform(1, 6) * UNIT
        dx = np.cos(np.radians(angle)) * length / 2
        dy = np.sin(np.radians(angle)) * length / 2
        ox, oy = rng.uniform(-1, 1) * UNIT, rng.uniform(-1, 1) * UNIT
        p0 = (round(cx + ox - dx), round(cy + oy - dy))
        p1 = (round(cx + ox + dx), round(cy + oy + dy))
        if rng.random() < 0.5:  # noqa: PLR2004
            cv2.line(canvas, p0, p1, 255, thickness)
        else:
            axes = (round(length / 2), round(rng.uniform(0.3, 1.5) * UNIT))
            cv2.ellipse(
                canvas, (round(cx + ox), round(cy + oy)), axes, angle, 180, 360, 255, thickness
            )

    def _number_around(
        self, canvas: NDArray, label: int, x0: int, y0: int, x1: int, y1: int
    ) -> None:
        """Other digits of a number in a similar font left and/or right of the digit."""
        rng = self.rng
        font = rng.choice(self.glyphs.text_fonts)
        scale = (y1 - y0) / font[label].shape[0]
        height = y1 - y0
        cy = (y0 + y1) / 2
        positions = rng.choice([[-1], [1], [-1, 1], [1, 2], [-2, -1]])
        for position in positions:
            digit = font[rng.randrange(10)]
            glyph = _scaled(digit, digit.shape[0] * scale)
            gap = rng.uniform(-0.03, 0.2) * height
            if position < 0:
                cx = x0 - gap - glyph.shape[1] / 2 - (abs(position) - 1) * (glyph.shape[1] + gap)
            else:
                cx = x1 + gap + glyph.shape[1] / 2 + (position - 1) * (glyph.shape[1] + gap)
            _paste(canvas, glyph, cx, cy + rng.uniform(-0.03, 0.03) * height)

    def sample(self) -> tuple[NDArray, int]:
        rng = self.rng
        size = round(crop_units * UNIT * 1.6)
        canvas = np.zeros((size, size), np.uint8)
        cx = cy = size / 2
        in_staff = rng.random() < 0.4  # noqa: PLR2004
        kind = rng.random()
        label = OTHER
        if kind < 0.5:  # noqa: PLR2004
            label, glyph, natural = self._digit(prefer_music=in_staff)
            if in_staff:
                height = 2 * UNIT * rng.uniform(0.85, 1.15)
            else:
                height = UNIT * rng.uniform(1.1, 3.2)
            if not natural or rng.random() < 0.7:  # noqa: PLR2004
                glyph = _scaled(glyph, height)
        else:
            glyph, natural = self._other()
            if not natural:
                glyph = _scaled(glyph, UNIT * rng.uniform(0.8, 3.2))
            elif rng.random() < 0.3:  # noqa: PLR2004
                glyph = _scaled(glyph, glyph.shape[0] * rng.uniform(0.7, 1.4))
        # The component center is the crop center, apart from small errors
        jitter = 0.08 * max(glyph.shape)
        gx = cx + rng.uniform(-jitter, jitter)
        gy = cy + rng.uniform(-jitter, jitter)
        x0, y0, x1, y1 = _paste(canvas, glyph, gx, gy)
        if label != OTHER and rng.random() < 0.08:  # noqa: PLR2004
            # Off center, e.g. a fragment of a digit
            canvas[:] = 0
            shift = rng.choice([-1, 1]) * rng.uniform(0.45, 0.8) * (x1 - x0)
            _paste(canvas, glyph, gx + shift, gy + rng.uniform(-0.4, 0.4) * (y1 - y0))
            label = OTHER

        # Neighbors
        if label != OTHER and not in_staff and rng.random() < 0.5:  # noqa: PLR2004
            self._number_around(canvas, label, x0, y0, x1, y1)
        if in_staff:
            # The other digit of a time signature
            if rng.random() < 0.7:  # noqa: PLR2004
                _, other, _ = self._digit(prefer_music=True)
                other = _scaled(other, y1 - y0)
                direction = rng.choice([-1, 1])
                _paste(canvas, other, gx, gy + direction * (y1 - y0 + rng.uniform(0, 0.15) * UNIT))
            top = y0 + rng.uniform(-0.1, 0.1) * UNIT
            if rng.random() < 0.5:  # noqa: PLR2004
                top -= 2 * UNIT
            line_rows = [top + i * UNIT for i in range(5)]
        else:
            gap = rng.uniform(0.2, 2.5) * UNIT
            line_rows = (
                [y1 + gap + i * UNIT for i in range(5)] if rng.random() < 0.6 else []
            )  # noqa: PLR2004
            if rng.random() < 0.3 and line_rows:  # noqa: PLR2004
                # H-bar of a multi-measure rest
                bar_y = line_rows[2]
                cv2.rectangle(
                    canvas,
                    (0, round(bar_y - 0.4 * UNIT)),
                    (size, round(bar_y + 0.4 * UNIT)),
                    255,
                    -1,
                )
        for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
            if rng.random() < 0.5:  # noqa: PLR2004
                neighbor = (
                    self._digit(prefer_music=False)[1] if rng.random() < 0.5 else self._other()[0]
                )  # noqa: PLR2004
                neighbor = _scaled(neighbor, (y1 - y0) * rng.uniform(0.7, 1.2))
                direction = rng.choice([-1, 1])
                dist = (x1 - x0) / 2 + neighbor.shape[1] / 2 + rng.uniform(0.05, 1.2) * UNIT
                _paste(canvas, neighbor, gx + direction * dist, gy + rng.uniform(-0.3, 0.3) * UNIT)
            else:
                self._stroke(canvas, cx + rng.uniform(-2, 2) * UNIT, cy + rng.uniform(-2, 2) * UNIT)
        line_thickness = max(1, round(rng.uniform(0.08, 0.2) * UNIT))
        for row in line_rows:
            cv2.line(canvas, (0, round(row)), (size, round(row)), 255, line_thickness)

        return self._degrade(canvas), label

    def _degrade(self, canvas: NDArray) -> NDArray:
        rng = self.rng
        size = canvas.shape[0]
        if rng.random() < 0.5:  # noqa: PLR2004
            k = rng.choice([2, 3, 4, 5, 6])
            kernel = np.ones((k, k), np.uint8)
            canvas = (
                cv2.dilate(canvas, kernel) if rng.random() < 0.5 else cv2.erode(canvas, kernel)
            )  # noqa: PLR2004
        # Rotation, shear and the scale to the crop
        unit_px = crop_size / crop_units * rng.uniform(0.92, 1.08)
        scale = unit_px / UNIT
        angle = rng.uniform(-4, 4)
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
        matrix[0, 1] += rng.uniform(-0.12, 0.12) * scale
        matrix[:, 2] = (
            np.array([crop_size / 2, crop_size / 2])
            - np.array([size / 2, size / 2]) @ matrix[:, :2].T
        )
        image = cv2.warpAffine(canvas, matrix, (crop_size, crop_size), flags=cv2.INTER_AREA)
        image = 255 - image.astype(np.float32)
        # Contrast, blur, noise
        paper = rng.uniform(170, 255)
        ink = rng.uniform(0, 90)
        image = ink + (paper - ink) * image / 255
        if rng.random() < 0.5:  # noqa: PLR2004
            image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.3, 1.0))
        image += np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, rng.uniform(0, 12), image.shape
        )
        if rng.random() < 0.3:  # noqa: PLR2004
            # Like the staff images: binarized
            image = np.where(image < (paper + ink) / 2, ink, paper)
        return np.clip(image, 0, 255).astype(np.uint8)


def generate(glyphs: Glyphs, count: int, seed: int) -> tuple[NDArray, NDArray]:
    generator = Generator(glyphs, random.Random(seed))
    images = np.zeros((count, crop_size, crop_size), np.uint8)
    labels = np.zeros(count, np.int64)
    for i in range(count):
        images[i], labels[i] = generator.sample()
    return images, labels


class DigitNet(nn.Module):
    def __init__(self) -> None:
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

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 96))
        flat = 96 * (crop_size // 8) ** 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Linear(128, len(CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def train(samples: int, epochs: int, output: str) -> None:
    torch.manual_seed(0)
    glyphs = Glyphs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_x, val_y = generate(glyphs, 5000, seed=1)
    model = DigitNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, total_steps=epochs * (samples // 256 + 1)
    )
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # New synthetic samples every epoch
        train_x, train_y = generate(glyphs, samples, seed=100 + epoch)
        model.train()
        order = np.random.default_rng(epoch).permutation(samples)
        total = 0.0
        for start in range(0, samples, 256):
            index = order[start : start + 256]
            x = torch.from_numpy(train_x[index]).float().div(255).unsqueeze(1).to(device)
            y = torch.from_numpy(train_y[index]).to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(index)
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(val_x).float().div(255).unsqueeze(1).to(device)
            predicted = model(x).argmax(1).cpu().numpy()
        accuracy = (predicted == val_y).mean()
        print(f"epoch {epoch} loss {total / samples:.4f} val accuracy {accuracy:.4f}", flush=True)

    model.eval().cpu()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(1, 1, crop_size, crop_size),),
        output,
        input_names=["crops"],
        output_names=["logits"],
        dynamic_axes={"crops": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print("Wrote", output)


def write_examples(path: str, count: int = 400) -> None:
    images, labels = generate(Glyphs(), count, seed=7)
    columns = 20
    rows = []
    for r in range(count // columns):
        row = []
        for image, label in zip(
            images[r * columns : (r + 1) * columns],
            labels[r * columns : (r + 1) * columns],
            strict=True,
        ):
            tile = cv2.copyMakeBorder(image, 12, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
            cv2.putText(tile, CLASSES[label][:1], (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0, 1)
            row.append(tile)
        rows.append(np.hstack(row))
    cv2.imwrite(path, np.vstack(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", default=model_path)
    parser.add_argument("--examples", help="Only write example crops to this image")
    args = parser.parse_args()
    if args.examples:
        write_examples(args.examples)
    else:
        train(args.samples, args.epochs, args.output)
