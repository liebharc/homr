from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from attacks.src.segmentation_onnx import SegNetONNX, extract_staff_crops_from_class_map


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError(f"Could not read image: {path}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def cache_staff_crops(
    images_dir: Path,
    output_dir: Path,
    model_path: Path,
    limit: int,
    use_cuda: bool,
    batch_size: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.png"))

    if limit > 0:
        image_paths = image_paths[:limit]

    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")

    segnet = SegNetONNX(
        model_path=model_path,
        use_cuda=use_cuda,
        batch_size=batch_size,
    )

    summary = []

    for image_index, image_path in enumerate(image_paths, start=1):
        print(f"[{image_index}/{len(image_paths)}] {image_path.name}")

        image_rgb = read_rgb(image_path)

        class_map = segnet.predict_class_map(image_rgb)

        crop_items = extract_staff_crops_from_class_map(
            image_rgb=image_rgb,
            class_map=class_map,
        )

        image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        class_map_path = image_output_dir / "class_map.npy"
        np.save(class_map_path, class_map)

        metadata = {
            "source_image": str(image_path),
            "num_crops": len(crop_items),
            "crops": [],
        }

        for item in crop_items:
            crop_index = item["index"]
            crop = item["crop"]

            crop_name = f"staff_{crop_index:02d}.png"
            crop_path = image_output_dir / crop_name

            save_rgb(crop_path, crop)

            metadata["crops"].append(
                {
                    "file": crop_name,
                    "y0": item["y0"],
                    "y1": item["y1"],
                    "raw_region": list(item["raw_region"]),
                    "shape": list(crop.shape),
                }
            )

        metadata_path = image_output_dir / "metadata.json"

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        summary.append(
            {
                "image": image_path.name,
                "num_crops": len(crop_items),
                "output_dir": str(image_output_dir),
            }
        )

        print(f"  saved {len(crop_items)} crop(s)")

    summary_path = output_dir / "summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Summary written to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache SegNet-derived staff crops.")

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("dataset/images"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/cached_crops"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/onnx/segnet.onnx"),
    )
    parser.add_argument(
        "-N",
        "--limit",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    cache_staff_crops(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        limit=args.limit,
        use_cuda=not args.cpu,
        batch_size=args.batch_size,
    )