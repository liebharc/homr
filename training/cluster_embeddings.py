# ruff: noqa: T201
# type: ignore

import argparse
import os
import shutil

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from homr.transformer.configs import default_config
from homr.transformer.staff2score import readimg
from homr.transformer.tromr_arch import load_model

MAX_NUMBER_OF_SAMPLES_WITH_IMAGES = 1000


def extract_embedding(model, img_tensor):
    """
    Forward pass to get the 'x' embedding from the transformer.
    img_tensor shape: (1, 1, H, W)
    """
    with torch.no_grad():
        # Get the x output directly
        x = model.encoder(img_tensor)
        # Mean-pool over sequence dimension → (1, embedding_dim)
        emb = x.mean(dim=1).squeeze(0).cpu().numpy()
        return emb


def load_and_prepare_image_for_tensorboard(img_path, max_size=200):
    """
    Load image and prepare it for TensorBoard visualization.
    Preserves aspect ratio and ensures good resolution for viewing.
    """
    try:
        # Load original image
        img = Image.open(img_path)

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Get original dimensions
        w, h = img.size

        # Only resize if image is larger than max_size
        # Use higher quality resampling and be more conservative with resizing
        if max(w, h) > max_size:
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)

            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert to tensor (C, H, W) with values in [0, 1]
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img_tensor = transform(img)

        return img_tensor
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


def create_simple_metadata(index):
    """
    Create simple, clean metadata for TensorBoard.
    """

    metadata = []
    for i, entry in enumerate(index):
        try:
            img_path = entry.split(",")[0].strip()
            filename = os.path.basename(img_path)
            metadata.append(filename)
        except Exception:
            metadata.append(f"{i:04d}_ERROR")

    return metadata


def main(index_file, logdir, max_image_size, include_images):
    if os.path.exists(logdir):
        shutil.rmtree(logdir)

    config = default_config
    model = load_model(config)
    writer = SummaryWriter(log_dir=logdir)

    with open(index_file) as f:
        index = f.readlines()

    if include_images and len(index) > MAX_NUMBER_OF_SAMPLES_WITH_IMAGES:
        index = list(np.random.choice(index, MAX_NUMBER_OF_SAMPLES_WITH_IMAGES, replace=False))
        print(
            f"⚠️ Dataset too large, sampling {MAX_NUMBER_OF_SAMPLES_WITH_IMAGES} random entries for TensorBoard."  # noqa: E501
        )

    embeddings = []
    images_for_tensorboard = []

    print(f"Processing {len(index)} images...")

    for entry in tqdm(index, desc="Processing images"):
        try:
            img_path = entry.split(",")[0].strip()

            # Extract embedding using the model's preprocessing
            img_tensor = readimg(config, img_path).unsqueeze(0)  # (1, 1, H, W)
            img_tensor = img_tensor.to(next(model.parameters()).device)
            emb = extract_embedding(model, img_tensor)

            embeddings.append(emb)

            # Prepare image for TensorBoard visualization if requested
            if include_images:
                tb_img = load_and_prepare_image_for_tensorboard(img_path, max_image_size)
                if tb_img is not None:
                    images_for_tensorboard.append(tb_img)
                else:
                    # Create a placeholder black image if loading fails
                    placeholder = torch.zeros(3, 200, 800)
                    images_for_tensorboard.append(placeholder)

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    if not embeddings:
        print("No embeddings were extracted!")
        return

    embeddings = np.stack(embeddings)

    # Create simple metadata for TensorBoard
    metadata = create_simple_metadata(index)

    # Prepare images tensor for TensorBoard if requested
    label_img = None
    if include_images and images_for_tensorboard:
        # Don't pad images to the same size - let TensorBoard handle different sizes
        # Just stack them as-is for better aspect ratio preservation
        try:
            label_img = torch.stack(images_for_tensorboard)
            print(f"Prepared {len(images_for_tensorboard)} images for TensorBoard visualization")
        except RuntimeError:
            # If stacking fails due to different sizes, pad minimally
            max_h = max(img.shape[1] for img in images_for_tensorboard)
            max_w = max(img.shape[2] for img in images_for_tensorboard)

            padded_images = []
            for img in images_for_tensorboard:
                c, h, w = img.shape
                # Create padded image with minimal padding
                padded = torch.zeros(c, max_h, max_w)
                padded[:, :h, :w] = img
                padded_images.append(padded)

            label_img = torch.stack(padded_images)
            print(f"Prepared {len(padded_images)} images for TensorBoard (with minimal padding)")

    # Save embeddings to TensorBoard
    writer.add_embedding(
        mat=embeddings, metadata=metadata, label_img=label_img, tag="image_embeddings"
    )

    writer.close()

    print(f"✅ Saved embeddings for {len(embeddings)} images to {logdir}")
    print(f"To view: tensorboard --logdir {logdir}")
    if include_images:
        print("📸 Click on points in TensorBoard projector to see images at good resolution")
    else:
        print("💡 Use --include-images to see images when clicking points in the projector")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract image embeddings for TensorBoard projector"
    )
    parser.add_argument("index", help="Index file")
    parser.add_argument("--logdir", default="./runs/embeddings", help="TensorBoard log directory")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=200,
        help="Maximum size for images in TensorBoard (default: 200)",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Include images in TensorBoard projector visualization",
    )

    args = parser.parse_args()

    main(args.index, args.logdir, args.max_image_size, args.include_images)
