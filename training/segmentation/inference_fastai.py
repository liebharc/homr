import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from homr.resize import calc_target_image_size
from homr.simple_logging import eprint

COLORS = {
    0: (255, 255, 255),  # background
    1: (0, 0, 0),  # staff
    2: (255, 0, 0),  # symbol
    3: (0, 255, 0),  # symbol
}


def load_model(onnx_model_path: str):
    """Load the ONNX model."""
    ort_session = ort.InferenceSession(onnx_model_path)
    return ort_session


def preprocess_image(image_path: str, patch_size=512):
    """Preprocess the image: resize it using calc_target_image_size,
    normalize, and split into patches."""
    image = Image.open(image_path).convert("RGB")
    target_size = calc_target_image_size(image)
    image = image.resize(target_size, Image.NEAREST)

    transform = T.Compose(
        [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    image_tensor = transform(image)

    patches = split_image_into_patches(image_tensor, patch_size)
    return image, image_tensor, patches


def split_image_into_patches(image_tensor: torch.Tensor, patch_size=512):
    """Split the image tensor into patches."""
    _, h, w = image_tensor.shape
    patches = []

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image_tensor[:, i : i + patch_size, j : j + patch_size]

            # Padding if the patch is smaller than required
            if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                pad_h = patch_size - patch.shape[1]
                pad_w = patch_size - patch.shape[2]
                patch = T.Pad((0, 0, pad_w, pad_h), fill=0)(patch)

            patches.append(patch)

    return patches


def infer_patch(ort_session, patch):
    """Run inference on a single patch."""
    inputs = {ort_session.get_inputs()[0].name: patch.unsqueeze(0).numpy()}
    output = ort_session.run(None, inputs)[0]
    return torch.tensor(output)


def run_inference(image_path: str, ort_session, patch_size=512):
    """Run inference on an image, process patches in parallel, and concatenate results."""
    image, image_tensor, patches = preprocess_image(image_path, patch_size)

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(lambda p: infer_patch(ort_session, p), patches),
                total=len(patches),
                desc="Running inference on patches",
            )
        )

    result_image = reconstruct_from_patches(results, image_tensor.shape[1:])
    return result_image


def reconstruct_from_patches(patches, original_size):
    """Reconstruct the full image from patches."""
    h, w = original_size
    patch_size = patches[0].shape[-1]

    result = torch.zeros((3, h, w))
    index = 0

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            result[:, i : i + patch_size, j : j + patch_size] = patches[index]
            index += 1

    return result


def postprocess_output(output):
    """Postprocess the model's output and resize back to original size."""
    shape_with_batch = 4  # (batch_size, num_classes, height, width)
    if len(output.shape) == shape_with_batch:
        output = output.squeeze(0)  # Remove the batch dimension

    output = np.argmax(output, axis=0)
    output = np.array(output).astype(np.uint8)
    output_image = Image.fromarray(output)
    return np.array(output_image)


def apply_color_map(mask):
    """Apply color map to the segmentation mask."""
    unique_values = np.unique(mask)
    unexpected_values = set(unique_values) - set(COLORS.keys())
    if unexpected_values:
        raise ValueError(f"Unexpected values found in the segmentation mask: {unexpected_values}")

    missing_values = set(COLORS.keys()) - set(unique_values)
    if missing_values:
        eprint("WARN: Didn't find these values in the result", missing_values)

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in COLORS.items():
        color_mask[mask == label] = color
    return color_mask


def save_output_image(output_mask, output_path):
    """Save the output image."""
    output_image = Image.fromarray(output_mask)
    output_image.save(output_path)


def inference(image_path: str, onnx_model_path: str, output_path: str):
    """Perform inference on the input image and save the segmented output."""
    ort_session = load_model(onnx_model_path)
    output = run_inference(image_path, ort_session)

    mask = postprocess_output(output)
    colored_mask = apply_color_map(mask)

    # Save the final output
    save_output_image(colored_mask, output_path)
    eprint(f"Output saved to {output_path}")


if __name__ == "__main__":
    image_path = sys.argv[1]
    onnx_model_path = "homr/segmentation/fastai_121-2062225c9553e375cc4b3e9839f5c6549a7b3e0eonnx"
    output_path = "segmentation_output.png"

    # Perform inference
    inference(image_path, onnx_model_path, output_path)
