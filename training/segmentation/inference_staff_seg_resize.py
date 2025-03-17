import sys

import numpy as np
import onnxruntime as ort
import torchvision.transforms as T
from PIL import Image

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


def preprocess_image(image_path: str, size=(512, 512)):
    """Preprocess the image (resize and normalize)."""
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize(size),  # Resize the image to match the model input size
            T.ToTensor(),  # Convert the image to tensor and normalize to [0, 1]
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Same normalization used during training
        ]
    )
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, image_tensor


def postprocess_output(output, original_size):
    """Postprocess the model's output and resize back to original size."""
    shape_with_batch = 4  # (batch_size, num_classes, height, width)
    if len(output.shape) == shape_with_batch:
        output = output.squeeze(0)  # Remove the batch dimension

    output = np.argmax(output, axis=0)

    output = output.astype(np.uint8)

    output_image = Image.fromarray(output)
    output_resized = output_image.resize(original_size, Image.NEAREST)
    return np.array(output_resized)


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

    original_image, image_tensor = preprocess_image(image_path)
    original_size = original_image.size

    # Run inference
    inputs = {ort_session.get_inputs()[0].name: image_tensor.numpy()}
    output = ort_session.run(None, inputs)[0]

    # Postprocess the output
    mask = postprocess_output(output, original_size)
    colored_mask = apply_color_map(mask)

    # Save the final output
    save_output_image(colored_mask, output_path)
    eprint(f"Output saved to {output_path}")


if __name__ == "__main__":
    image_path = sys.argv[1]
    onnx_model_path = "homr/segmentation/fastai_121-2062225c9553e375cc4b3e9839f5c6549a7b3e0e.onnx"
    output_path = "segmentation_output_rs.png"

    # Perform inference
    inference(image_path, onnx_model_path, output_path)
