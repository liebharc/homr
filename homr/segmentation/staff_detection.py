import cv2
import numpy as np
import onnxruntime as ort  # type: ignore
import torch
import torchvision.transforms as T  # type: ignore
from PIL import Image

from homr.segmentation import config
from homr.type_definitions import NDArray

loaded_model: ort.InferenceSession | None = None  # type: ignore


def load_model(onnx_model_path: str) -> ort.InferenceSession:  # type: ignore
    """Load the ONNX model."""
    global loaded_model  # noqa: PLW0603
    if loaded_model is not None:
        return loaded_model
    ort_session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])
    loaded_model = ort_session
    return ort_session


def preprocess_image(
    image: NDArray, patch_size: int = 512
) -> tuple[NDArray, NDArray, list[torch.Tensor]]:
    """Preprocess the image: resize it using calc_target_image_size,
    normalize, and split into patches."""
    pil_image = Image.fromarray(image).convert("RGB")
    pil_image = pil_image.resize((1024, 1024), Image.NEAREST)

    transform = T.Compose(
        [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    image_tensor = transform(pil_image)

    patches = split_image_into_patches(image_tensor, patch_size)
    return image, image_tensor, patches


def split_image_into_patches(
    image_tensor: torch.Tensor, patch_size: int = 512
) -> list[torch.Tensor]:
    """Split the image tensor into patches."""
    _, h, w = image_tensor.shape
    patches = []

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image_tensor[:, i : i + patch_size, j : j + patch_size]

            if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                raise ValueError("Image has an invalid size " + str(image_tensor.shape))

            patches.append(patch)

    return patches


def infer_patches_batch(  # type: ignore
    ort_session: ort.InferenceSession, patches: list[torch.Tensor]
) -> torch.Tensor:
    """Run inference on a batch of patches."""
    batch_input = torch.stack(patches).numpy()  # Stack patches into a batch
    inputs = {ort_session.get_inputs()[0].name: batch_input}
    outputs = ort_session.run(None, inputs)[0]  # Run inference on batch
    return torch.tensor(outputs)  # Convert back to tensor


def run_inference(  # type: ignore
    image: NDArray, ort_session: ort.InferenceSession, patch_size: int = 512
) -> torch.Tensor:
    """Run inference on an image with batched patches."""
    image, image_tensor, patches = preprocess_image(image, patch_size)

    # Run batched inference instead of multiple calls
    results = infer_patches_batch(ort_session, patches)

    # Split batched results back into individual patches
    split = [results[i] for i in range(results.shape[0])]

    result_image = reconstruct_from_patches(split, (image_tensor.shape[1], image_tensor.shape[2]))
    return result_image


def reconstruct_from_patches(
    patches: list[torch.Tensor], original_size: tuple[int, int]
) -> torch.Tensor:
    """Reconstruct the full image from patches."""
    h, w = original_size
    patch_size = patches[0].shape[-1]

    result = torch.zeros((4, h, w))
    index = 0

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            result[:, i : i + patch_size, j : j + patch_size] = patches[index]
            index += 1

    return result


def postprocess_output(output: torch.Tensor) -> NDArray:
    """Postprocess the model's output and resize back to original size."""
    shape_with_batch = 4  # (batch_size, num_classes, height, width)
    if len(output.shape) == shape_with_batch:
        output = output.squeeze(0)  # Remove the batch dimension

    output = np.argmax(output, axis=0)
    return np.array(output).astype(np.uint8)


def inference(image: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Perform inference on the input image and save the segmented output."""
    ort_session = load_model(config.staffs_path)
    output = run_inference(image, ort_session)

    postprocessed = postprocess_output(output)
    staff_area_class = 1
    staff_line_class = 3
    staffs = np.where(
        (postprocessed == staff_area_class) | (postprocessed == staff_line_class), 1, 0
    ).astype(np.uint8)
    brackets_class = 2
    brackets = np.where(postprocessed == brackets_class, 1, 0).astype(np.uint8)
    staff_lines = np.where(postprocessed == staff_line_class, 1, 0).astype(np.uint8)
    staffs_resized = cv2.resize(staffs, [image.shape[1], image.shape[0]])
    brackets_resized = cv2.resize(brackets, [image.shape[1], image.shape[0]])
    staff_lines_resized = cv2.resize(staff_lines, [image.shape[1], image.shape[0]])
    return (staffs_resized, brackets_resized, staff_lines_resized)
