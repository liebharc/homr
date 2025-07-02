import numpy as np
import torch

from homr.segmentation.model import create_segnet, create_unet  # type: ignore
from homr.type_definitions import NDArray


def split_into_patches(image: NDArray, win_size: int, step_size: int = -1) -> NDArray:
    if step_size < 0:
        step_size = win_size // 2

    data = []
    for y in range(0, image.shape[0], step_size):
        if y + win_size > image.shape[0]:
            y = image.shape[0] - win_size  # noqa: PLW2901
        for x in range(0, image.shape[1], step_size):
            if x + win_size > image.shape[1]:
                x = image.shape[1] - win_size  # noqa: PLW2901
            hop = image[y : y + win_size, x : x + win_size]
            data.append(hop)
    return np.array(data)


def merge_patches(
    patches: NDArray, image_shape: list[int], win_size: int, step_size: int = -1
) -> NDArray:
    if step_size < 0:
        step_size = win_size // 2

    reconstructed = np.zeros(image_shape, dtype=np.float32)
    weight = np.zeros(image_shape, dtype=np.float32)

    idx = 0
    for iy in range(0, image_shape[0], step_size):
        if iy + win_size > image_shape[0]:
            y = image_shape[0] - win_size
        else:
            y = iy
        for ix in range(0, image_shape[1], step_size):
            if ix + win_size > image_shape[1]:
                x = image_shape[1] - win_size
            else:
                x = ix

            reconstructed[y : y + win_size, x : x + win_size] += patches[idx]
            weight[y : y + win_size, x : x + win_size] += 1
            idx += 1

    # Avoid division by zero
    weight[weight == 0] = 1
    reconstructed /= weight

    return reconstructed.astype(patches[0].dtype)


def inference(
    model_path: str,
    image: NDArray,
) -> NDArray:
    if "segnet" in model_path:
        model = create_segnet()
    elif "unet" in model_path:
        model = create_unet()
    else:
        raise ValueError("Unknown model type: " + model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    images = split_into_patches(image, win_size=320, step_size=320)

    # Switch the model to evaluation mode
    with torch.inference_mode():
        model.eval()
        logits = model(torch.tensor(images).permute(0, 3, 1, 2))  # Get raw logits from the model

    # Apply softmax to get class probabilities
    # Shape: [batch_size, num_classes, H, W]

    pr_masks = logits.softmax(dim=1)
    # Convert class probabilities to predicted class labels
    pr_masks = pr_masks.argmax(dim=1)  # Shape: [batch_size, H, W]

    return merge_patches(
        pr_masks.cpu().detach().numpy(), image.shape[0:2], win_size=320, step_size=320
    )
