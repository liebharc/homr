import os

import cv2
import numpy as np
import safetensors
import torch

from homr.debug import AttentionDebug
from homr.transformer.configs import Config
from homr.transformer.tromr_arch import TrOMR
from homr.type_definitions import NDArray


class Staff2Score:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrOMR(config)
        self.model.eval_mode()
        checkpoint_file_path = config.filepaths.checkpoint
        if not os.path.exists(checkpoint_file_path):
            raise RuntimeError("Please download the model first to " + checkpoint_file_path)
        if ".safetensors" in checkpoint_file_path:
            tensors = {}
            with safetensors.safe_open(checkpoint_file_path, framework="pt", device=0) as f:  # type: ignore
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            self.model.load_state_dict(tensors, strict=False)
        elif torch.cuda.is_available():
            self.model.load_state_dict(
                torch.load(checkpoint_file_path, weights_only=True), strict=False
            )
        else:
            self.model.load_state_dict(
                torch.load(
                    checkpoint_file_path, weights_only=True, map_location=torch.device("cpu")
                ),
                strict=False,
            )
        self.model.to(self.device)

        if not os.path.exists(config.filepaths.rhythmtokenizer):
            raise RuntimeError("Failed to find tokenizer config" + config.filepaths.rhythmtokenizer)

    def predict(self, image: NDArray, debug: AttentionDebug | None = None) -> list[str]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgs_tensor = self._image_to_tensor(image)
        return self._generate(
            imgs_tensor,
            debug=debug,
        )

    def _image_to_tensor(self, image: NDArray) -> torch.Tensor:
        transformed = _transform(image=image)
        imgs_tensor = transformed.float().unsqueeze(1)
        return imgs_tensor.to(self.device)

    def _generate(
        self,
        imgs_tensor: torch.Tensor,
        debug: AttentionDebug | None = None,
    ) -> list[str]:
        return self.model.generate(
            imgs_tensor,
            debug=debug,
        )


class ConvertToTensor:
    def __init__(self) -> None:
        self.mean = torch.tensor([0.7931]).view(1, 1, 1)
        self.std = torch.tensor([0.1738]).view(1, 1, 1)

    def to_tensor(self, img: NDArray) -> torch.Tensor:
        img_array = img.astype(np.float32) / 255.0
        return torch.tensor(img_array)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def __call__(self, image: NDArray) -> torch.Tensor:
        tensor = self.to_tensor(image)
        tensor = self.normalize(tensor)
        return tensor


_transform = ConvertToTensor()


def readimg(config: Config, path: str) -> torch.Tensor:
    img: NDArray = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to read image from " + path)

    if img.shape[-1] == 4:  # noqa: PLR2004
        img = 255 - img[:, :, 3]
    elif img.shape[-1] == 3:  # noqa: PLR2004
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:  # noqa: PLR2004
        # Image is already gray scale
        pass
    else:
        raise RuntimeError("Unsupport image type!")

    h, w = img.shape
    size_h = config.max_height
    new_h = size_h
    new_w = int(size_h / h * w)
    new_w = new_w // config.patch_size * config.patch_size
    img = cv2.resize(img, (new_w, new_h))
    tensor = _transform(image=img)
    return tensor
