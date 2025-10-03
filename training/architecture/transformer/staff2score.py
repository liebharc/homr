import os
from typing import Any

import cv2
import numpy as np
import safetensors
import torch
from PIL import Image

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray
from training.architecture.transformer.tromr_arch import TrOMR
from training.transformer.training_vocabulary import token_lines_to_str


def load_model_weights(checkpoint_file_path: str) -> Any:
    if ".safetensors" in checkpoint_file_path:
        tensors = {}
        with safetensors.safe_open(checkpoint_file_path, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors
    elif torch.cuda.is_available():
        return torch.load(checkpoint_file_path, weights_only=True)
    else:
        return torch.load(checkpoint_file_path, weights_only=True, map_location=torch.device("cpu"))


class Staff2Score:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrOMR(config)
        self.model.eval_mode()
        checkpoint_file_path = config.filepaths.checkpoint
        self.model.load_state_dict(load_model_weights(checkpoint_file_path), strict=False)
        self.model.to(self.device)

        if not os.path.exists(config.filepaths.rhythmtokenizer):
            raise RuntimeError("Failed to find tokenizer config" + config.filepaths.rhythmtokenizer)

    def predict(self, image: NDArray) -> list[EncodedSymbol]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgs_tensor = self._image_to_tensor(image)
        return self._generate(
            imgs_tensor,
        )

    def _image_to_tensor(self, image: NDArray) -> torch.Tensor:
        transformed = _transform(image=image)
        imgs_tensor = transformed.float().unsqueeze(1)
        return imgs_tensor.to(self.device)

    def _generate(
        self,
        imgs_tensor: torch.Tensor,
    ) -> list[EncodedSymbol]:
        return self.model.generate(
            imgs_tensor,
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
    img: NDArray = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # type: ignore
    if img is None:
        raise ValueError("Failed to read image from " + path)

    if img.shape[-1] == 4:
        img = 255 - img[:, :, 3]
    elif img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        # Image is already gray scale
        pass
    else:
        raise RuntimeError("Unsupport image type!")

    tensor = _transform(image=img)
    return tensor


if __name__ == "__main__":
    import sys

    model = Staff2Score(Config())
    image = np.array(Image.open(sys.argv[1]))
    out = model.predict(image)
    eprint(token_lines_to_str(out))
