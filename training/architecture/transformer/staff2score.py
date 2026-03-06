import os
from typing import Any

import numpy as np
import safetensors
import torch
from PIL import Image

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray
from training.architecture.transformer.tromr_arch import TrOMR
from training.transformer.image_utils import (
    ndarray_to_tensor,
    pad_to_3_dims,
    prepare_for_tensor,
    read_image_to_ndarray,
)
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
        image = prepare_for_tensor(image)
        tensor = ndarray_to_tensor(image)
        tensor = pad_to_3_dims(tensor)
        imgs_tensor = tensor.float().unsqueeze(0).to(self.device)
        return self._generate(
            imgs_tensor,
        )

    def _generate(
        self,
        imgs_tensor: torch.Tensor,
    ) -> list[EncodedSymbol]:
        return self.model.generate(
            imgs_tensor,
        )


def readimg(config: Config, path: str) -> torch.Tensor:
    img = read_image_to_ndarray(path)
    img = prepare_for_tensor(img)
    tensor = ndarray_to_tensor(img)
    return pad_to_3_dims(tensor)


if __name__ == "__main__":
    import sys

    model = Staff2Score(Config())
    image = np.array(Image.open(sys.argv[1]))
    out = model.predict(image)
    eprint(token_lines_to_str(out))
