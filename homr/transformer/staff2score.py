import os

import albumentations as alb  # type: ignore
import cv2
import safetensors
import torch
from albumentations.pytorch import ToTensorV2  # type: ignore

from homr.debug import AttentionDebug
from homr.transformer.configs import Config
from homr.transformer.tromr_arch import TrOMR
from homr.type_definitions import NDArray


class Staff2Score:
    def __init__(self, config: Config, keep_all_symbols_in_chord: bool = False) -> None:
        self.config = config
        self.keep_all_symbols_in_chord = keep_all_symbols_in_chord
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
            self.model.load_state_dict(torch.load(checkpoint_file_path), strict=False)
        else:
            self.model.load_state_dict(
                torch.load(checkpoint_file_path, map_location=torch.device("cpu")), strict=False
            )
        self.model.to(self.device)

        if not os.path.exists(config.filepaths.rhythmtokenizer):
            raise RuntimeError("Failed to find tokenizer config" + config.filepaths.rhythmtokenizer)

    def predict(self, image: NDArray, debug: AttentionDebug | None = None) -> list[str]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imgs_tensor = self._image_to_tensor(image)
        return self._generate(
            imgs_tensor,
            debug=debug,
        )

    def _image_to_tensor(self, image: NDArray) -> torch.Tensor:
        transformed = _transform(image=image)["image"][:1]
        imgs_tensor = transformed.float().unsqueeze(1)
        return imgs_tensor.to(self.device)  # type: ignore

    def _generate(
        self,
        imgs_tensor: torch.Tensor,
        debug: AttentionDebug | None = None,
    ) -> list[str]:
        return self.model.generate(
            imgs_tensor,
            keep_all_symbols_in_chord=self.keep_all_symbols_in_chord,
            debug=debug,
        )


_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        ToTensorV2(),
    ]
)


def readimg(config: Config, path: str) -> NDArray:
    img: NDArray = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to read image from " + path)

    if img.shape[-1] == 4:  # noqa: PLR2004
        img = 255 - img[:, :, 3]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 3:  # noqa: PLR2004
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:  # noqa: PLR2004
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise RuntimeError("Unsupport image type!")

    h, w, c = img.shape
    size_h = config.max_height
    new_h = size_h
    new_w = int(size_h / h * w)
    new_w = new_w // config.patch_size * config.patch_size
    img = cv2.resize(img, (new_w, new_h))
    img = _transform(image=img)["image"][:1]
    return img
