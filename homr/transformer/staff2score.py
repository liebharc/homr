import os
from typing import Any

import albumentations as alb  # type: ignore
import cv2
import safetensors
import torch
from albumentations.pytorch import ToTensorV2  # type: ignore
from transformers import PreTrainedTokenizerFast  # type: ignore

from homr.types import NDArray

from .configs import Config
from .tromr_arch import TrOMR


class Staff2Score:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrOMR(config)
        checkpoint_file_path = config.filepaths.checkpoint
        if not os.path.exists(checkpoint_file_path):
            raise RuntimeError("Please download the model first to " + checkpoint_file_path)
        if ".safetensors" in checkpoint_file_path:
            tensors = {}
            with safetensors.safe_open(checkpoint_file_path, framework="pt", device=0) as f:  # type: ignore
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            self.model.load_state_dict(tensors, strict=False)
        else:
            self.model.load_state_dict(torch.load(checkpoint_file_path), strict=False)
        self.model.to(self.device)

        if not os.path.exists(config.filepaths.rhythmtokenizer):
            raise RuntimeError("Failed to find tokenizer config" + config.filepaths.rhythmtokenizer)
        self.lifttokenizer = PreTrainedTokenizerFast(tokenizer_file=config.filepaths.lifttokenizer)
        self.pitchtokenizer = PreTrainedTokenizerFast(
            tokenizer_file=config.filepaths.pitchtokenizer
        )
        self.rhythmtokenizer = PreTrainedTokenizerFast(
            tokenizer_file=config.filepaths.rhythmtokenizer
        )

    def detokenize(self, tokens: Any, tokenizer: Any) -> list[list[str]]:
        toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ""
                toks[b][i] = toks[b][i].replace("Ġ", " ").strip()
                if toks[b][i] in (["[BOS]", "[EOS]", "[PAD]"]):
                    del toks[b][i]
        return toks

    def predict_token(self, imgpath: str) -> tuple[NDArray, NDArray, NDArray]:
        imgs: list[NDArray] = []
        if os.path.isdir(imgpath):
            for item in os.listdir(imgpath):
                imgs.append(readimg(self.config, os.path.join(imgpath, item)))
        else:
            imgs.append(readimg(self.config, imgpath))
        imgs_tensor = torch.cat(imgs).float().unsqueeze(1)  # type: ignore
        output = self.model.generate(
            imgs_tensor.to(self.device), temperature=self.config.temperature
        )
        rhythm, pitch, lift = output
        return rhythm, pitch, lift

    def predict(self, imgpath: str) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
        rhythm, pitch, lift = self.predict_token(imgpath)

        predlift = self.detokenize(lift, self.lifttokenizer)
        predpitch = self.detokenize(pitch, self.pitchtokenizer)
        predrhythm = self.detokenize(rhythm, self.rhythmtokenizer)
        return predrhythm, predpitch, predlift


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
