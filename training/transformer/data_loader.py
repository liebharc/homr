import os
import random
from typing import Any

import cv2
import numpy as np
import torch

from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.transformer.configs import Config, default_config
from homr.transformer.vocabulary import Vocabulary
from homr.type_definitions import NDArray
from training.transformer.image_utils import (
    distort_image,
    ndarray_to_tensor,
    pad_to_3_dims,
    prepare_for_tensor,
    random_staff_crop,
    read_crop_info,
    read_image_to_ndarray,
)
from training.transformer.training_vocabulary import (
    DecoderBranches,
    max_ledger_lines,
    read_tokens,
    to_decoder_branches,
)

script_location = os.path.dirname(os.path.realpath(__file__))

git_root = os.path.join(script_location, "..", "..")

label_names = ["rhythms", "positions", "lifts", "pitchs", "articulations", "slurs"]

MAX_ALLOWED_LEDGER_LINES = 5


class DataLoader:
    def __init__(
        self,
        corpus_list: list[str],
        config: Config,
        is_validation: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        """
        cache_dir: stores the prepared validation images, so that every epoch
        validates on the same images. Ignored for training data.
        """
        self.corpus_list = self._add_mask_steps(corpus_list)
        self.vocab = Vocabulary()
        self.config = config
        self.is_validation = is_validation
        self.cache_dir = cache_dir if is_validation else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _add_mask_steps(self, corpus_list: list[str]) -> Any:
        result = []
        for entry in corpus_list:
            image, token_file = entry.strip().split(",")
            result.append({"image": image, "tokens": token_file})
        return result

    def __len__(self) -> int:
        return len(self.corpus_list)

    def _read_tokens(self, path: str) -> DecoderBranches:
        tokens = read_tokens(path)
        return to_decoder_branches(tokens)

    def __getitem__(self, idx: int) -> Any:
        entry = self.corpus_list[idx]
        cache_path = os.path.join(self.cache_dir, f"{idx}.png") if self.cache_dir else None
        img = None
        if cache_path and os.path.exists(cache_path):
            img = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = self._prepare_image(idx)
            if img is None:
                return self.__getitem__((idx + 1) % len(self))
            if cache_path:
                tmp_path = f"{cache_path}.{os.getpid()}.png"
                cv2.imwrite(tmp_path, img)
                os.replace(tmp_path, cache_path)

        sample_img = ndarray_to_tensor(img)
        sample_img = pad_to_3_dims(sample_img)

        tokens_full_filepath = entry["tokens"]
        tokens = self._read_tokens(tokens_full_filepath)

        # Remember to extend label_names if you add something to the results
        result = {
            "inputs": sample_img,
            "rhythms": tokens.rhythms,
            "pitchs": tokens.pitchs,
            "lifts": tokens.lifts,
            "positions": tokens.positions,
            "articulations": tokens.articulations,
            "slurs": tokens.slurs,
            "mask": tokens.mask,
        }
        return result

    def _prepare_image(self, idx: int) -> NDArray | None:
        entry = self.corpus_list[idx]
        sample_filepath = os.path.join(git_root, entry["image"])

        try:
            img = read_image_to_ndarray(sample_filepath)
        except ValueError:
            eprint(f"Warning: skipping corrupted image {sample_filepath}")
            return None

        if self.is_validation:
            state = random.getstate()
            np_state = np.random.get_state()
            random.seed(idx)
            np.random.seed(idx)

        crop_info = read_crop_info(sample_filepath)
        if crop_info is not None:
            img = random_staff_crop(img, crop_info)
        img = distort_image(img, allow_occlusions=not self.is_validation)
        img = add_image_into_tr_omr_canvas(img)

        if self.is_validation:
            random.setstate(state)
            np.random.set_state(np_state)

        return prepare_for_tensor(img)


def _filter_valid_samples(samples: list[str]) -> list[str]:
    # Filter out samples whose notes require more than the allowed number of ledger lines.
    # MuseScore in these cases might decide to change the clef and then we get a mismatch
    # between what is visible on the image and the ground truth
    valid_samples: list[str] = []
    filter_count = 0
    too_long_count = 0
    for entry in samples:
        image, token_file = entry.strip().split(",")
        tokens = read_tokens(token_file)
        if len(tokens) > default_config.max_seq_len - 2:
            too_long_count += 1
            continue
        if max_ledger_lines(tokens) > MAX_ALLOWED_LEDGER_LINES:
            filter_count += 1
            continue
        valid_samples.append(entry)

    if filter_count > 0:
        eprint(f"Filtered out {filter_count} samples with too many ledger lines")
    if too_long_count > 0:
        eprint(f"Filtered out {too_long_count} samples with sequences longer than max_seq_len")
    return valid_samples


def load_dataset(
    samples: list[str],
    config: Config,
    val_split: float = 0.0,
    validation_cache_dir: str | None = None,
) -> dict[str, Any]:
    samples = _filter_valid_samples(samples)
    val_idx = int(len(samples) * val_split)
    training_list = samples[val_idx:]
    validation_list = samples[:val_idx]

    eprint(
        "Training with "
        + str(len(training_list))
        + " and validating with "
        + str(len(validation_list))
    )
    return {
        "train": DataLoader(
            training_list,
            config,
            is_validation=False,
        ),
        "train_list": training_list,
        "validation": DataLoader(
            validation_list,
            config,
            is_validation=True,
            cache_dir=validation_cache_dir,
        ),
        "validation_list": validation_list,
    }


SEQ_LEN_MULTIPLE = 64


def collate_trimmed(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Stacks samples and cuts the token padding to the longest sequence in the batch,
    rounded up to SEQ_LEN_MULTIPLE to limit torch.compile recompilations.
    The decoder is causal and the losses are masked, so the result doesn't change.
    """
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}
    longest = int(batch["mask"].sum(dim=1).max())
    seq_len = min(-(-longest // SEQ_LEN_MULTIPLE) * SEQ_LEN_MULTIPLE, batch["mask"].shape[1])
    for key in ["mask", *label_names]:
        batch[key] = batch[key][:, :seq_len]
    return batch
