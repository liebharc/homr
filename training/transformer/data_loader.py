import os
import random
from typing import Any

import numpy as np

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.vocabulary import Vocabulary
from training.transformer.image_utils import (
    distort_image,
    ndarray_to_tensor,
    prepare_for_tensor,
    read_image_to_ndarray,
    rotate_and_unsqueeze,
)
from training.transformer.training_vocabulary import (
    DecoderBranches,
    read_tokens,
    to_decoder_branches,
)

script_location = os.path.dirname(os.path.realpath(__file__))

git_root = os.path.join(script_location, "..", "..")

label_names = ["rhythms", "positions", "lifts", "pitchs", "articulations"]


class DataLoader:
    def __init__(
        self,
        corpus_list: list[str],
        config: Config,
        is_validation: bool = False,
    ) -> None:
        self.corpus_list = self._add_mask_steps(corpus_list)
        self.vocab = Vocabulary()
        self.config = config
        self.is_validation = is_validation

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
        sample_filepath = os.path.join(git_root, entry["image"])

        img = read_image_to_ndarray(sample_filepath)

        if self.is_validation:
            state = random.getstate()
            np_state = np.random.get_state()
            random.seed(idx)
            np.random.seed(idx)

        img = distort_image(img, allow_occlusions=not self.is_validation)

        if self.is_validation:
            random.setstate(state)
            np.random.set_state(np_state)

        img = prepare_for_tensor(img)
        sample_img = ndarray_to_tensor(img)
        sample_img = rotate_and_unsqueeze(sample_img)

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
            "mask": tokens.mask,
        }
        return result


def load_dataset(samples: list[str], config: Config, val_split: float = 0.0) -> dict[str, Any]:
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
        ),
        "validation_list": validation_list,
    }
