import os
from typing import Any

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.vocabulary import Vocabulary
from training.architecture.transformer.staff2score import readimg
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
    ) -> None:
        self.corpus_list = self._add_mask_steps(corpus_list)
        self.vocab = Vocabulary()
        self.config = config

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
        sample_filepath = entry["image"]
        sample_img = readimg(self.config, os.path.join(git_root, sample_filepath))

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
        ),
        "train_list": training_list,
        "validation": DataLoader(
            validation_list,
            config,
        ),
        "validation_list": validation_list,
    }
