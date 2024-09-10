import os
import shutil
import sys

import safetensors
import torch
import torch._dynamo
from transformers import Trainer, TrainingArguments  # type: ignore

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.tromr_arch import TrOMR
from training.convert_grandstaff import convert_grandstaff, grandstaff_train_index
from training.convert_lieder import convert_lieder, lieder_train_index
from training.convert_primus import (
    convert_primus_dataset,
    primus_distorted_train_index,
    primus_train_index,
)
from training.run_id import get_run_id
from training.transformer.data_loader import load_dataset
from training.transformer.mix_datasets import mix_training_sets

torch._dynamo.config.suppress_errors = True


def load_training_index(file_path: str) -> list[str]:
    with open(file_path) as f:
        return f.readlines()


def contains_supported_clef(semantic: str) -> bool:
    if semantic.count("clef-") != 1:
        return False
    return "clef-G2" in semantic or "clef-F4" in semantic


def filter_for_clefs(file_paths: list[str]) -> list[str]:
    result = []
    for entry in file_paths:
        semantic = entry.strip().split(",")[1]
        if semantic == "nosymbols":
            continue
        with open(semantic) as f:
            lines = f.readlines()
            if all(contains_supported_clef(line) for line in lines):
                result.append(entry)
    return result


def check_data_source(all_file_paths: list[str]) -> bool:
    result = True
    for file_paths in all_file_paths:
        paths = file_paths.strip().split(",")
        for path in paths:
            if path == "nosymbols":
                continue
            if not os.path.exists(path):
                eprint(f"Index {file_paths} does not exist due to {path}")
                result = False
    return result


def load_and_mix_training_sets(
    index_paths: list[str], weights: list[float], number_of_files: int
) -> list[str]:
    if len(index_paths) != len(weights):
        eprint("Error: Number of index paths and weights do not match")
        sys.exit(1)
    data_sources = [load_training_index(index) for index in index_paths]
    if not all(check_data_source(data) for data in data_sources):
        eprint("Error in datasets found")
        sys.exit(1)
    data_sources = [filter_for_clefs(data) for data in data_sources]
    eprint(
        "Total number of training files to choose from", sum([len(data) for data in data_sources])
    )
    return mix_training_sets(data_sources, weights, number_of_files)


script_location = os.path.dirname(os.path.realpath(__file__))

vocabulary = os.path.join(script_location, "vocabulary_semantic.txt")
git_root = os.path.join(script_location, "..", "..")


def _check_datasets_are_present() -> None:
    if not os.path.exists(primus_train_index) or not os.path.exists(primus_distorted_train_index):
        convert_primus_dataset()

    if not os.path.exists(grandstaff_train_index):
        convert_grandstaff()

    if not os.path.exists(lieder_train_index):
        convert_lieder()


def train_transformer(fp32: bool = False, pretrained: bool = False, resume: str = "") -> None:
    number_of_files = -1
    number_of_epochs = 30
    resume_from_checkpoint = None

    checkpoint_folder = "current_training"
    if resume:
        resume_from_checkpoint = os.path.join(git_root, checkpoint_folder, resume)
    elif os.path.exists(os.path.join(git_root, checkpoint_folder)):
        shutil.rmtree(os.path.join(git_root, checkpoint_folder))

    _check_datasets_are_present()

    train_index = load_and_mix_training_sets(
        [primus_train_index, grandstaff_train_index, lieder_train_index],
        [1.0, 1.0, 1.0],
        number_of_files,
    )

    config = Config()
    datasets = load_dataset(train_index, config, val_split=0.1)

    compile_threshold = 50000
    compile_model = (
        number_of_files < 0 or number_of_files * number_of_epochs >= compile_threshold
    )  # Compiling needs time, but pays off for large datasets
    if compile_model:
        eprint("Compiling model")

    run_id = get_run_id()

    train_args = TrainingArguments(
        checkpoint_folder,
        torch_compile=compile_model,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        # TrOMR Paper page 3 specifies a rate of 1e-3, but that can cause issues with fp16 mode
        learning_rate=1e-4,
        optim="adamw_torch",  # TrOMR Paper page 3 species an Adam optimizer
        per_device_train_batch_size=16,  # TrOMR Paper page 3
        per_device_eval_batch_size=8,
        num_train_epochs=number_of_epochs,
        weight_decay=0.01,
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        logging_dir=os.path.join("logs", f"run{run_id}"),
        save_strategy="epoch",
        label_names=["rhythms_seq", "note_seq", "lifts_seq", "pitchs_seq"],
        fp16=not fp32,
        dataloader_pin_memory=True,
        dataloader_num_workers=12,
    )

    if pretrained:
        eprint("Loading pretrained model")
        model = TrOMR(config)
        checkpoint_file_path = config.filepaths.checkpoint
        if ".safetensors" in checkpoint_file_path:
            tensors = {}
            with safetensors.safe_open(checkpoint_file_path, framework="pt", device=0) as f:  # type: ignore
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            model.load_state_dict(tensors, strict=False)
        else:
            model.load_state_dict(torch.load(checkpoint_file_path), strict=False)
    else:
        model = TrOMR(config)

    try:
        trainer = Trainer(
            model,
            train_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        eprint("Interrupted")

    model_destination = os.path.join(git_root, "homr", "transformer", f"pytorch_model_{run_id}.pth")
    torch.save(model.state_dict(), model_destination)
    eprint(f"Saved model to {model_destination}")
