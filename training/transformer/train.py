import os
import shutil
import sys
from typing import Any

import torch
import torch._dynamo
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from training.architecture.transformer.tromr_arch import TrOMR, load_model
from training.omr_datasets.convert_grandstaff import (
    convert_grandstaff,
    grandstaff_train_index,
)
from training.omr_datasets.convert_lieder import convert_lieder, lieder_train_index
from training.omr_datasets.convert_pdmx import convert_pdmx, pdmx_train_index
from training.omr_datasets.filter_from_index import (
    read_invalid_tokens,
    without_invalid_tokens,
    write_invalid_tokens,
)
from training.run_id import get_run_id
from training.transformer.data_loader import collate_trimmed, label_names, load_dataset
from training.transformer.distribute import Distribute
from training.transformer.metrics import HomrTrainer
from training.transformer.mix_datasets import mix_training_sets

torch._dynamo.config.suppress_errors = True
# Batches are padded to multiples of 64 tokens. Compile each length statically: Inductor
# fails on a dynamic sequence length and then falls back to eager mode.
# Up to 10 lengths, each for training, the no-grad pass and evaluation.
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.recompile_limit = 64


class FreezeCallback(TrainerCallback):
    """
    Callback to freeze the backbone for a set number of epochs.
    Standard practice is ~2 epochs.
    """

    def __init__(self, epochs_to_freeze: int = 2):
        self.epochs_to_freeze = epochs_to_freeze
        self._backbone_frozen = False

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        model = kwargs.get("model")
        if model and hasattr(model, "freeze_backbone"):
            eprint(f"Freezing backbone for the first {self.epochs_to_freeze} epochs")
            model.freeze_backbone()
            self._backbone_frozen = True

    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        model = kwargs.get("model")
        if model and self._backbone_frozen and state.epoch and state.epoch >= self.epochs_to_freeze:
            eprint(f"Unfreezing backbone at epoch {state.epoch}")
            model.unfreeze_backbone()
            self._backbone_frozen = False


def load_training_index(file_path: str) -> list[str]:
    with open(file_path) as f:
        return f.readlines()


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
    index_paths: list[str],
    weights: list[float],
    number_of_files: int,
    invalid_tokens: set[str] | None = None,
) -> list[str]:
    if len(index_paths) != len(weights):
        eprint("Error: Number of index paths and weights do not match")
        sys.exit(1)
    data_sources = [
        without_invalid_tokens(load_training_index(index), invalid_tokens or set())
        for index in index_paths
    ]
    if not all(check_data_source(data) for data in data_sources):
        eprint("Error in datasets found")
        sys.exit(1)
    eprint(
        "Total number of training files to choose from", sum([len(data) for data in data_sources])
    )
    return mix_training_sets(data_sources, weights, number_of_files)


script_location = os.path.dirname(os.path.realpath(__file__))

git_root = os.path.join(script_location, "..", "..")


def _check_datasets_are_present(selected_datasets: list[str], invalid_tokens_file: str) -> None:
    for dataset in selected_datasets:
        if dataset == grandstaff_train_index and not os.path.exists(grandstaff_train_index):
            convert_grandstaff()

        if dataset == lieder_train_index and not os.path.exists(lieder_train_index):
            convert_lieder()

        if dataset == pdmx_train_index and not os.path.exists(pdmx_train_index):
            convert_pdmx()
    write_invalid_tokens(invalid_tokens_file)


def train_transformer(
    fp32: bool = False, resume: str = "", smoke_test: bool = False, fine_tune: bool = False
) -> None:
    distribute = Distribute()

    number_of_epochs = 35
    if smoke_test:
        number_of_epochs = 10
    elif fine_tune:
        number_of_epochs = 15
    resume_from_checkpoint = None

    checkpoint_folder = "current_training"
    if resume:
        resume_from_checkpoint = os.path.join(git_root, checkpoint_folder, resume)
    elif os.path.exists(os.path.join(git_root, checkpoint_folder)):
        if distribute.is_rank0():
            shutil.rmtree(os.path.join(git_root, checkpoint_folder))

    # PrIMuS isn't used: it has no beams and its Hugging Face copy is CC BY-NC (non-commercial)
    # MuseTrainer isn't used: 1k samples, it adds little next to PDMX and has no license file
    dataset_index = [
        lieder_train_index,
        grandstaff_train_index,
        pdmx_train_index,
    ]
    dataset_weights = [1.0, 1.0, 1.0]
    invalid_tokens_file = os.path.join(git_root, checkpoint_folder, "invalid_tokens.txt")
    if distribute.is_rank0():
        _check_datasets_are_present(dataset_index, invalid_tokens_file)
    distribute.barrier()

    number_of_files = -1
    train_index = load_and_mix_training_sets(
        dataset_index,
        dataset_weights,
        number_of_files,
        read_invalid_tokens(invalid_tokens_file),
    )

    config = Config()
    datasets = load_dataset(
        train_index,
        config,
        val_split=0.1,
        validation_cache_dir=os.path.join(git_root, checkpoint_folder, "validation_cache"),
    )

    compile_threshold = 50000
    compile_model = (
        number_of_files < 0 or number_of_files * number_of_epochs >= compile_threshold
    )  # Compiling needs time, but pays off for large datasets
    if compile_model:
        eprint("Compiling model")

    run_id = get_run_id()

    batch_size = 16  # 16gb vram
    eval_batch_size = 32

    train_args = TrainingArguments(
        checkpoint_folder,
        torch_compile=compile_model,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5 if fine_tune else 1e-4,
        optim="adamw_torch_fused",
        gradient_accumulation_steps=max(1, 2 // distribute.get_world_size()),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=number_of_epochs,
        weight_decay=0.05,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=["tensorboard"],
        logging_dir=os.path.join("logs", f"run{run_id}"),
        label_names=label_names,
        bf16=not fp32,
        logging_nan_inf_filter=False,  # the filter syncs with the GPU every step
        dataloader_pin_memory=True,
        dataloader_num_workers=12,
    )

    if fine_tune:
        eprint("Fine tuning model from", config.filepaths.checkpoint)
        model = load_model(config)
        model.freeze_encoder()
        model.freeze_decoder()
        model.unfreeze_lift_decoder()
    else:
        model = TrOMR(config)

    model_name = "pytorch_model"

    model_destination = os.path.join(
        git_root, "training", "architecture", "transformer", f"{model_name}_{run_id}.pth"
    )

    if os.path.exists(model_destination):
        eprint("Model already exists", model_destination)
        distribute.destroy()
        return

    try:
        callbacks: list[TrainerCallback] = [EarlyStoppingCallback(early_stopping_patience=5)]
        if not fine_tune:
            callbacks.append(FreezeCallback(epochs_to_freeze=2))

        trainer = HomrTrainer(
            model,
            train_args,
            train_dataset=datasets["train"],
            data_collator=collate_trimmed,
            eval_dataset=datasets["validation"],
            callbacks=callbacks,
            distribute=distribute,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        eprint("Interrupted")
    if distribute.is_rank0():
        torch.save(model.state_dict(), model_destination)
        eprint(f"Saved model to {model_destination}")
    distribute.barrier()
    distribute.destroy()


if __name__ == "__main__":
    if "--fine" in sys.argv:
        train_transformer(fp32=False, fine_tune=True)
    elif len(sys.argv) > 1:
        raise ValueError("Unknown argument")
    else:
        train_transformer(smoke_test=True)
