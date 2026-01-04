import os
import shutil
import sys

import torch
import torch._dynamo
from transformers import (
    Trainer,
    TrainingArguments,
)

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from training.architecture.transformer.tromr_arch import TrOMR, load_model
from training.datasets.convert_grandstaff import (
    convert_grandstaff,
    grandstaff_train_index,
)
from training.datasets.convert_lieder import convert_lieder, lieder_train_index
from training.datasets.convert_primus import convert_primus_dataset, primus_train_index
from training.run_id import get_run_id
from training.transformer.data_loader import label_names, load_dataset
from training.transformer.mix_datasets import mix_training_sets

torch._dynamo.config.suppress_errors = True


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
    index_paths: list[str], weights: list[float], number_of_files: int
) -> list[str]:
    if len(index_paths) != len(weights):
        eprint("Error: Number of index paths and weights do not match")
        sys.exit(1)
    data_sources = [load_training_index(index) for index in index_paths]
    if not all(check_data_source(data) for data in data_sources):
        eprint("Error in datasets found")
        sys.exit(1)
    eprint(
        "Total number of training files to choose from", sum([len(data) for data in data_sources])
    )
    return mix_training_sets(data_sources, weights, number_of_files)


script_location = os.path.dirname(os.path.realpath(__file__))

git_root = os.path.join(script_location, "..", "..")


def _check_datasets_are_present(selected_datasets: list[str]) -> list[str]:
    for dataset in selected_datasets:
        if dataset == primus_train_index and not os.path.exists(primus_train_index):
            convert_primus_dataset()

        if dataset == grandstaff_train_index and not os.path.exists(grandstaff_train_index):
            convert_grandstaff()

        if dataset == lieder_train_index and not os.path.exists(lieder_train_index):
            convert_lieder()
    return selected_datasets


def train_transformer(
    fp32: bool = False, resume: str = "", smoke_test: bool = False, fine_tune: bool = False
) -> None:
    number_of_epochs = 70
    if smoke_test:
        number_of_epochs = 15
    elif fine_tune:
        number_of_epochs = 15
    resume_from_checkpoint = None

    checkpoint_folder = "current_training"
    if resume:
        resume_from_checkpoint = os.path.join(git_root, checkpoint_folder, resume)
    elif os.path.exists(os.path.join(git_root, checkpoint_folder)):
        shutil.rmtree(os.path.join(git_root, checkpoint_folder))

    if smoke_test:
        number_of_files = -1
        train_index = load_and_mix_training_sets(
            _check_datasets_are_present([lieder_train_index]),
            [1.0],
            number_of_files,
        )
    else:
        number_of_files = -1
        train_index = load_and_mix_training_sets(
            _check_datasets_are_present(
                [lieder_train_index, grandstaff_train_index, primus_train_index]
            ),
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

    batch_size = 6 if fp32 else 16

    train_args = TrainingArguments(
        checkpoint_folder,
        torch_compile=compile_model,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        learning_rate=1e-5 if fine_tune else 1e-4,
        optim="adamw_torch_fused",
        gradient_accumulation_steps=4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        num_train_epochs=number_of_epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        logging_dir=os.path.join("logs", f"run{run_id}"),
        save_strategy="epoch",
        label_names=label_names,
        bf16=not fp32,
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
        return

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
    torch.save(model.state_dict(), model_destination)
    eprint(f"Saved model to {model_destination}")


if __name__ == "__main__":
    if "--fine" in sys.argv:
        train_transformer(fp32=False, fine_tune=True)
    else:
        train_transformer(smoke_test=True)
