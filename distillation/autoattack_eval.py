#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import editdistance
import torch
import torch.nn as nn

from autoattack import AutoAttack

from distillation.evaluate_surrogate import (
    load_checkpoint_model,
    loader_args,
    stats_from_config,
)
from distillation.train_student import (
    BRANCHES,
    FullPageStudent,
    make_loader,
    move_batch_to_device,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attack clean and PGD-trained surrogates with AutoAttack and compare robustness."
    )
    parser.add_argument("--clean-checkpoint", required=True, type=Path)
    parser.add_argument("--pgd-checkpoint", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--epsilon-grid", required=True, nargs="+", type=float)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--version", default="standard", choices=("standard", "plus", "rand", "custom"))
    parser.add_argument("--attacks", nargs="+", default=["apgd-ce", "square"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--repo-root", default=None, type=Path)
    return parser.parse_args()


class StaffCountClassifier(nn.Module):
    def __init__(self, model: FullPageStudent) -> None:
        super().__init__()
        self.model = model
        self.max_staffs = int(model.max_staffs)

    def forward(self, images_unit: torch.Tensor) -> torch.Tensor:
        images = images_unit.mul(2.0).sub(1.0)
        outputs = self.model(images, max_staffs=self.max_staffs, max_seq_len=1)
        return outputs["staff_count_logits"]


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_full_validation_batch(
    val_manifest: Path, checkpoint: dict[str, Any], cli_args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    rows = read_jsonl(val_manifest)
    stats = stats_from_config(checkpoint["model_config"])
    args = loader_args(checkpoint, cli_args)
    args.batch_size = len(rows)
    loader = make_loader(rows, args=args, stats=stats, shuffle=False)
    single_batch = next(iter(loader))
    return move_batch_to_device(single_batch, device)


def staff_count_accuracy(classifier: StaffCountClassifier, images_unit: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        predictions = classifier(images_unit).argmax(dim=-1)
    return float((predictions == labels).float().mean().item())


def token_metrics_under_images(
    model: FullPageStudent, images_unit: torch.Tensor, batch: dict[str, Any]
) -> dict[str, Any]:
    images = images_unit.mul(2.0).sub(1.0)
    max_staffs = int(batch["sequence_mask"].shape[1])
    max_seq_len = int(batch["sequence_mask"].shape[2])
    with torch.no_grad():
        outputs = model(images, max_staffs=max_staffs, max_seq_len=max_seq_len)

    sequence_mask = batch["sequence_mask"]
    correct = 0
    counted = 0
    edit_distance = 0
    reference_length = 0
    for branch in BRANCHES:
        predictions = outputs[f"{branch}_logits"].argmax(dim=-1)
        targets = batch["targets"][branch]
        hits = (predictions == targets) & sequence_mask
        correct += int(hits.sum().item())
        counted += int(sequence_mask.sum().item())
        for example_index in range(sequence_mask.shape[0]):
            for staff_index in range(max_staffs):
                row_mask = sequence_mask[example_index, staff_index]
                if not bool(row_mask.any()):
                    continue
                predicted_sequence = predictions[example_index, staff_index][row_mask].tolist()
                target_sequence = targets[example_index, staff_index][row_mask].tolist()
                edit_distance += int(editdistance.eval(predicted_sequence, target_sequence))
                reference_length += len(target_sequence)
    return {
        "token_accuracy": (correct / counted) if counted else 0.0,
        "mean_ser": (edit_distance / reference_length) if reference_length else 0.0,
    }


def attack_model(
    model: FullPageStudent,
    batch: dict[str, Any],
    *,
    epsilon: float,
    version: str,
    attacks: list[str],
    device: torch.device,
) -> dict[str, Any]:
    classifier = StaffCountClassifier(model).to(device).eval()
    images_unit = batch["images"].mul(0.5).add(0.5).clamp(0.0, 1.0).detach()
    labels = batch["n_staffs"].to(device)

    clean_count_accuracy = staff_count_accuracy(classifier, images_unit, labels)
    clean_tokens = token_metrics_under_images(model, images_unit, batch)

    if epsilon <= 0.0:
        return {
            "clean_count_accuracy": clean_count_accuracy,
            "robust_count_accuracy": clean_count_accuracy,
            "clean_token_accuracy": clean_tokens["token_accuracy"],
            "robust_token_accuracy": clean_tokens["token_accuracy"],
            "clean_mean_ser": clean_tokens["mean_ser"],
            "robust_mean_ser": clean_tokens["mean_ser"],
        }

    adversary = AutoAttack(
        classifier, norm="Linf", eps=epsilon, version=version, device=device, verbose=False
    )
    if version == "custom":
        adversary.attacks_to_run = list(attacks)
    adversarial_unit = adversary.run_standard_evaluation(images_unit, labels, bs=images_unit.shape[0])

    robust_count_accuracy = staff_count_accuracy(classifier, adversarial_unit, labels)
    robust_tokens = token_metrics_under_images(model, adversarial_unit, batch)
    return {
        "clean_count_accuracy": clean_count_accuracy,
        "robust_count_accuracy": robust_count_accuracy,
        "clean_token_accuracy": clean_tokens["token_accuracy"],
        "robust_token_accuracy": robust_tokens["token_accuracy"],
        "clean_mean_ser": clean_tokens["mean_ser"],
        "robust_mean_ser": robust_tokens["mean_ser"],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    print("AutoAttack robustness: before defense (clean surrogate) vs after defense (PGD surrogate)")
    header = (
        f"{'epsilon':>8} | {'before_acc':>12} {'before_ser':>12} | "
        f"{'after_acc':>12} {'after_ser':>12}"
    )
    print(header)
    print("-" * len(header))
    for entry in results:
        before = entry["before_defense"]
        after = entry["after_defense"]
        print(
            f"{entry['epsilon']:>8.4f} | "
            f"{before['robust_count_accuracy']:>12.4f} {before['robust_mean_ser']:>12.4f} | "
            f"{after['robust_count_accuracy']:>12.4f} {after['robust_mean_ser']:>12.4f}"
        )


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    clean_model, clean_checkpoint = load_checkpoint_model(args.clean_checkpoint, device)
    pgd_model, _ = load_checkpoint_model(args.pgd_checkpoint, device)

    batch = load_full_validation_batch(args.val_manifest, clean_checkpoint, args, device)

    results: list[dict[str, Any]] = []
    for epsilon in args.epsilon_grid:
        before_defense = attack_model(
            clean_model, batch, epsilon=epsilon, version=args.version, attacks=args.attacks, device=device
        )
        after_defense = attack_model(
            pgd_model, batch, epsilon=epsilon, version=args.version, attacks=args.attacks, device=device
        )
        results.append(
            {"epsilon": epsilon, "before_defense": before_defense, "after_defense": after_defense}
        )

    print_comparison_table(results)

    payload = {
        "schema": "homr_surrogate_autoattack_comparison_v1",
        "attack": "AutoAttack",
        "attack_target": "staff_count_head",
        "input_domain": "unit_interval_0_1",
        "version": args.version,
        "attacks_to_run": args.attacks if args.version == "custom" else args.version,
        "clean_checkpoint": str(args.clean_checkpoint),
        "pgd_checkpoint": str(args.pgd_checkpoint),
        "val_manifest": str(args.val_manifest),
        "device": str(device),
        "val_rows": int(batch["images"].shape[0]),
        "epsilon_grid": list(args.epsilon_grid),
        "results": results,
    }
    out_path = args.out_dir / "autoattack_comparison.json"
    write_json_atomic(out_path, payload)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
