#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import editdistance
import torch

from distillation.pgd_attack import pgd_attack
from distillation.train_student import (
    BRANCHES,
    FullPageStudent,
    ManifestStats,
    choose_device,
    make_loader,
    move_batch_to_device,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a clean surrogate against a PGD-trained surrogate under an L-inf epsilon grid."
    )
    parser.add_argument("--clean-checkpoint", required=True, type=Path)
    parser.add_argument("--pgd-checkpoint", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--epsilon-grid", required=True, nargs="+", type=float)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-alpha", type=float, default=0.005)
    parser.add_argument("--staff-loss-weight", type=float, default=0.10)
    parser.add_argument("--repo-root", default=None, type=Path)
    return parser.parse_args()


def load_checkpoint_model(path: Path, device: torch.device) -> tuple[FullPageStudent, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = FullPageStudent(
        vocab_sizes={branch: int(size) for branch, size in config["vocab_sizes"].items()},
        max_staffs=int(config["max_staffs"]),
        max_seq_len=int(config["max_seq_len"]),
        embed_dim=int(config["embed_dim"]),
        decoder_layers=int(config["decoder_layers"]),
        decoder_heads=int(config["decoder_heads"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def stats_from_config(config: dict[str, Any]) -> ManifestStats:
    return ManifestStats(
        rows=0,
        max_staffs=int(config["max_staffs"]),
        max_seq_len=int(config["max_seq_len"]),
        vocab_sizes={branch: int(size) for branch, size in config["vocab_sizes"].items()},
    )


def loader_args(checkpoint: dict[str, Any], cli_args: argparse.Namespace) -> argparse.Namespace:
    training_args = checkpoint.get("training_args", {})
    return argparse.Namespace(
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        image_height=int(training_args.get("image_height", 768)),
        image_width=int(training_args.get("image_width", 512)),
        repo_root=cli_args.repo_root,
    )


def evaluate_model(
    model: FullPageStudent,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    epsilon: float,
    steps: int,
    alpha: float,
    staff_loss_weight: float,
) -> dict[str, Any]:
    correct = {branch: 0 for branch in BRANCHES}
    counted = {branch: 0 for branch in BRANCHES}
    edit_distance = {branch: 0 for branch in BRANCHES}
    reference_length = {branch: 0 for branch in BRANCHES}

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        max_staffs = int(batch["sequence_mask"].shape[1])
        max_seq_len = int(batch["sequence_mask"].shape[2])

        images = pgd_attack(
            model,
            batch["images"],
            batch["targets"],
            batch["sequence_mask"],
            batch["staff_exists"],
            batch["n_staffs"],
            epsilon=epsilon,
            steps=steps,
            alpha=alpha,
            staff_loss_weight=staff_loss_weight,
        )

        with torch.no_grad():
            outputs = model(images, max_staffs=max_staffs, max_seq_len=max_seq_len)

        sequence_mask = batch["sequence_mask"]
        for branch in BRANCHES:
            predictions = outputs[f"{branch}_logits"].argmax(dim=-1)
            targets = batch["targets"][branch]
            hits = (predictions == targets) & sequence_mask
            correct[branch] += int(hits.sum().item())
            counted[branch] += int(sequence_mask.sum().item())

            batch_size = sequence_mask.shape[0]
            for example_index in range(batch_size):
                for staff_index in range(max_staffs):
                    row_mask = sequence_mask[example_index, staff_index]
                    if not bool(row_mask.any()):
                        continue
                    predicted_sequence = predictions[example_index, staff_index][row_mask].tolist()
                    target_sequence = targets[example_index, staff_index][row_mask].tolist()
                    edit_distance[branch] += int(editdistance.eval(predicted_sequence, target_sequence))
                    reference_length[branch] += len(target_sequence)

    branch_accuracy = {
        branch: (correct[branch] / counted[branch]) if counted[branch] else 0.0
        for branch in BRANCHES
    }
    branch_ser = {
        branch: (edit_distance[branch] / reference_length[branch]) if reference_length[branch] else 0.0
        for branch in BRANCHES
    }
    total_correct = sum(correct.values())
    total_counted = sum(counted.values())
    total_edit_distance = sum(edit_distance.values())
    total_reference_length = sum(reference_length.values())
    return {
        "overall_accuracy": (total_correct / total_counted) if total_counted else 0.0,
        "branch_accuracy": branch_accuracy,
        "mean_ser": (total_edit_distance / total_reference_length) if total_reference_length else 0.0,
        "branch_ser": branch_ser,
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
    header = f"{'epsilon':>9} | {'clean_acc':>10} {'clean_ser':>10} | {'pgd_acc':>10} {'pgd_ser':>10}"
    print(header)
    print("-" * len(header))
    for entry in results:
        clean = entry["clean"]
        pgd = entry["pgd"]
        print(
            f"{entry['epsilon']:>9.4f} | "
            f"{clean['overall_accuracy']:>10.4f} {clean['mean_ser']:>10.4f} | "
            f"{pgd['overall_accuracy']:>10.4f} {pgd['mean_ser']:>10.4f}"
        )


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    clean_model, clean_checkpoint = load_checkpoint_model(args.clean_checkpoint, device)
    pgd_model, pgd_checkpoint = load_checkpoint_model(args.pgd_checkpoint, device)

    stats = stats_from_config(clean_checkpoint["model_config"])
    val_rows = read_jsonl(args.val_manifest)
    loader = make_loader(val_rows, args=loader_args(clean_checkpoint, args), stats=stats, shuffle=False)

    results: list[dict[str, Any]] = []
    for epsilon in args.epsilon_grid:
        clean_metrics = evaluate_model(
            clean_model,
            loader,
            device=device,
            epsilon=epsilon,
            steps=args.pgd_steps,
            alpha=args.pgd_alpha,
            staff_loss_weight=args.staff_loss_weight,
        )
        pgd_metrics = evaluate_model(
            pgd_model,
            loader,
            device=device,
            epsilon=epsilon,
            steps=args.pgd_steps,
            alpha=args.pgd_alpha,
            staff_loss_weight=args.staff_loss_weight,
        )
        results.append({"epsilon": epsilon, "clean": clean_metrics, "pgd": pgd_metrics})

    print_comparison_table(results)

    payload = {
        "schema": "homr_surrogate_pgd_comparison_v1",
        "clean_checkpoint": str(args.clean_checkpoint),
        "pgd_checkpoint": str(args.pgd_checkpoint),
        "val_manifest": str(args.val_manifest),
        "device": str(device),
        "pgd_steps": args.pgd_steps,
        "pgd_alpha": args.pgd_alpha,
        "staff_loss_weight": args.staff_loss_weight,
        "val_rows": len(val_rows),
        "epsilon_grid": list(args.epsilon_grid),
        "results": results,
    }
    out_path = args.out_dir / "comparison.json"
    write_json_atomic(out_path, payload)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
