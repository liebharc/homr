#!/usr/bin/env python3
"""
Train a full-page Track C student on HOMR-factorized page targets.

Input rows are produced by distillation/vocab.py and are page-level JSONL rows
with this target shape:

    homr_target_staffs: [
        {
            rhythm_ids: [...],
            pitch_ids: [...],
            lift_ids: [...],
            articulation_ids: [...],
            position_ids: [...],
            mask: [...]
        },
        ...
    ]

This script intentionally trains from one rendered full page image to ordered
staff sequences. It does not split the dataset into one row per staff, does not
invent a <STAFF_BREAK> token, and does not call HOMR neural inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - environment/setup error.
    raise RuntimeError(
        "PyTorch is required for distillation/train_student.py. Install torch "
        "in the active environment before running student training."
    ) from exc

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - environment/setup error.
    raise RuntimeError(
        "Pillow is required for image loading. Install pillow in the active "
        "environment before running student training."
    ) from exc

try:
    from progress import emit_pipeline_event
except Exception:  # pragma: no cover - keeps script usable if copied alone.
    def emit_pipeline_event(*, quiet: bool = False, **kwargs: Any) -> dict[str, Any]:
        payload = {key: value for key, value in kwargs.items() if key != "stage"}
        if not quiet:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        return payload


BRANCHES = ("rhythm", "pitch", "lift", "articulation", "position")
BRANCH_FIELDS = {
    "rhythm": "rhythm_ids",
    "pitch": "pitch_ids",
    "lift": "lift_ids",
    "articulation": "articulation_ids",
    "position": "position_ids",
}
PAD_ID = 0


class TrainStudentError(RuntimeError):
    """Raised when the student training run cannot proceed safely."""


@dataclass(frozen=True)
class ManifestStats:
    rows: int
    max_staffs: int
    max_seq_len: int
    vocab_sizes: dict[str, int]


@dataclass
class EpochMetrics:
    epoch: int
    split: str
    rows: int
    loss: float
    rhythm_loss: float
    pitch_loss: float
    lift_loss: float
    articulation_loss: float
    position_loss: float
    staff_exists_loss: float
    staff_count_loss: float
    token_count: int
    staff_count: int
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a full-page student on HOMR-factorized encoded manifests."
    )
    parser.add_argument(
        "--train-manifest",
        required=True,
        type=Path,
        help="Path to training_manifest.train.encoded.jsonl.",
    )
    parser.add_argument(
        "--val-manifest",
        default=None,
        type=Path,
        help="Optional path to training_manifest.val.encoded.jsonl.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Run directory. Writes latest.pt, best_clean.pt, training_log.jsonl, metrics.json.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=768)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--staff-loss-weight", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <out-dir>/latest.pt when it exists.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        type=Path,
        help="Repository root for resolving relative image paths. Defaults to cwd.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print uniform terminal progress every N batches. Use 0 to print only stage start/end.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress uniform terminal progress events. JSONL/metrics outputs are still written.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise TrainStudentError(f"Manifest does not exist: {path}")
    if not path.is_file():
        raise TrainStudentError(f"Manifest path is not a file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TrainStudentError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TrainStudentError(
                    f"Line {line_number} of {path} is {type(row).__name__}; expected object."
                )
            row["_manifest_path"] = str(path)
            row["_manifest_line_number"] = line_number
            validate_encoded_row(row, path=path, line_number=line_number)
            rows.append(row)

    if not rows:
        raise TrainStudentError(f"Manifest has no non-empty rows: {path}")
    return rows


def validate_encoded_row(row: dict[str, Any], *, path: Path, line_number: int) -> None:
    if row.get("schema") != "homr_factorized_page_training_manifest_v1":
        raise TrainStudentError(
            f"{path}:{line_number} has schema {row.get('schema')!r}; "
            "expected 'homr_factorized_page_training_manifest_v1'."
        )
    if not isinstance(row.get("image_path"), str) or not row["image_path"].strip():
        raise TrainStudentError(f"{path}:{line_number} is missing non-empty image_path.")
    staffs = row.get("homr_target_staffs")
    if not isinstance(staffs, list) or not staffs:
        raise TrainStudentError(f"{path}:{line_number} has no homr_target_staffs.")

    for staff_index, staff in enumerate(staffs):
        if not isinstance(staff, dict):
            raise TrainStudentError(
                f"{path}:{line_number} staff {staff_index} is {type(staff).__name__}; expected object."
            )
        lengths: set[int] = set()
        for field in (*BRANCH_FIELDS.values(), "mask"):
            value = staff.get(field)
            if not isinstance(value, list) or not value:
                raise TrainStudentError(
                    f"{path}:{line_number} staff {staff_index} has invalid {field}; expected non-empty list."
                )
            lengths.add(len(value))
        if len(lengths) != 1:
            raise TrainStudentError(
                f"{path}:{line_number} staff {staff_index} branch/mask lengths differ: {sorted(lengths)}"
            )
        for branch, field in BRANCH_FIELDS.items():
            for token_index, value in enumerate(staff[field]):
                if not isinstance(value, int) or value < 0:
                    raise TrainStudentError(
                        f"{path}:{line_number} staff {staff_index} {branch}[{token_index}] "
                        f"is {value!r}; expected non-negative int."
                    )
        if not all(isinstance(value, bool) for value in staff["mask"]):
            raise TrainStudentError(
                f"{path}:{line_number} staff {staff_index} mask must contain booleans."
            )

    declared_staffs = row.get("n_staffs")
    if declared_staffs is not None and int(declared_staffs) != len(staffs):
        raise TrainStudentError(
            f"{path}:{line_number} n_staffs={declared_staffs}, but homr_target_staffs has {len(staffs)}."
        )


def candidate_roots(manifest_path: Path, repo_root: Path | None) -> list[Path]:
    roots: list[Path] = []
    if repo_root is not None:
        roots.append(repo_root)
    roots.append(Path.cwd())
    roots.append(manifest_path.parent)
    for parent in manifest_path.resolve().parents:
        roots.append(parent)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def resolve_image_path(raw_path: str, manifest_path: Path, repo_root: Path | None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        if path.exists():
            return path
        raise TrainStudentError(f"Image path does not exist: {path}")

    normalized = Path(raw_path.replace("\\", "/"))
    for root in candidate_roots(manifest_path, repo_root):
        candidate = root / normalized
        if candidate.exists():
            return candidate
    raise TrainStudentError(
        f"Could not resolve image path {raw_path!r} from manifest {manifest_path}. "
        "Run from the repository root or pass --repo-root."
    )


def manifest_stats(rows: Iterable[dict[str, Any]]) -> ManifestStats:
    row_count = 0
    max_staffs = 0
    max_seq_len = 0
    max_ids = {branch: 0 for branch in BRANCHES}

    for row in rows:
        row_count += 1
        staffs = row["homr_target_staffs"]
        max_staffs = max(max_staffs, len(staffs))
        for staff in staffs:
            max_seq_len = max(max_seq_len, len(staff["mask"]))
            for branch, field in BRANCH_FIELDS.items():
                max_ids[branch] = max(max_ids[branch], max(int(value) for value in staff[field]))

    if row_count == 0:
        raise TrainStudentError("Cannot compute stats for empty row set.")
    return ManifestStats(
        rows=row_count,
        max_staffs=max_staffs,
        max_seq_len=max_seq_len,
        vocab_sizes={branch: max_id + 1 for branch, max_id in max_ids.items()},
    )


class PageStaffDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        image_height: int,
        image_width: int,
        repo_root: Path | None,
    ) -> None:
        self.rows = rows
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.repo_root = repo_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        manifest_path = Path(str(row["_manifest_path"]))
        image_path = resolve_image_path(str(row["image_path"]), manifest_path, self.repo_root)
        image = load_page_image(image_path, height=self.image_height, width=self.image_width)
        return {
            "image": image,
            "row": row,
            "image_path": str(image_path),
        }


def load_page_image(path: Path, *, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((width, height), resample=Image.BILINEAR)
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        data = data.view(height, width).to(dtype=torch.float32).div_(255.0)
    # One grayscale channel; normalize to approximately [-1, 1].
    return data.unsqueeze(0).sub_(0.5).div_(0.5)


def collate_page_batch(
    samples: list[dict[str, Any]],
    *,
    max_staffs_limit: int,
    max_seq_len_limit: int,
) -> dict[str, Any]:
    if not samples:
        raise TrainStudentError("Cannot collate an empty batch.")

    batch_size = len(samples)
    max_staffs = min(max(len(sample["row"]["homr_target_staffs"]) for sample in samples), max_staffs_limit)
    max_seq_len = min(
        max(
            len(staff["mask"])
            for sample in samples
            for staff in sample["row"]["homr_target_staffs"][:max_staffs]
        ),
        max_seq_len_limit,
    )

    images = torch.stack([sample["image"] for sample in samples], dim=0)
    targets = {
        branch: torch.full((batch_size, max_staffs, max_seq_len), PAD_ID, dtype=torch.long)
        for branch in BRANCHES
    }
    sequence_mask = torch.zeros((batch_size, max_staffs, max_seq_len), dtype=torch.bool)
    staff_exists = torch.zeros((batch_size, max_staffs), dtype=torch.float32)
    n_staffs = torch.zeros((batch_size,), dtype=torch.long)

    metadata: list[dict[str, Any]] = []
    for batch_index, sample in enumerate(samples):
        row = sample["row"]
        staffs = row["homr_target_staffs"][:max_staffs]
        n_staffs[batch_index] = min(len(row["homr_target_staffs"]), max_staffs)
        metadata.append(
            {
                "score_id": row.get("score_id"),
                "page_id": row.get("page_id"),
                "page_number": row.get("page_number"),
                "image_path": sample["image_path"],
                "manifest_line_number": row.get("_manifest_line_number"),
            }
        )
        for staff_index, staff in enumerate(staffs):
            staff_exists[batch_index, staff_index] = 1.0
            length = min(len(staff["mask"]), max_seq_len)
            mask_values = torch.tensor(staff["mask"][:length], dtype=torch.bool)
            sequence_mask[batch_index, staff_index, :length] = mask_values
            for branch, field in BRANCH_FIELDS.items():
                targets[branch][batch_index, staff_index, :length] = torch.tensor(
                    staff[field][:length], dtype=torch.long
                )

    return {
        "images": images,
        "targets": targets,
        "sequence_mask": sequence_mask,
        "staff_exists": staff_exists,
        "n_staffs": n_staffs,
        "metadata": metadata,
    }


class FullPageStudent(nn.Module):
    def __init__(
        self,
        *,
        vocab_sizes: dict[str, int],
        max_staffs: int,
        max_seq_len: int,
        embed_dim: int,
        decoder_layers: int,
        decoder_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vocab_sizes = dict(vocab_sizes)
        self.max_staffs = int(max_staffs)
        self.max_seq_len = int(max_seq_len)
        self.embed_dim = int(embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.page_norm = nn.LayerNorm(embed_dim)
        self.staff_embedding = nn.Embedding(max_staffs, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=decoder_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=decoder_layers)
        self.dropout = nn.Dropout(dropout)

        self.branch_heads = nn.ModuleDict(
            {
                branch: nn.Linear(embed_dim, int(size))
                for branch, size in self.vocab_sizes.items()
            }
        )
        self.staff_exists_head = nn.Linear(embed_dim, 1)
        self.staff_count_head = nn.Linear(embed_dim, max_staffs + 1)

    def forward(self, images: torch.Tensor, *, max_staffs: int, max_seq_len: int) -> dict[str, torch.Tensor]:
        if max_staffs > self.max_staffs:
            raise TrainStudentError(f"Batch max_staffs {max_staffs} exceeds model max {self.max_staffs}.")
        if max_seq_len > self.max_seq_len:
            raise TrainStudentError(f"Batch max_seq_len {max_seq_len} exceeds model max {self.max_seq_len}.")

        batch_size = images.shape[0]
        page = self.encoder(images).flatten(1)
        page = self.page_norm(page)

        staff_ids = torch.arange(max_staffs, device=images.device)
        position_ids = torch.arange(max_seq_len, device=images.device)
        staff = self.staff_embedding(staff_ids).view(1, max_staffs, 1, self.embed_dim)
        pos = self.position_embedding(position_ids).view(1, 1, max_seq_len, self.embed_dim)
        page_context = page.view(batch_size, 1, 1, self.embed_dim)
        queries = self.query_norm(page_context + staff + pos)
        queries = queries.view(batch_size, max_staffs * max_seq_len, self.embed_dim)
        decoded = self.decoder(self.dropout(queries))
        decoded = decoded.view(batch_size, max_staffs, max_seq_len, self.embed_dim)

        branch_logits = {
            branch: head(decoded)
            for branch, head in self.branch_heads.items()
        }
        # Staff-existence logits are produced from the first sequence query of each staff.
        staff_exists_logits = self.staff_exists_head(decoded[:, :, 0, :]).squeeze(-1)
        staff_count_logits = self.staff_count_head(page)
        return {
            **{f"{branch}_logits": logits for branch, logits in branch_logits.items()},
            "staff_exists_logits": staff_exists_logits,
            "staff_count_logits": staff_count_logits,
        }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["images"] = batch["images"].to(device, non_blocking=True)
    moved["sequence_mask"] = batch["sequence_mask"].to(device, non_blocking=True)
    moved["staff_exists"] = batch["staff_exists"].to(device, non_blocking=True)
    moved["n_staffs"] = batch["n_staffs"].to(device, non_blocking=True)
    moved["targets"] = {
        key: value.to(device, non_blocking=True) for key, value in batch["targets"].items()
    }
    return moved


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    staff_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    sequence_mask = batch["sequence_mask"]
    if not bool(sequence_mask.any()):
        raise TrainStudentError("Batch has no valid target tokens after masking.")

    branch_losses: dict[str, torch.Tensor] = {}
    for branch in BRANCHES:
        logits = outputs[f"{branch}_logits"]
        targets = batch["targets"][branch]
        active_logits = logits[sequence_mask]
        active_targets = targets[sequence_mask]
        branch_losses[branch] = F.cross_entropy(active_logits, active_targets)

    staff_exists_loss = F.binary_cross_entropy_with_logits(
        outputs["staff_exists_logits"], batch["staff_exists"]
    )
    staff_count_loss = F.cross_entropy(outputs["staff_count_logits"], batch["n_staffs"])
    token_loss = sum(branch_losses.values())
    total = token_loss + staff_loss_weight * (staff_exists_loss + staff_count_loss)

    loss_parts = {f"{branch}_loss": float(value.detach().cpu()) for branch, value in branch_losses.items()}
    loss_parts["staff_exists_loss"] = float(staff_exists_loss.detach().cpu())
    loss_parts["staff_count_loss"] = float(staff_count_loss.detach().cpu())
    loss_parts["loss"] = float(total.detach().cpu())
    return total, loss_parts


def assert_finite_loss(loss: torch.Tensor, loss_parts: dict[str, float]) -> None:
    if not torch.isfinite(loss).item():
        raise TrainStudentError(f"Non-finite total loss: {loss_parts}")
    for name, value in loss_parts.items():
        if not math.isfinite(float(value)):
            raise TrainStudentError(f"Non-finite {name}: {value}; all losses={loss_parts}")


def run_epoch(
    model: FullPageStudent,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    split: str,
    staff_loss_weight: float,
    grad_clip: float,
    progress_every: int,
    quiet: bool,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    started = time.time()

    totals = {
        "loss": 0.0,
        "rhythm_loss": 0.0,
        "pitch_loss": 0.0,
        "lift_loss": 0.0,
        "articulation_loss": 0.0,
        "position_loss": 0.0,
        "staff_exists_loss": 0.0,
        "staff_count_loss": 0.0,
    }
    rows = 0
    token_count = 0
    staff_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    total_rows = len(loader.dataset) if hasattr(loader, "dataset") else None
    total_batches = len(loader)

    emit_pipeline_event(
        stage="student_train",
        event="split_start",
        status="running",
        unit="rows",
        done=0,
        total=total_rows,
        start_time=started,
        metrics={"epoch": epoch, "split": split, "device": str(device), "batches_total": total_batches},
        quiet=quiet,
    )

    with context:
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = move_batch_to_device(raw_batch, device)
            batch_rows = int(batch["images"].shape[0])
            max_staffs = int(batch["sequence_mask"].shape[1])
            max_seq_len = int(batch["sequence_mask"].shape[2])

            outputs = model(batch["images"], max_staffs=max_staffs, max_seq_len=max_seq_len)
            loss, loss_parts = compute_loss(outputs, batch, staff_loss_weight=staff_loss_weight)
            assert_finite_loss(loss, loss_parts)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            rows += batch_rows
            token_count += int(batch["sequence_mask"].sum().detach().cpu())
            staff_count += int(batch["staff_exists"].sum().detach().cpu())
            for key in totals:
                totals[key] += float(loss_parts[key]) * batch_rows

            if progress_every > 0 and (batch_index % progress_every == 0 or batch_index == total_batches):
                running_loss = totals["loss"] / max(1, rows)
                emit_pipeline_event(
                    stage="student_train",
                    event="batch_progress",
                    status="running",
                    unit="rows",
                    done=rows,
                    total=total_rows,
                    start_time=started,
                    counts={"rows": rows, "token_count": token_count, "staff_count": staff_count},
                    metrics={
                        "epoch": epoch,
                        "split": split,
                        "batch": batch_index,
                        "batches_total": total_batches,
                        "loss": running_loss,
                        "last_loss": loss_parts["loss"],
                        "rhythm_loss": loss_parts["rhythm_loss"],
                        "pitch_loss": loss_parts["pitch_loss"],
                        "lift_loss": loss_parts["lift_loss"],
                        "articulation_loss": loss_parts["articulation_loss"],
                        "position_loss": loss_parts["position_loss"],
                        "staff_exists_loss": loss_parts["staff_exists_loss"],
                        "staff_count_loss": loss_parts["staff_count_loss"],
                        "device": str(device),
                    },
                    quiet=quiet,
                )

    if rows == 0:
        raise TrainStudentError(f"No rows were processed for split {split}.")
    averaged = {key: value / rows for key, value in totals.items()}
    metrics = EpochMetrics(
        epoch=epoch,
        split=split,
        rows=rows,
        loss=averaged["loss"],
        rhythm_loss=averaged["rhythm_loss"],
        pitch_loss=averaged["pitch_loss"],
        lift_loss=averaged["lift_loss"],
        articulation_loss=averaged["articulation_loss"],
        position_loss=averaged["position_loss"],
        staff_exists_loss=averaged["staff_exists_loss"],
        staff_count_loss=averaged["staff_count_loss"],
        token_count=token_count,
        staff_count=staff_count,
        seconds=time.time() - started,
    )
    emit_pipeline_event(
        stage="student_train",
        event="split_done",
        status="ok",
        unit="rows",
        done=rows,
        total=total_rows,
        elapsed_seconds=metrics.seconds,
        counts={"rows": rows, "token_count": token_count, "staff_count": staff_count},
        metrics=asdict(metrics) | {"device": str(device)},
        quiet=quiet,
    )
    return metrics


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_checkpoint(
    path: Path,
    *,
    model: FullPageStudent,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    train_stats: ManifestStats,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "homr_full_page_student_checkpoint_v1",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": {
                "vocab_sizes": train_stats.vocab_sizes,
                "max_staffs": train_stats.max_staffs,
                "max_seq_len": train_stats.max_seq_len,
                "embed_dim": args.embed_dim,
                "decoder_layers": args.decoder_layers,
                "decoder_heads": args.decoder_heads,
                "dropout": args.dropout,
            },
            "training_args": vars(args),
            "train_stats": asdict(train_stats),
            "metrics": metrics,
        },
        tmp,
    )
    tmp.replace(path)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise TrainStudentError("--device cuda was requested, but CUDA is not available.")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    stats: ManifestStats,
    shuffle: bool,
) -> DataLoader:
    dataset = PageStaffDataset(
        rows,
        image_height=args.image_height,
        image_width=args.image_width,
        repo_root=args.repo_root,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda samples: collate_page_batch(
            samples,
            max_staffs_limit=stats.max_staffs,
            max_seq_len_limit=stats.max_seq_len,
        ),
    )


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise TrainStudentError(f"--epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise TrainStudentError(f"--batch-size must be positive, got {args.batch_size}")
    if args.image_height <= 0 or args.image_width <= 0:
        raise TrainStudentError("--image-height and --image-width must be positive.")

    run_started = time.time()
    seed_everything(args.seed)
    device = choose_device(args.device)

    emit_pipeline_event(
        stage="student_train",
        event="start",
        status="running",
        unit="epochs",
        done=0,
        total=args.epochs,
        start_time=run_started,
        metrics={"device": str(device), "batch_size": args.batch_size, "epochs": args.epochs},
        paths={"train_manifest": args.train_manifest, "val_manifest": args.val_manifest, "out_dir": args.out_dir},
        quiet=args.quiet,
    )

    train_rows = read_jsonl(args.train_manifest)
    val_rows = read_jsonl(args.val_manifest) if args.val_manifest is not None else []

    # Include validation rows in shape/vocab bounds so evaluation cannot hit an
    # out-of-range class if the validation split contains a component unseen in train.
    stats_rows = train_rows + val_rows if val_rows else train_rows
    stats = manifest_stats(stats_rows)

    train_loader = make_loader(train_rows, args=args, stats=stats, shuffle=True)
    val_loader = make_loader(val_rows, args=args, stats=stats, shuffle=False) if val_rows else None

    model = FullPageStudent(
        vocab_sizes=stats.vocab_sizes,
        max_staffs=stats.max_staffs,
        max_seq_len=stats.max_seq_len,
        embed_dim=args.embed_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.out_dir / "latest.pt"
    best_path = args.out_dir / "best_clean.pt"
    log_path = args.out_dir / "training_log.jsonl"
    metrics_path = args.out_dir / "metrics.json"

    start_epoch = 1
    best_loss = float("inf")
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        previous_metrics = checkpoint.get("metrics", {})
        best_loss = float(previous_metrics.get("best_loss", float("inf")))

    run_summary: dict[str, Any] = {
        "schema": "homr_full_page_student_training_metrics_v1",
        "status": "running",
        "device": str(device),
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest) if args.val_manifest else None,
        "out_dir": str(args.out_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "max_staffs": stats.max_staffs,
        "max_seq_len": stats.max_seq_len,
        "vocab_sizes": stats.vocab_sizes,
        "epochs_requested": args.epochs,
        "epochs": [],
        "best_loss": best_loss,
    }
    write_json_atomic(metrics_path, run_summary)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            split="train",
            staff_loss_weight=args.staff_loss_weight,
            grad_clip=args.grad_clip,
            progress_every=args.progress_every,
            quiet=args.quiet,
        )
        append_jsonl(log_path, asdict(train_metrics))

        epoch_record: dict[str, Any] = {"train": asdict(train_metrics)}
        score_loss = train_metrics.loss

        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                epoch=epoch,
                split="val",
                staff_loss_weight=args.staff_loss_weight,
                grad_clip=args.grad_clip,
                progress_every=args.progress_every,
                quiet=args.quiet,
            )
            append_jsonl(log_path, asdict(val_metrics))
            epoch_record["val"] = asdict(val_metrics)
            score_loss = val_metrics.loss

        is_best = score_loss < best_loss
        if is_best:
            best_loss = score_loss

        run_summary["epochs"].append(epoch_record)
        run_summary["best_loss"] = best_loss
        run_summary["status"] = "ok"
        write_json_atomic(metrics_path, run_summary)
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            train_stats=stats,
            metrics=run_summary,
        )
        if is_best:
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                train_stats=stats,
                metrics=run_summary,
            )

        emit_pipeline_event(
            stage="student_train",
            event="epoch_done",
            status="ok",
            unit="epochs",
            done=epoch,
            total=args.epochs,
            start_time=run_started,
            metrics={"epoch": epoch, "best_loss": best_loss, "score_loss": score_loss},
            extra=epoch_record,
            quiet=args.quiet,
        )

    if not latest_path.exists():
        raise TrainStudentError(f"Training finished but did not write {latest_path}")
    if not log_path.exists():
        raise TrainStudentError(f"Training finished but did not write {log_path}")

    final_payload = {
        "event": "done",
        "status": "ok",
        "latest": str(latest_path),
        "best_clean": str(best_path),
        "training_log": str(log_path),
        "metrics": str(metrics_path),
        "best_loss": best_loss,
    }
    emit_pipeline_event(
        stage="student_train",
        event="done",
        status="ok",
        unit="epochs",
        done=args.epochs,
        total=args.epochs,
        start_time=run_started,
        metrics={"best_loss": best_loss, "device": str(device)},
        paths={
            "latest": latest_path,
            "best_clean": best_path,
            "training_log": log_path,
            "metrics": metrics_path,
        },
        extra=final_payload,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainStudentError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
