from __future__ import annotations

from typing import Any

import torch

from distillation.train_student import compute_loss


def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: dict[str, torch.Tensor],
    sequence_mask: torch.Tensor,
    staff_exists: torch.Tensor,
    n_staffs: torch.Tensor,
    epsilon: float,
    steps: int,
    alpha: float,
    staff_loss_weight: float,
    input_min: float = -1.0,
    input_max: float = 1.0,
) -> torch.Tensor:
    if epsilon <= 0.0 or steps <= 0:
        return images.detach()

    model_was_training = model.training
    model.eval()

    original = images.detach()
    max_staffs = int(sequence_mask.shape[1])
    max_seq_len = int(sequence_mask.shape[2])

    loss_batch: dict[str, Any] = {
        "targets": targets,
        "sequence_mask": sequence_mask,
        "staff_exists": staff_exists,
        "n_staffs": n_staffs,
    }

    noise = torch.empty_like(original).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(original + noise, input_min, input_max).detach()

    for _ in range(int(steps)):
        x_adv.requires_grad_(True)
        outputs = model(x_adv, max_staffs=max_staffs, max_seq_len=max_seq_len)
        loss, _ = compute_loss(outputs, loss_batch, staff_loss_weight=staff_loss_weight)
        gradient = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * gradient.sign()
            x_adv = torch.max(torch.min(x_adv, original + epsilon), original - epsilon)
            x_adv = torch.clamp(x_adv, input_min, input_max)
        x_adv = x_adv.detach()

    model.train(model_was_training)
    return x_adv.detach()
