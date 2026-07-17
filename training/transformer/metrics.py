import os
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from transformers import Trainer

from training.transformer.distribute import Distribute
from training.transformer.training_vocabulary import map_rhythm_symbol_with_position

LOSS_COMPONENTS = [
    "loss_rhythm",
    "loss_pitch",
    "loss_lift",
    "loss_consist",
    "loss_position",
    "loss_articulations",
    "loss_slurs",
]


class HomrTrainer(Trainer):
    """
    Custom Trainer for HOMR transformer.
    Tracks per-branch losses, per-branch token accuracy,
    and overall token-weighted (micro) accuracy.
    """

    def __init__(self, *args: Any, distribute: Distribute, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._distribute = distribute

        # Training loss accumulators
        self._loss_sum: dict[str, float] = defaultdict(float)
        self._loss_count: dict[str, int] = defaultdict(int)

        # Evaluation accumulators
        self._eval_loss_sum: dict[str, float] = defaultdict(float)
        self._eval_loss_count: dict[str, int] = defaultdict(int)
        self._acc_correct: dict[str, int] = defaultdict(int)
        self._acc_total: dict[str, int] = defaultdict(int)

        self._rhythm_map = map_rhythm_symbol_with_position()

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if model.training:
            # Calculate history dropout probability (ramps up from start to end)
            config = getattr(model, "config", None)
            if config:
                start_p = config.history_dropout_start_prob
                end_p = config.history_dropout_end_prob
                decay = config.history_dropout_decay_steps
                step = self.state.global_step
                dropout_prob = min(end_p, start_p + (end_p - start_p) * (step / decay))
                inputs["history_dropout_prob"] = dropout_prob
            else:
                inputs["history_dropout_prob"] = 0.0

        outputs = model(**inputs)
        loss = outputs["loss"]

        if model.training:
            for component in LOSS_COMPONENTS:
                self._loss_sum[component] += outputs[component].detach().item()
                self._loss_count[component] += 1

        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)

        # Accumulate per-branch eval losses
        for k in LOSS_COMPONENTS:
            self._eval_loss_sum[k] += outputs[k].detach().item()
            self._eval_loss_count[k] += 1

        # Accumulate per-branch accuracy stats
        self._accumulate_accuracy(outputs, inputs)

        if not self._distribute.enabled:
            loss = outputs["loss"].mean().detach().cpu()
        else:
            # when using distributed training, loss on each device should be gathered,
            # which can only run on gpu
            loss = outputs["loss"].mean().detach()
        return loss, None, None

    def _accumulate_accuracy(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, torch.Tensor],
    ) -> None:
        rhythm, pitch, lift, position, articulations, slurs = outputs["logits"]
        # Pull binary mask from inputs and align with y_out (shift by 1)
        eval_mask = inputs["mask"][:, 1:]

        rhythms_output = inputs["rhythms"][:, 1:]
        _rhythm_map_on_device = self._rhythm_map.to(rhythms_output.device)
        rhythms_output_clamped = rhythms_output.clamp(min=0, max=_rhythm_map_on_device.shape[0] - 1)
        is_positional = _rhythm_map_on_device[rhythms_output_clamped]

        branches = [
            ("rhythm", rhythm, inputs["rhythms"]),
            ("pitch", pitch, inputs["pitchs"]),
            ("lift", lift, inputs["lifts"]),
            ("position", position, inputs["positions"]),
            ("articulations", articulations, inputs["articulations"]),
            ("slurs", slurs, inputs["slurs"]),
        ]

        for name, logits, input_labels in branches:
            preds = logits.argmax(dim=-1)
            labels = input_labels[:, 1:]  # next-token alignment

            min_len = min(preds.shape[1], labels.shape[1], eval_mask.shape[1])
            preds = preds[:, :min_len]
            labels = labels[:, :min_len]
            current_eval_mask = eval_mask[:, :min_len]

            # Combine explicit ignore_index with the binary mask
            mask = (labels != -100) & current_eval_mask
            if not mask.any():
                continue

            correct = ((preds == labels) & mask).sum().item()
            total = mask.sum().item()

            self._acc_correct[name] += int(correct)
            self._acc_total[name] += int(total)

            if name == "pitch" and os.environ.get("DEBUG_PITCH_RHYTHM_ACCURACY", "0") == "1":
                current_is_pos = is_positional[:, :min_len]
                is_correct = preds == labels
                is_pos = mask & current_is_pos
                is_nonpos = mask & (~current_is_pos)
                self._acc_correct["pitch_posRhy"] += int((is_correct & is_pos).sum().item())
                self._acc_total["pitch_posRhy"] += int(is_pos.sum().item())
                self._acc_correct["pitch_nonposRhy"] += int((is_correct & is_nonpos).sum().item())
                self._acc_total["pitch_nonposRhy"] += int(is_nonpos.sum().item())

    def _all_reduce(self, d: dict[str, float]) -> None:
        if not self._distribute.enabled or not d:
            return

        keys = sorted(d.keys())
        tensor = torch.tensor(
            [float(d[k]) for k in keys],
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        for i, k in enumerate(keys):
            d[k] = tensor[i].item()

    def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):  # type: ignore
        # Reset eval accumulators
        self._eval_loss_sum.clear()
        self._eval_loss_count.clear()
        self._acc_correct.clear()
        self._acc_total.clear()

        output = super().evaluation_loop(
            *args,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )

        for accumulator in [
            self._eval_loss_sum,
            self._eval_loss_count,
            self._acc_correct,
            self._acc_total,
        ]:
            self._all_reduce(accumulator)

        # Per-component eval losses
        for k in LOSS_COMPONENTS:
            if self._eval_loss_count[k] > 0:
                output.metrics[f"{metric_key_prefix}_{k}"] = (
                    self._eval_loss_sum[k] / self._eval_loss_count[k]
                )

        # Per-branch accuracies
        for k in self._acc_total:
            if self._acc_total[k] > 0:
                output.metrics[f"{metric_key_prefix}_{k}_accuracy"] = (
                    self._acc_correct[k] / self._acc_total[k]
                )

        # Overall token-weighted (micro) accuracy
        total_correct = sum(self._acc_correct.values())
        total_tokens = sum(self._acc_total.values())
        if total_tokens > 0:
            output.metrics[f"{metric_key_prefix}_accuracy"] = total_correct / total_tokens

        return output

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self._loss_count and "loss" in logs:
            for component in LOSS_COMPONENTS:
                if self._loss_count[component] > 0:
                    logs[component] = self._loss_sum[component] / self._loss_count[component]
            self._loss_sum.clear()
            self._loss_count.clear()

        super().log(logs, start_time)
