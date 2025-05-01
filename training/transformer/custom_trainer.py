# type: ignore
# ruff: noqa: T201

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class UnfreezeBackboneCallback(TrainerCallback):
    def __init__(self, unfreeze_epoch=3):
        self.unfreeze_epoch = unfreeze_epoch
        self.unfrozen = False

    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        if state.epoch is not None and state.epoch >= self.unfreeze_epoch and not self.unfrozen:
            model = kwargs["model"]
            for param in model.encoder.patch_embed.backbone.parameters():
                param.requires_grad = True
            print(f"🔓 Unfroze backbone at epoch {int(state.epoch)}")
            self.unfrozen = True
        return control

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        model = kwargs["model"]
        for param in model.encoder.patch_embed.backbone.parameters():
            param.requires_grad = False
        print("🧊 Backbone frozen at start of training")
        return control
