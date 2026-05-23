# type: ignore

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler

from homr.simple_logging import eprint


class CamVidModel(pl.LightningModule):
    """
    Based on https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/examples/camvid_segmentation_multiclass.ipynb
    """

    def __init__(
        self,
        arch: str = "Unet",
        encoder_name: str = "resnet18",
        in_channels: int = 3,
        out_classes: int = 0,
        skip_weights_download: bool = False,
        **kwargs,
    ) -> None:
        """
        Build the segmentation network, normalization buffers and loss function.
        """
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=None if skip_weights_download else "imagenet",
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        """
        Normalize a batch of images and predict segmentation logits.
        """
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        """
        Run one train/validation/test step and compute segmentation statistics.
        """
        image, mask = batch

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Predict mask logits
        logits_mask = self.forward(image)

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        """
        Aggregate per-step segmentation metrics for an epoch.
        """
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        losses = torch.stack([x["loss"] for x in outputs])
        avg_loss = losses.mean()

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_loss": avg_loss,
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        eprint("Loss", avg_loss)
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        """
        Run one Lightning training step.
        """
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        """
        Log and clear accumulated training metrics at epoch end.
        """
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """
        Run one Lightning validation step.
        """
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        """
        Log and clear accumulated validation metrics at epoch end.
        """
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """
        Run one Lightning test step.
        """
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        """
        Log and clear accumulated test metrics at epoch end.
        """
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configure optimizer and cosine learning-rate scheduler for Lightning.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def create_segnet(skip_weights_download: bool = False) -> CamVidModel:
    """
    Create the HOMR segmentation model.

    Args:
        skip_weights_download: When true, initialize the encoder without ImageNet
            weights.

    Returns:
        Configured ``CamVidModel`` for six-class segmentation.
    """
    return CamVidModel(
        encoder_name="resnet18", out_classes=6, skip_weights_download=skip_weights_download
    )
