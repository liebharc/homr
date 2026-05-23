from typing import Any

import torch
from torch import nn

from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.decoder import get_decoder
from training.architecture.transformer.encoder import get_encoder


class TrOMR(nn.Module):
    """
    End-to-end transformer model for optical music recognition.

    ``TrOMR`` combines an image encoder with the multi-branch score decoder used
    during both supervised training and autoregressive generation.
    """

    def __init__(self, config: Config):
        """
        Create the encoder and decoder from a shared configuration.

        Args:
            config: Model and vocabulary configuration.
        """
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.config = config

    def eval_mode(self) -> None:
        """
        Put both encoder and decoder into evaluation mode.
        """
        self.decoder.eval()
        self.encoder.eval()

    def forward(
        self,
        inputs: torch.Tensor,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        sampling_prob: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """
        Run the training forward pass and return decoder losses.

        Args:
            inputs: Batched input image tensor.
            rhythms: Rhythm label ids, including BOS/EOS/PAD.
            pitchs: Pitch label ids aligned with ``rhythms``.
            lifts: Accidental/lift label ids aligned with ``rhythms``.
            articulations: Articulation label ids aligned with ``rhythms``.
            positions: Staff-position label ids aligned with ``rhythms``.
            mask: Boolean mask identifying non-padding positions.
            sampling_prob: Scheduled-sampling probability passed to the decoder.
            **kwargs: Additional decoder arguments.

        Returns:
            Decoder output dictionary containing total and per-branch losses.
        """
        context = self.encoder(inputs)
        loss = self.decoder(
            rhythms=rhythms,
            pitchs=pitchs,
            lifts=lifts,
            articulations=articulations,
            positions=positions,
            context=context,
            mask=mask,
            sampling_prob=sampling_prob,
            **kwargs,
        )
        return loss

    @torch.no_grad()
    def generate(self, x: torch.Tensor) -> list[EncodedSymbol]:
        """
        Generate encoded music symbols from input staff images.

        Args:
            x: Batched image tensor on the target device.

        Returns:
            Autoregressively decoded symbol sequence.
        """
        start_token = torch.tensor([[1]], dtype=torch.long, device=x.device)
        nonote_token = torch.tensor([[0]], dtype=torch.long, device=x.device)

        context = self.encoder(x)
        out = self.decoder.generate(start_token, nonote_token, context=context)

        return out

    def freeze_decoder(self) -> None:
        """Freeze all decoder parameters to prevent updates during training."""
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters to prevent updates during training."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_backbone(self) -> None:
        """Freeze only the encoder backbone."""
        if hasattr(self.encoder, "freeze_backbone"):
            self.encoder.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        """Unfreeze the encoder backbone."""
        if hasattr(self.encoder, "unfreeze_backbone"):
            self.encoder.unfreeze_backbone()

    def unfreeze_lift_decoder(self) -> None:
        """
        Unfreeze only the lift branch embedding and output projection.
        """
        for param in self.decoder.net.lift_emb.parameters():
            param.requires_grad = True
        for param in self.decoder.net.to_logits_lift.parameters():
            param.requires_grad = True


def load_model(config: Config) -> TrOMR:
    """Load model from checkpoint."""
    model = TrOMR(config)
    checkpoint_path = config.filepaths.checkpoint
    if checkpoint_path.endswith(".safetensors"):
        import safetensors  # noqa: PLC0415

        tensors = {}
        with safetensors.safe_open(checkpoint_path, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False
        )
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model
