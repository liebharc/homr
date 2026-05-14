from typing import Any

import timm
import torch
from torch import nn

from homr.transformer.configs import Config


class ConvNeXtEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        # Use configurable pretrained flag
        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            in_chans=config.channels,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.1,
        )

        self.encoder_dim = config.encoder_dim

        # Extract features up to stage 2 (stride 16)
        self.feature_info = self.model.feature_info[2]  # type: ignore
        in_features = int(self.feature_info["num_chs"])  # type: ignore
        self.proj = nn.Linear(in_features, config.encoder_dim)

        # Instead of learning pos_embed for each of 1280 positions,
        # we use fixed sinusoidal 2D positional encoding
        self.num_patches_h = config.max_height // 16  # 256 / 16 = 16 (vertical/pitch)
        self.num_patches_w = config.max_width // 16  # 1280 / 16 = 80 (horizontal/time)

    def freeze_backbone(self) -> None:
        """Freeze only the ConvNeXt backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the ConvNeXt backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def build_2d_sinusoidal_pe(self, h, w, d_model, device, dtype):
        """
        Builds a fixed 2D sinusoidal positional embedding of shape (1, h, w, d_model).
        First half of d_model dimensions encode horizontal position (time axis).
        Second half encode vertical position (pitch axis).
        Matches the formulation in the Sheet Music Transformer (Rios-Vila et al., ICDAR 2024).
        """
        assert d_model % 2 == 0, "encoder_dim must be even"
        half = d_model // 2

        def sinusoidal_1d(length, dim):
            # Returns (length, dim)
            pe = torch.zeros(length, dim, device=device, dtype=dtype)
            position = torch.arange(0, length, device=device).unsqueeze(1).float()
            div_term = torch.pow(
                10000.0, torch.arange(0, dim, 2, device=device).float() / dim
            )
            pe[:, 0::2] = torch.sin(position / div_term)
            pe[:, 1::2] = torch.cos(position / div_term)
            return pe

        pe_w = sinusoidal_1d(w, half)  # (w, half)  horizontal / time
        pe_h = sinusoidal_1d(h, half)  # (h, half)  vertical / pitch

        # Expand to (h, w, d_model) by broadcasting
        pe_w = pe_w.unsqueeze(0).expand(h, -1, -1)  # (h, w, half)
        pe_h = pe_h.unsqueeze(1).expand(-1, w, -1)  # (h, w, half)

        pe = torch.cat([pe_w, pe_h], dim=-1)  # (h, w, d_model)
        return pe.unsqueeze(0)  # (1, h, w, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, requested_stages = self.model.forward_intermediates(x, indices=[2])  # type: ignore
        features = requested_stages[0]  # Get stage 2 (stride 16)

        # features shape: (B, C, H, W) = (B, 384, 16, 80)
        b, c, h, w = features.shape
        # Rearrange to (B, H, W, C)
        features = features.permute(0, 2, 3, 1)

        # Project to encoder_dim
        features = self.proj(features)  # (B, H, W, D)

        # Add sinusoidal positional embeddings
        pos_embed = self.build_2d_sinusoidal_pe(h, w, self.encoder_dim, features.device, features.dtype)
        features = features + pos_embed

        # Flatten to sequence: (B, H, W, D) -> (B, H*W, D)
        features = features.reshape(b, h * w, -1)

        return features


def get_encoder(config: Config) -> Any:
    return ConvNeXtEncoder(config)
