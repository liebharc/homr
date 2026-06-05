from typing import Any

import timm
import torch
from torch import nn

from homr.transformer.configs import Config


def _get_sinusoid_encoding(n_position: int, d_hid: int) -> torch.Tensor:
    position = torch.arange(n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_hid, 2, dtype=torch.float) * -(torch.log(torch.tensor(10000.0)) / d_hid)
    )
    sinusoid_table = torch.zeros(n_position, d_hid)
    sinusoid_table[:, 0::2] = torch.sin(position * div_term[: (d_hid + 1) // 2])
    sinusoid_table[:, 1::2] = torch.cos(position * div_term[: d_hid // 2])
    return sinusoid_table.unsqueeze(0)


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
        # we learn separate embeddings for 80 horizontal + 16 vertical positions
        # Memory: (1280 * D) vs (80 * D + 16 * D) = ~14x less parameters
        num_patches_h = config.max_height // 16  # 256 / 16 = 16 (vertical/pitch)
        num_patches_w = config.max_width // 16  # 1280 / 16 = 80 (horizontal/time)

        # Split encoder_dim between height and width embeddings
        # Give slightly more capacity to vertical (pitch) dimension
        self.h_dim = config.encoder_h_dim
        self.w_dim = config.encoder_dim - self.h_dim

        self.pos_embed_h = nn.Parameter(
            _get_sinusoid_encoding(num_patches_h, self.h_dim), requires_grad=True
        )
        self.pos_embed_w = nn.Parameter(
            _get_sinusoid_encoding(num_patches_w, self.w_dim), requires_grad=True
        )

    def freeze_backbone(self) -> None:
        """Freeze only the ConvNeXt backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the ConvNeXt backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, requested_stages = self.model.forward_intermediates(x, indices=[2])  # type: ignore
        features = requested_stages[0]  # Get stage 2 (stride 16)

        # features shape: (B, C, H, W) = (B, 384, 16, 80)
        b, c, h, w = features.shape
        # Rearrange to (B, H, W, C)
        features = features.permute(0, 2, 3, 1)

        # Project to encoder_dim
        features = self.proj(features)  # (B, H, W, D)

        # Add factorized positional embeddings
        # self.pos_embed_h is (1, 80, h_dim) -> (1, 80, 1, h_dim)
        # self.pos_embed_w is (1, 16, w_dim) -> (1, 1, 16, w_dim)
        # Why this works:
        # Each position gets a full embedding by combining a vertical and horizontal vector.
        # Vertical embeddings are learned once and reused across all columns,
        # so the model efficiently learns how pitch relates to row position.
        # Horizontal embeddings handle time separately.
        # This factorization focuses capacity where it matters and reduces
        # parameters while preserving positional information.
        pos_h = self.pos_embed_h.unsqueeze(2).expand(-1, -1, w, -1)
        pos_w = self.pos_embed_w.unsqueeze(1).expand(-1, h, -1, -1)

        # Concatenate dimension: (1, 80, 16, D)
        pos_embed = torch.cat([pos_h, pos_w], dim=-1)
        features = features + pos_embed

        # Flatten to sequence: (B, H, W, D) -> (B, H*W, D)
        features = features.reshape(b, h * w, -1)

        return features


def get_encoder(config: Config) -> Any:
    return ConvNeXtEncoder(config)
