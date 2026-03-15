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
    """
    ConvNeXt encoder with exact-correspondence multi-scale fusion.
    Each stage3 patch is tiled onto its 4 stage2 children via repeat_interleave,
    preserving the strict 1:4 spatial relationship the backbone computes.
    No interpolation across patch boundaries — pitch resolution is fully preserved.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            in_chans=config.channels,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.1,
        )

        stage2_chs: int = int(self.model.feature_info[2]["num_chs"])  # type: ignore
        stage3_chs: int = int(self.model.feature_info[3]["num_chs"])  # type: ignore

        # Reduce stage3 channels to match stage2 before tiling
        self.stage3_lateral = nn.Conv2d(stage3_chs, stage2_chs, kernel_size=1)

        # Gate starts near zero: stage3 must earn its contribution
        self.fpn_gate = nn.Parameter(torch.full((1,), -4.0))

        # Depthwise-separable conv to mix fused features spatially
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(stage2_chs, stage2_chs, kernel_size=3, padding=1, groups=stage2_chs),
            nn.Conv2d(stage2_chs, stage2_chs, kernel_size=1),
            nn.GroupNorm(32, stage2_chs),
            nn.GELU(),
        )

        hidden_dim = max(config.encoder_dim * 2, stage2_chs * 2)
        self.proj = nn.Sequential(
            nn.Linear(stage2_chs, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.encoder_dim),
        )

        num_patches_h = config.max_height // 16  # 256 // 16 = 16
        num_patches_w = config.max_width // 16  # 1280 // 16 = 80

        self.h_dim = config.encoder_h_dim
        self.w_dim = config.encoder_dim - self.h_dim

        self.pos_embed_h = nn.Parameter(
            _get_sinusoid_encoding(num_patches_h, self.h_dim), requires_grad=True
        )
        self.pos_embed_w = nn.Parameter(
            _get_sinusoid_encoding(num_patches_w, self.w_dim), requires_grad=True
        )

    def freeze_backbone(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, stages = self.model.forward_intermediates(x, indices=[2, 3])  # type: ignore
        feat2 = stages[0]
        feat3 = stages[1]

        feat3 = self.stage3_lateral(feat3)

        # Tile each stage3 patch onto its exact 4 stage2 children.
        # No learned interpolation — zero cross-patch contamination.
        feat3_tiled = feat3.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        feat3_tiled = feat3_tiled[:, :, : feat2.shape[2], : feat2.shape[3]]

        features = self.fuse_conv(feat2 + torch.sigmoid(self.fpn_gate) * feat3_tiled)
        b, _c, h, w = features.shape

        features = features.permute(0, 2, 3, 1)
        features = self.proj(features)

        pos_h = self.pos_embed_h[:, :h, :].unsqueeze(2).expand(-1, -1, w, -1)
        pos_w = self.pos_embed_w[:, :w, :].unsqueeze(1).expand(-1, h, -1, -1)
        pos_embed = torch.cat([pos_h, pos_w], dim=-1)

        features = features + pos_embed
        return features.reshape(b, h * w, -1)


def get_encoder(config: Config) -> Any:
    return ConvNeXtEncoder(config)
