from typing import Any

import timm
import torch
import torch.nn.functional as F
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
    ConvNeXt encoder with standard FPN-style top-down fusion of stage1 + stage2.

    ConvNeXt-Tiny channel/stride map (timm feature_info indices):
      [0]  96ch  stride  4
      [1] 192ch  stride  8  ← feat1 (fine spatial detail)
      [2] 384ch  stride 16  ← feat2 (semantic context)
      [3] 768ch  stride 32

    Fusion (indices=[1, 2]) produces stride-8 output → 32×160 tokens for 256×1280 input.

    Standard FPN steps:
      1. lateral 1×1 conv  — project each scale to fpn_channels (192, the finer width)
      2. F.interpolate(size=, nearest)  — upsample feat2 to feat1 spatial size
      3. element-wise addition  — inject top-down semantic context
      4. 3×3 conv + GroupNorm + GELU  — smooth aliasing from upsampling
      5. 1×1 conv  — project fpn_channels → encoder_dim
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

        # convnext_tiny: feat1 = 192ch stride 8, feat2 = 384ch stride 16
        stage1_chs: int = int(self.model.feature_info[1]["num_chs"])  # type: ignore  → 192
        stage2_chs: int = int(self.model.feature_info[2]["num_chs"])  # type: ignore  → 384
        fpn_channels: int = stage1_chs  # 192 — use finer scale width as common FPN width

        # Step 1: lateral 1×1 convs — align both scales to fpn_channels
        # stage1 is already 192ch but we still apply 1×1 for consistent learned projection
        self.lateral_stage1 = nn.Conv2d(stage1_chs, fpn_channels, kernel_size=1, bias=False)
        self.lateral_stage2 = nn.Conv2d(stage2_chs, fpn_channels, kernel_size=1, bias=False)

        # Step 4: 3×3 smoothing conv — reduces aliasing artifacts after addition
        self.output_conv = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, fpn_channels),
            nn.GELU(),
        )

        # Step 5: 1×1 projection — expand fpn_channels (192) → encoder_dim (512 or 768)
        # Equivalent to a per-token linear layer; no hidden layer needed here since
        # the transformer encoder that follows handles all channel mixing.
        self.proj = nn.Conv2d(fpn_channels, config.encoder_dim, kernel_size=1, bias=False)

        # Stride-8 output: 256//8=32 height patches, 1280//8=160 width patches
        num_patches_h = config.max_height // 8   # 32  (vertical / pitch)
        num_patches_w = config.max_width // 8    # 160 (horizontal / time)

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
        _, stages = self.model.forward_intermediates(x, indices=[1, 2])  # type: ignore
        feat1 = stages[0]  # (B, 192, H/8,  W/8)  — stride-8, fine spatial detail
        feat2 = stages[1]  # (B, 384, H/16, W/16) — stride-16, semantic context

        # Step 1: lateral projections → common channel width (192)
        lat1 = self.lateral_stage1(feat1)
        lat2 = self.lateral_stage2(feat2)

        # Step 2: upsample stride-16 → stride-8 using exact target size (robust to odd dims)
        top_down = F.interpolate(lat2, size=lat1.shape[-2:], mode="nearest")

        # Step 3: element-wise addition — merge semantic context into fine spatial features
        fused = lat1 + top_down

        # Step 4: smooth aliasing from upsampling + addition
        fused = self.output_conv(fused)  # (B, 192, H/8, W/8)

        # Step 5: project to encoder_dim via 1×1 conv (= per-token linear, no hidden layer)
        fused = self.proj(fused)  # (B, encoder_dim, H/8, W/8)

        b, _c, h, w = fused.shape

        # Positional embeddings — factorized H×W → separate h_dim + w_dim embeddings
        pos_h = self.pos_embed_h[:, :h, :].unsqueeze(2).expand(-1, -1, w, -1)
        pos_w = self.pos_embed_w[:, :w, :].unsqueeze(1).expand(-1, h, -1, -1)
        pos_embed = torch.cat([pos_h, pos_w], dim=-1)

        # (B, C, H, W) → (B, H, W, C) → add pos → (B, H*W, encoder_dim)
        fused = fused.permute(0, 2, 3, 1) + pos_embed
        return fused.reshape(b, h * w, -1)


def get_encoder(config: Config) -> Any:
    return ConvNeXtEncoder(config)