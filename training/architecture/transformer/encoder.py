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
    ConvNeXt encoder with standard FPN-style top-down fusion of stage2 + stage3.

    ConvNeXt-Tiny channel/stride map (timm feature_info indices):
      [0]  96ch  stride  4
      [1] 192ch  stride  8
      [2] 384ch  stride 16  ← feat2 (fine scale, output resolution)
      [3] 768ch  stride 32  ← feat3 (global context: clefs, key sigs, bar structure)

    Single FPN merge (indices=[2, 3]), output at stride 16.
    Token count: 16×80 = 1280 for 256×1280 input — same as the original encoder.

    Standard FPN steps:
      1. lateral 1×1 conv  — project each scale to fpn_channels (384, the finer width)
      2. F.interpolate(size=, nearest)  — upsample feat3 to feat2 spatial size
      3. element-wise addition  — inject global context into spatial features
      4. 3×3 conv + GroupNorm + GELU  — smooth aliasing from upsampling
      5. 1×1 conv  — project fpn_channels (384) → encoder_dim (512 or 768)
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

        # convnext_tiny confirmed channel sizes:
        stage2_chs: int = int(self.model.feature_info[2]["num_chs"])  # type: ignore  → 384
        stage3_chs: int = int(self.model.feature_info[3]["num_chs"])  # type: ignore  → 768
        fpn_channels: int = stage2_chs  # 384 — use finer scale width as common FPN width

        # Step 1: lateral 1×1 convs — project both scales to fpn_channels
        self.lateral_stage2 = nn.Conv2d(stage2_chs, fpn_channels, kernel_size=1, bias=False)
        self.lateral_stage3 = nn.Conv2d(stage3_chs, fpn_channels, kernel_size=1, bias=False)

        # Step 4: 3×3 smoothing conv — reduces aliasing after addition
        self.output_conv = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, fpn_channels),
            nn.GELU(),
        )

        # Step 5: 1×1 projection — fpn_channels (384) → encoder_dim (512 or 768)
        self.proj = nn.Conv2d(fpn_channels, config.encoder_dim, kernel_size=1, bias=False)

        # Stride-16 output: 256//16=16 height patches, 1280//16=80 width patches
        num_patches_h = config.max_height // 16  # 16 (vertical / pitch)
        num_patches_w = config.max_width // 16   # 80 (horizontal / time)

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
        feat2 = stages[0]  # (B, 384, H/16, W/16) — stride-16, fine spatial detail
        feat3 = stages[1]  # (B, 768, H/32, W/32) — stride-32, global context

        # Step 1: lateral projections → 384ch
        lat2 = self.lateral_stage2(feat2)
        lat3 = self.lateral_stage3(feat3)

        # Step 2: upsample stride-32 → stride-16 using exact target size
        top_down = F.interpolate(lat3, size=lat2.shape[-2:], mode="nearest")

        # Step 3: element-wise addition — merge global context into spatial features
        fused = lat2 + top_down

        # Step 4: smooth aliasing from upsampling + addition
        fused = self.output_conv(fused)  # (B, 384, H/16, W/16)

        # Step 5: project to encoder_dim
        fused = self.proj(fused)  # (B, encoder_dim, H/16, W/16)

        b, _c, h, w = fused.shape

        # Factorized positional embeddings
        pos_h = self.pos_embed_h[:, :h, :].unsqueeze(2).expand(-1, -1, w, -1)
        pos_w = self.pos_embed_w[:, :w, :].unsqueeze(1).expand(-1, h, -1, -1)
        pos_embed = torch.cat([pos_h, pos_w], dim=-1)

        # (B, C, H, W) → (B, H, W, C) → add pos → (B, H*W, encoder_dim)
        fused = fused.permute(0, 2, 3, 1) + pos_embed
        return fused.reshape(b, h * w, -1)


def get_encoder(config: Config) -> Any:
    return ConvNeXtEncoder(config)