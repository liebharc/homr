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


def _make_output_conv(channels: int) -> nn.Sequential:
    """Standard FPN smoothing conv: 3×3 + GroupNorm + GELU."""
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(32, channels),
        nn.GELU(),
    )


class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt encoder with 3-stage FPN fusion (indices [1, 2, 3]).

    ConvNeXt-Tiny channel/stride map (timm feature_info indices):
      [0]  96ch  stride  4
      [1] 192ch  stride  8  ← feat1 (fine spatial detail)
      [2] 384ch  stride 16  ← feat2 (mid-range structure)
      [3] 768ch  stride 32  ← feat3 (global context: clefs, key sigs, bar structure)

    Two cascaded FPN merges, top-down:
      feat3 (stride 32) → lateral → upsample → add into feat2
      feat2 (stride 16) → lateral → upsample → add into feat1  (carries feat3 signal)
      feat1 (stride  8) → lateral → final output at stride 8

    Output: (B, 32×160, encoder_dim) for 256×1280 input.
    Transformer token count: 5120 — same as [1,2] fusion, no extra cost.
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
        stage1_chs: int = int(self.model.feature_info[1]["num_chs"])  # type: ignore  → 192
        stage2_chs: int = int(self.model.feature_info[2]["num_chs"])  # type: ignore  → 384
        stage3_chs: int = int(self.model.feature_info[3]["num_chs"])  # type: ignore  → 768
        fpn_channels: int = stage1_chs  # 192 — common FPN width throughout all levels

        # Lateral 1×1 convs — project each scale to fpn_channels
        # stage1 already 192ch but explicit projection keeps all levels symmetric
        self.lateral_stage1 = nn.Conv2d(stage1_chs, fpn_channels, kernel_size=1, bias=False)
        self.lateral_stage2 = nn.Conv2d(stage2_chs, fpn_channels, kernel_size=1, bias=False)
        self.lateral_stage3 = nn.Conv2d(stage3_chs, fpn_channels, kernel_size=1, bias=False)

        # Smoothing convs — one per merge point (after each addition)
        # output_conv_2: applied after merging feat3 into feat2 (stride 16)
        # output_conv_1: applied after merging fused-feat2 into feat1 (stride 8, final)
        self.output_conv_2 = _make_output_conv(fpn_channels)
        self.output_conv_1 = _make_output_conv(fpn_channels)

        # Final 1×1 projection: fpn_channels (192) → encoder_dim (512 or 768)
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
        _, stages = self.model.forward_intermediates(x, indices=[1, 2, 3])  # type: ignore
        feat1 = stages[0]  # (B, 192, H/8,  W/8)
        feat2 = stages[1]  # (B, 384, H/16, W/16)
        feat3 = stages[2]  # (B, 768, H/32, W/32)

        # Step 1: lateral projections — all scales → 192ch
        lat1 = self.lateral_stage1(feat1)
        lat2 = self.lateral_stage2(feat2)
        lat3 = self.lateral_stage3(feat3)

        # Step 2: top-down merge — feat3 into feat2 (stride 32 → 16)
        # lat3 carries global context (clefs, key signatures, bar-level structure)
        top_down_2 = F.interpolate(lat3, size=lat2.shape[-2:], mode="nearest")
        fused2 = self.output_conv_2(lat2 + top_down_2)  # (B, 192, H/16, W/16)

        # Step 3: top-down merge — fused feat2 into feat1 (stride 16 → 8)
        # fused2 already carries feat3 signal, so feat3 context reaches stride-8 tokens
        top_down_1 = F.interpolate(fused2, size=lat1.shape[-2:], mode="nearest")
        fused1 = self.output_conv_1(lat1 + top_down_1)  # (B, 192, H/8, W/8)

        # Step 4: project to encoder_dim
        fused = self.proj(fused1)  # (B, encoder_dim, H/8, W/8)

        b, _c, h, w = fused.shape

        # Factorized positional embeddings — separate h_dim + w_dim vectors per axis
        pos_h = self.pos_embed_h[:, :h, :].unsqueeze(2).expand(-1, -1, w, -1)
        pos_w = self.pos_embed_w[:, :w, :].unsqueeze(1).expand(-1, h, -1, -1)
        pos_embed = torch.cat([pos_h, pos_w], dim=-1)

        # (B, C, H, W) → (B, H, W, C) → add pos → (B, H*W, encoder_dim)
        fused = fused.permute(0, 2, 3, 1) + pos_embed
        return fused.reshape(b, h * w, -1)


def get_encoder(config: Config) -> Any:
    return ConvNeXtEncoder(config)