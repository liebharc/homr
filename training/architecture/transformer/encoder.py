from typing import Any

import timm
import torch
from timm.layers import trunc_normal_  # type: ignore
from torch import nn

from homr.transformer.configs import Config


class ConvNeXtEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # Using convnext_tiny as a powerful but efficient backbone
        # Custom ConvNeXt configuration based on SMTNeXt paper
        # Dims: [64, 128, 256, 512] (we use up to 256)
        # Depths: [3, 3, 9, 3] (standard tiny depths as baseline)
        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            in_chans=config.channels,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.1,
            dims=[64, 128, 256, 512],
            depths=[3, 3, 9, 3],
        )
        # ConvNeXt stage 3 has a total stride of 16
        # Stages are: stem (stride 4), stage 1 (stride 4), stage 2 (stride 8),
        # stage 3 (stride 16), stage 4 (stride 32)
        # We want stride 16 to match the previous patch_size=16 behavior
        self.encoder_dim = config.encoder_dim

        # Extract features up to stage 2 (stride 16)
        self.feature_info = self.model.feature_info[2]  # type: ignore
        in_features = int(self.feature_info["num_chs"])  # type: ignore

        self.proj = nn.Linear(in_features, config.encoder_dim)

        # Learnable positional embeddings
        # Number of tokens = (1280/16) * (256/16) = 80 * 16 = 1280
        num_patches = (config.max_height // 16) * (config.max_width // 16)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.encoder_dim))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get stages
        # forward_intermediates returns (last_stage_tensor, [requested_stages_tensors])
        _, requested_stages = self.model.forward_intermediates(x, indices=[2])  # type: ignore
        features = requested_stages[0]  # Get stage 2 (stride 16)
        # features shape: (B, C, H/16, W/16)

        # Rearrange to sequence: (B, C, H', W') -> (B, H'*W', C)
        b, c, h, w = features.shape
        features = features.view(b, c, -1).transpose(1, 2)

        # Project to encoder_dim
        features = self.proj(features)

        # Add positional embeddings
        features = features + self.pos_embed
        return features


def get_encoder(config: Config) -> Any:
    return ConvNeXtEncoder(config)
