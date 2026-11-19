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


class _PadBottomConv2d(nn.Conv2d):
    """Conv2d with one zero row below the input, so that a vertical stride of 1 keeps the rows."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (0, 0, 0, 1)))


def _halve_vertical_stride_of_stage(stage: Any) -> None:
    """
    Gives the stage half the vertical stride with dilated convolutions (atrous, as in DeepLab).
    The pretrained weights stay valid: the even output rows are the original output rows.
    """
    old = stage.downsample[1]
    conv = _PadBottomConv2d(
        old.in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=(1, old.stride[1]),
        bias=old.bias is not None,
    )
    conv.load_state_dict(old.state_dict())
    stage.downsample[1] = conv
    for block in stage.blocks:
        dw = block.conv_dw
        dw.dilation = (2 * dw.dilation[0], dw.dilation[1])
        dw.padding = (2 * dw.padding[0], dw.padding[1])


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

        # Stage 2 has stride 8 vertically and 16 horizontally: a finer grid for the pitch
        _halve_vertical_stride_of_stage(self.model.stages[2])  # type: ignore

        self.encoder_dim = config.encoder_dim

        # Extract features up to stage 2
        self.feature_info = self.model.feature_info[2]  # type: ignore
        in_features = int(self.feature_info["num_chs"])  # type: ignore
        self.proj = nn.Linear(in_features, config.encoder_dim)

        # Instead of learning pos_embed for each of 2560 positions,
        # we learn separate embeddings for 80 horizontal + 32 vertical positions
        # Memory: (2560 * D) vs (80 * D + 32 * D) = ~23x less parameters
        num_patches_h = config.max_height // config.patch_height  # 32 (vertical/pitch)
        num_patches_w = config.max_width // config.patch_size  # 80 (horizontal/time)

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

    def _load_from_state_dict(
        self, state_dict: Any, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        # Checkpoints with 16 rows: every row becomes two identical rows
        key = prefix + "pos_embed_h"
        if key in state_dict and state_dict[key].shape[1] * 2 == self.pos_embed_h.shape[1]:
            state_dict[key] = state_dict[key].repeat_interleave(2, dim=1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def freeze_backbone(self) -> None:
        """Freeze only the ConvNeXt backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the ConvNeXt backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get stage 2, the stages after it aren't computed
        features = self.model.forward_intermediates(  # type: ignore
            x, indices=[2], stop_early=True, intermediates_only=True
        )[0]

        # features shape: (B, C, H, W) = (B, 384, 32, 80)
        b, c, h, w = features.shape
        # Rearrange to (B, H, W, C)
        features = features.permute(0, 2, 3, 1)

        # Project to encoder_dim
        features = self.proj(features)  # (B, H, W, D)

        # Add factorized positional embeddings
        # self.pos_embed_h is (1, 32, h_dim) -> (1, 32, 1, h_dim)
        # self.pos_embed_w is (1, 80, w_dim) -> (1, 1, 80, w_dim)
        # Why this works:
        # Each position gets a full embedding by combining a vertical and horizontal vector.
        # Vertical embeddings are learned once and reused across all columns,
        # so the model efficiently learns how pitch relates to row position.
        # Horizontal embeddings handle time separately.
        # This factorization focuses capacity where it matters and reduces
        # parameters while preserving positional information.
        pos_h = self.pos_embed_h.unsqueeze(2).expand(-1, -1, w, -1)
        pos_w = self.pos_embed_w.unsqueeze(1).expand(-1, h, -1, -1)

        # Concatenate dimension: (1, 32, 80, D)
        pos_embed = torch.cat([pos_h, pos_w], dim=-1)
        features = features + pos_embed

        # Flatten to sequence: (B, H, W, D) -> (B, H*W, D)
        features = features.reshape(b, h * w, -1)

        return features


def get_encoder(config: Config) -> Any:
    return ConvNeXtEncoder(config)
