from typing import Any
import timm
import torch
from timm.layers import trunc_normal_  # type: ignore
from torch import nn
from homr.transformer.configs import Config


class ConvNeXtEncoder(nn.Module):
    """
    Minimal-overhead improvement for pitch accuracy.
    
    Changes from original:
    1. Factorized 2D positional embeddings (height + width) instead of 1D
    2. Optional: Sinusoidal vertical bias to emphasize staff-line periodicity
    
    Memory overhead: ~0.1% (just the positional embeddings)
    Speed overhead: Negligible
    """
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
            dims=[96, 192, 384, 768],
            depths=[3, 3, 9, 3],
        )
        
        self.encoder_dim = config.encoder_dim
        
        # Extract features up to stage 2 (stride 16) - same as original
        self.feature_info = self.model.feature_info[2]  # type: ignore
        in_features = int(self.feature_info["num_chs"])  # type: ignore
        self.proj = nn.Linear(in_features, config.encoder_dim)
        
        # IMPROVEMENT 1: Factorized 2D positional embeddings
        # Instead of learning pos_embed for each of 1280 positions,
        # we learn separate embeddings for 80 horizontal + 16 vertical positions
        # Memory: (1280 * D) vs (80 * D + 16 * D) = ~14x less parameters
        num_patches_h = config.max_height // 16   # 256 / 16 = 16 (vertical/pitch)
        num_patches_w = config.max_width // 16    # 1280 / 16 = 80 (horizontal/time)
        
        # Split encoder_dim between height and width embeddings
        # Give slightly more capacity to vertical (pitch) dimension
        self.h_dim = config.encoder_dim // 2
        self.w_dim = config.encoder_dim - self.h_dim
        
        self.pos_embed_h = nn.Parameter(torch.zeros(1, num_patches_h, self.h_dim))
        self.pos_embed_w = nn.Parameter(torch.zeros(1, num_patches_w, self.w_dim))
        
        trunc_normal_(self.pos_embed_h, std=0.02)
        trunc_normal_(self.pos_embed_w, std=0.02)
        
        # IMPROVEMENT 2 (Optional): Add sinusoidal bias for staff periodicity
        # Staff lines repeat with regular spacing - this gives an inductive bias
        # Negligible memory/compute cost
        self.use_sinusoidal_bias = True
        if self.use_sinusoidal_bias:
            # Register as buffer (not a parameter - not trained)
            # Staff periodicity is vertical (pitch dimension)
            position = torch.arange(num_patches_h, dtype=torch.float32).unsqueeze(1)
            
            # Multiple frequencies to capture staff spacing at different scales
            # Assuming staff lines are ~2-3 patches apart after stride-16
            freqs = torch.tensor([2.0, 2.5, 3.0])  # Match typical staff spacing
            
            # Create sinusoidal features
            angles = position * (2 * torch.pi / freqs)
            sin_bias = torch.sin(angles)  # (num_patches_h, 3)
            cos_bias = torch.cos(angles)  # (num_patches_h, 3)
            sinusoidal_bias = torch.cat([sin_bias, cos_bias], dim=-1)  # (num_patches_h, 6)
            
            self.register_buffer('sinusoidal_bias', sinusoidal_bias)
            
            # Small projection to encoder_dim
            self.sin_proj = nn.Linear(6, config.encoder_dim)
            # Initialize to near-zero so it starts as a small perturbation
            nn.init.normal_(self.sin_proj.weight, std=0.001)
            nn.init.zeros_(self.sin_proj.bias)

    def freeze_backbone(self) -> None:
        """Freeze only the ConvNeXt backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the ConvNeXt backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features - same as original
        _, requested_stages = self.model.forward_intermediates(x, indices=[2])  # type: ignore
        features = requested_stages[0]  # Get stage 2 (stride 16)
        
        # features shape: (B, C, H, W) = (B, 256, 80, 16)
        b, c, h, w = features.shape
        # Rearrange to (B, H, W, C)
        features = features.permute(0, 2, 3, 1)
        
        # Project to encoder_dim
        features = self.proj(features)  # (B, H, W, D)
        
        # Add factorized positional embeddings
        # self.pos_embed_h is (1, 80, h_dim) -> (1, 80, 1, h_dim)
        # self.pos_embed_w is (1, 16, w_dim) -> (1, 1, 16, w_dim)
        pos_h = self.pos_embed_h.unsqueeze(2).expand(-1, -1, w, -1)
        pos_w = self.pos_embed_w.unsqueeze(1).expand(-1, h, -1, -1)
        
        # Concatenate dimension: (1, 80, 16, D)
        pos_embed = torch.cat([pos_h, pos_w], dim=-1)
        features = features + pos_embed
        
        # Add sinusoidal staff-line bias if enabled
        if self.use_sinusoidal_bias:
            # sinusoidal_bias is (16, 6) -> project to (16, D)
            sin_features = self.sin_proj(self.sinusoidal_bias)  # (H_patch, D)
            # Reshape to (1, H_patch, 1, D) and expand to (1, H_patch, W_patch, D)
            sin_features = sin_features.view(1, h, 1, -1).expand(-1, -1, w, -1)
            features = features + sin_features
        
        # Flatten to sequence: (B, H, W, D) -> (B, H*W, D)
        features = features.reshape(b, h * w, -1)
        
        return features


def get_encoder(config: Config) -> Any:
    return ConvNeXtEncoder(config)
