import torch
from timm.models.vision_transformer_hybrid import HybridEmbed
from torch import nn


class PositionalEncoding2D(nn.Module):

    def __init__(self, dim, h_max, w_max):
        super().__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim
        self.pe = torch.zeros(
            (1, dim, h_max, w_max),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            requires_grad=False,
        )

        div = torch.exp(
            -torch.arange(0.0, dim // 2, 2) / dim * torch.log(torch.tensor(10000.0))
        ).unsqueeze(1)
        w_pos = torch.arange(0.0, w_max)
        h_pos = torch.arange(0.0, h_max)
        self.pe[:, : dim // 2 : 2, :, :] = (
            torch.sin(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        )
        self.pe[:, 1 : dim // 2 : 2, :, :] = (
            torch.cos(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        )
        self.pe[:, dim // 2 :: 2, :, :] = (
            torch.sin(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        )
        self.pe[:, dim // 2 + 1 :: 2, :, :] = (
            torch.cos(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        )

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: (B, C, H, W)
        """
        return x + self.pe[:, :, : x.size(2), : x.size(3)]

    def get_pe_by_size(self, h, w, device):
        return self.pe[:, :, :h, :w].to(device)


class HybridEmbedWith2DPos(HybridEmbed):
    def __init__(self, *args, **kwargs):
        max_height = kwargs.pop("max_height")
        max_width = kwargs.pop("max_width")
        super().__init__(*args, **kwargs)
        # Assuming the output size is known; otherwise, compute dynamically
        self.pos_encoding = PositionalEncoding2D(
            dim=self.proj.out_channels,
            h_max=max_height,
            w_max=max_width,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)
        x = self.pos_encoding(x)
        x = x.flatten(2).transpose(1, 2)  # Flatten and transpose for transformer
        return x
