from math import ceil
from typing import Any

import torch
from torch import nn
from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    Decoder,
    TokenEmbedding,
)

from training.convert_onnx.transformer.configs import Config


class ScoreTransformerWrapper(nn.Module):
    """
    Based on x_transformers.TransformerWrapper to support multiple embeddings.
    """

    def __init__(
        self,
        config: Config,
        attn_layers: Any,
        l2norm_embed: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(attn_layers, AttentionLayers):
            raise ValueError("attention layers must be an instance of AttentionLayers")

        dim = attn_layers.dim
        self.max_seq_len = config.max_seq_len
        self.l2norm_embed = l2norm_embed
        self.lift_emb = TokenEmbedding(
            config.decoder_dim, config.num_lift_tokens, l2norm_embed=l2norm_embed
        )
        self.pitch_emb = TokenEmbedding(
            config.decoder_dim, config.num_pitch_tokens, l2norm_embed=l2norm_embed
        )
        self.rhythm_emb = TokenEmbedding(
            config.decoder_dim, config.num_rhythm_tokens, l2norm_embed=l2norm_embed
        )
        self.pos_emb = AbsolutePositionalEmbedding(
            config.decoder_dim, config.max_seq_len, l2norm_embed=l2norm_embed
        )
        self.attention_dim = config.max_width * config.max_height // config.patch_size**2 + 1
        self.attention_width = config.max_width // config.patch_size
        self.attention_height = config.max_height // config.patch_size
        self.patch_size = config.patch_size

        self.project_emb = (
            nn.Linear(config.decoder_dim, dim) if config.decoder_dim != dim else nn.Identity()
        )
        self.attn_layers = attn_layers
        self.post_emb_norm = nn.LayerNorm(dim)
        self.init_()

        self.to_logits_lift = nn.Linear(dim, config.num_lift_tokens)
        self.to_logits_pitch = nn.Linear(dim, config.num_pitch_tokens)
        self.to_logits_rhythm = nn.Linear(dim, config.num_rhythm_tokens)
        self.to_logits_note = nn.Linear(dim, config.num_note_tokens)

    def init_(self) -> None:
        if self.l2norm_embed:
            nn.init.normal_(self.lift_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pitch_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.rhythm_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
            return

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_center_of_attention: bool = False,
        **kwargs: Any,
    ) -> Any:
        x = (
            self.rhythm_emb(rhythms)
            + self.pitch_emb(pitchs)
            + self.lift_emb(lifts)
            + self.pos_emb(rhythms)
        )

        x = self.post_emb_norm(x)
        x = self.project_emb(x)
        debug = kwargs.pop("debug", None)

        x = self.attn_layers(x, mask=mask, return_hiddens=False, **kwargs)
        center_of_attention = None

        out_lifts = self.to_logits_lift(x)
        out_pitchs = self.to_logits_pitch(x)
        out_rhythms = self.to_logits_rhythm(x)
        out_notes = self.to_logits_note(x)
        return out_rhythms, out_pitchs, out_lifts, out_notes, x, center_of_attention


def top_k(logits: torch.Tensor, thres: float = 0.9) -> torch.Tensor:
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


def get_decoder_onnx(config: Config):
    return ScoreTransformerWrapper(
            config=config,
            attn_layers=Decoder(
                dim=config.decoder_dim,
                depth=config.decoder_depth,
                heads=config.decoder_heads,
                attn_flash=True,
                **config.decoder_args.to_dict(),
            ),
        )
