from math import ceil
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    Decoder,
    TokenEmbedding,
)

from homr.debug import AttentionDebug
from homr.results import TransformerChord
from homr.simple_logging import eprint
from homr.transformer.configs import Config
from training.architecture.transformer.split_merge_symbols import SymbolMerger


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

        if return_center_of_attention and False:
            x, hiddens = self.attn_layers(x, mask=mask, return_hiddens=True, **kwargs)
            center_of_attention = self.calculate_center_of_attention(
                debug, hiddens.attn_intermediates
            )
        else:
            x = self.attn_layers(x, mask=mask, return_hiddens=False, **kwargs)
            center_of_attention = None

        out_lifts = self.to_logits_lift(x)
        out_pitchs = self.to_logits_pitch(x)
        out_rhythms = self.to_logits_rhythm(x)
        out_notes = self.to_logits_note(x)
        return out_rhythms, out_pitchs, out_lifts, out_notes, x, center_of_attention

    def calculate_center_of_attention(
        self, debug: AttentionDebug | None, intermediates: Any
    ) -> tuple[float, float]:
        filtered_intermediate = [
            tensor.post_softmax_attn[:, :, -1, :]
            for tensor in intermediates
            if tensor.post_softmax_attn.shape[-1] == self.attention_dim
        ]

        attention_all_layers = torch.mean(torch.stack(filtered_intermediate), dim=0)
        attention_all_layers = attention_all_layers.squeeze(0).squeeze(1)
        attention_all_layers = attention_all_layers.mean(dim=0)

        image_attention = attention_all_layers[1:]
        image_attention_2d = (
            image_attention.reshape(self.attention_height, self.attention_width).cpu().numpy()
        )
        center = np.unravel_index(image_attention_2d.argmax(), image_attention_2d.shape)
        center_of_attention = (
            float(center[0] * self.patch_size),
            float(center[1] * self.patch_size),
        )

        if debug is not None:
            debug.add_attention(image_attention_2d, center_of_attention)

        return center_of_attention


def top_k(logits: torch.Tensor, thres: float = 0.9) -> torch.Tensor:
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


class ScoreDecoder(nn.Module):
    def __init__(
        self,
        transformer: ScoreTransformerWrapper,
        noteindexes: list[int],
        config: Config,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.pad_value = (config.pad_token,)
        self.ignore_index = ignore_index
        self.config = config
        self.net = transformer
        self.max_seq_len = transformer.max_seq_len

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}

        note_mask = torch.zeros(config.num_rhythm_tokens)
        note_mask[noteindexes] = 1
        self.note_mask = nn.Parameter(note_mask)

        # Weight the actual lift tokens (so neither nonote nor null) higher
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def generate(  # noqa: PLR0915
        self,
        start_tokens: torch.Tensor,
        nonote_tokens: torch.Tensor,
        seq_len: int,
        eos_token: int | None = None,
        temperature: float = 1.0,
        filter_thres: float = 0.7,
        **kwargs: Any,
    ) -> list[TransformerChord]:
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out_rhythm = start_tokens
        out_pitch = nonote_tokens
        out_lift = nonote_tokens
        mask = kwargs.pop("mask", None)
        merger = SymbolMerger()

        if mask is None:
            mask = torch.full_like(out_rhythm, True, dtype=torch.bool, device=out_rhythm.device)

        for _position_in_seq in range(seq_len):
            mask = mask[:, -self.max_seq_len :]
            x_lift = out_lift[:, -self.max_seq_len :]
            x_pitch = out_pitch[:, -self.max_seq_len :]
            x_rhythm = out_rhythm[:, -self.max_seq_len :]

            rhythmsp, pitchsp, liftsp, notesp, _ignored, center_of_attention = self.net(
                x_rhythm, x_pitch, x_lift, mask=mask, return_center_of_attention=True, **kwargs
            )

            filtered_lift_logits = top_k(liftsp[:, -1, :], thres=filter_thres)
            filtered_pitch_logits = top_k(pitchsp[:, -1, :], thres=filter_thres)
            filtered_rhythm_logits = top_k(rhythmsp[:, -1, :], thres=filter_thres)

            current_temperature = temperature
            retry = True
            attempt = 0
            max_attempts = 5

            while retry and attempt < max_attempts:
                lift_probs = F.softmax(filtered_lift_logits / current_temperature, dim=-1)
                pitch_probs = F.softmax(filtered_pitch_logits / current_temperature, dim=-1)
                rhythm_probs = F.softmax(filtered_rhythm_logits / current_temperature, dim=-1)

                lift_sample = torch.multinomial(lift_probs, 1)
                pitch_sample = torch.multinomial(pitch_probs, 1)
                rhythm_sample = torch.multinomial(rhythm_probs, 1)

                sorted_probs, sorted_indices = torch.sort(rhythm_probs, descending=True)

                rhythm_confidence = sorted_probs[0, 0].item()
                alternative_confidence = sorted_probs[0, 1].item()

                top_token_id = sorted_indices[0, 0].unsqueeze(0)
                alt_token_id = sorted_indices[0, 1].unsqueeze(0)

                rhythm_token = detokenize(top_token_id, self.inv_rhythm_vocab)
                alternative_rhythm_token = detokenize(alt_token_id, self.inv_rhythm_vocab)

                lift_token = detokenize(lift_sample, self.inv_lift_vocab)
                pitch_token = detokenize(pitch_sample, self.inv_pitch_vocab)

                is_eos = len(rhythm_token)
                if is_eos == 0:
                    break

                if len(alternative_rhythm_token) == 0:
                    alternative_rhythm_token = [""]
                    alternative_confidence = 0

                retry = merger.add_symbol_and_alternative(
                    rhythm_token[0],
                    rhythm_confidence,
                    pitch_token[0],
                    lift_token[0],
                    alternative_rhythm_token[0],
                    alternative_confidence,
                )

                current_temperature *= 3.5
                attempt += 1

            out_lift = torch.cat((out_lift, lift_sample), dim=-1)
            out_pitch = torch.cat((out_pitch, pitch_sample), dim=-1)
            out_rhythm = torch.cat((out_rhythm, rhythm_sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if (
                eos_token is not None
                and (torch.cumsum(out_rhythm == eos_token, 1)[:, -1] >= 1).all()
            ):
                break

        out_lift = out_lift[:, t:]
        out_pitch = out_pitch[:, t:]
        out_rhythm = out_rhythm[:, t:]

        self.net.train(was_training)
        return merger.complete()

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        notes: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        liftsi = lifts[:, :-1]
        liftso = lifts[:, 1:]
        pitchsi = pitchs[:, :-1]
        pitchso = pitchs[:, 1:]
        rhythmsi = rhythms[:, :-1]
        rhythmso = rhythms[:, 1:]
        noteso = notes[:, 1:]

        mask = kwargs.get("mask", None)
        if mask is not None and mask.shape[1] == rhythms.shape[1]:
            mask = mask[:, :-1]
            kwargs["mask"] = mask
        if mask is None:
            raise ValueError("A mask is required")

        rhythmsp, pitchsp, liftsp, notesp, x, _attention = self.net(
            rhythmsi, pitchsi, liftsi, **kwargs
        )  # this calls ScoreTransformerWrapper.forward

        loss_consist = self.calConsistencyLoss(rhythmsp, pitchsp, liftsp, notesp, mask)
        loss_rhythm = self.masked_logits_cross_entropy(rhythmsp, rhythmso, mask)
        loss_pitch = self.masked_logits_cross_entropy(pitchsp, pitchso, mask)
        loss_lift = self.masked_logits_cross_entropy(liftsp, liftso, mask)
        loss_note = self.masked_logits_cross_entropy(notesp, noteso, mask)
        # From the TR OMR paper equation 2, we use however different values for alpha and beta
        alpha = 0.1
        beta = 1
        loss_sum = loss_rhythm + loss_pitch + loss_lift + loss_note
        loss = alpha * loss_sum + beta * loss_consist

        return {
            "loss_rhythm": loss_rhythm,
            "loss_pitch": loss_pitch,
            "loss_lift": loss_lift,
            "loss_consist": loss_consist,
            "loss_note": loss_note,
            "loss": loss,
        }

    def calConsistencyLoss(
        self,
        rhythmsp: torch.Tensor,
        pitchsp: torch.Tensor,
        liftsp: torch.Tensor,
        notesp: torch.Tensor,
        mask: torch.Tensor,
        gamma: int = 10,
    ) -> torch.Tensor:
        notesp_soft = torch.softmax(notesp, dim=2)
        note_flag = notesp_soft[:, :, 1] * mask

        rhythmsp_soft = torch.softmax(rhythmsp, dim=2)
        rhythmsp_note = torch.sum(rhythmsp_soft * self.note_mask, dim=2) * mask

        pitchsp_soft = torch.softmax(pitchsp, dim=2)
        pitchsp_note = torch.sum(pitchsp_soft[:, :, 1:], dim=2) * mask

        liftsp_soft = torch.softmax(liftsp, dim=2)
        liftsp_note = torch.sum(liftsp_soft[:, :, 1:], dim=2) * mask

        loss = (
            gamma
            * (
                F.l1_loss(rhythmsp_note, note_flag, reduction="none")
                + F.l1_loss(note_flag, liftsp_note, reduction="none")
                + F.l1_loss(note_flag, pitchsp_note, reduction="none")
            )
            / 3.0
        )

        # Apply the mask to the loss and average over the non-masked elements
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def masked_logits_cross_entropy(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Calculate the cross-entropy loss
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            target,
            reduction="none",
            weight=weights,
            ignore_index=self.ignore_index,
        )

        # As reduction is "none", we can apply the mask to the loss
        # and this way we ignore the loss for the padded tokens
        loss = loss * mask
        loss = loss.sum() / mask.sum()

        return loss


def get_decoder(config: Config) -> ScoreDecoder:
    return ScoreDecoder(
        ScoreTransformerWrapper(
            config=config,
            attn_layers=Decoder(
                dim=config.decoder_dim,
                depth=config.decoder_depth,
                heads=config.decoder_heads,
                attn_flash=True,
                **config.decoder_args.to_dict(),
            ),
        ),
        config=config,
        noteindexes=config.noteindexes,
    )


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


def detokenize(tokens: torch.Tensor, vocab: Any) -> list[str]:
    toks = [vocab[tok.item()] for tok in tokens]
    toks = [t for t in toks if t not in ("[BOS]", "[EOS]", "[PAD]")]
    return toks


def tokenize(
    symbols: list[str], vocab: Any, default_token: int, vocab_name: str, file_name: str
) -> list[int]:

    result = []
    for symbol in symbols:
        if symbol in vocab:
            result.append(vocab[symbol])
        else:
            eprint("Warning " + file_name + ": " + symbol + " not in " + vocab_name + " vocabulary")
            result.append(default_token)
    return result
