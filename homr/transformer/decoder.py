from math import ceil
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedTokenizerFast  # type: ignore
from x_transformers.x_transformers import (  # type: ignore
    AbsolutePositionalEmbedding,
    AttentionLayers,
    Decoder,
    TokenEmbedding,
)

from homr.transformer.configs import Config
from training.transformer.split_merge_symbols import SymbolMerger


class ScoreTransformerWrapper(nn.Module):
    def __init__(
        self,
        num_note_tokens: int,
        num_rhythm_tokens: int,
        num_pitch_tokens: int,
        num_lift_tokens: int,
        max_seq_len: int,
        attn_layers: int,
        emb_dim: int,
        l2norm_embed: bool = False,
    ):
        super().__init__()
        if not isinstance(attn_layers, AttentionLayers):
            raise ValueError("attention layers must be an instance of AttentionLayers")

        dim = attn_layers.dim
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.lift_emb = TokenEmbedding(emb_dim, num_lift_tokens, l2norm_embed=l2norm_embed)
        self.pitch_emb = TokenEmbedding(emb_dim, num_pitch_tokens, l2norm_embed=l2norm_embed)
        self.rhythm_emb = TokenEmbedding(emb_dim, num_rhythm_tokens, l2norm_embed=l2norm_embed)
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed=l2norm_embed)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.init_()

        self.to_logits_lift = nn.Linear(dim, num_lift_tokens)
        self.to_logits_pitch = nn.Linear(dim, num_pitch_tokens)
        self.to_logits_rhythm = nn.Linear(dim, num_rhythm_tokens)
        self.to_logits_note = nn.Linear(dim, num_note_tokens)

    def init_(self) -> None:
        if self.l2norm_embed:
            nn.init.normal_(self.lift_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pitch_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.rhythm_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
            return

        nn.init.kaiming_normal_(self.lift_emb.emb.weight)
        nn.init.kaiming_normal_(self.pitch_emb.emb.weight)
        nn.init.kaiming_normal_(self.rhythm_emb.emb.weight)

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_hiddens: bool = True,
        **kwargs: Any,
    ) -> Any:
        x = (
            self.rhythm_emb(rhythms)
            + self.pitch_emb(pitchs)
            + self.lift_emb(lifts)
            + self.pos_emb(rhythms)
        )
        x = self.project_emb(x)
        x, hiddens = self.attn_layers(x, mask=mask, return_hiddens=return_hiddens, **kwargs)

        x = self.norm(x)

        out_lifts = self.to_logits_lift(x)
        out_pitchs = self.to_logits_pitch(x)
        out_rhythms = self.to_logits_rhythm(x)
        out_notes = self.to_logits_note(x)
        return out_rhythms, out_pitchs, out_lifts, out_notes, x


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

        self.lifttokenizer = PreTrainedTokenizerFast(tokenizer_file=config.filepaths.lifttokenizer)
        self.pitchtokenizer = PreTrainedTokenizerFast(
            tokenizer_file=config.filepaths.pitchtokenizer
        )
        self.rhythmtokenizer = PreTrainedTokenizerFast(
            tokenizer_file=config.filepaths.rhythmtokenizer
        )

        self.net = transformer
        self.max_seq_len = transformer.max_seq_len

        note_mask = torch.zeros(config.num_rhythm_tokens)
        note_mask[noteindexes] = 1
        self.note_mask = nn.Parameter(note_mask)

        # Weight the actual lift tokens (so neither nonote nor null) higher
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        nonote_tokens: torch.Tensor,
        seq_len: int,
        eos_token: int | None = None,
        temperature: float = 1.0,
        filter_thres: float = 0.9,
        **kwargs: Any,
    ) -> list[str]:
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

        for _ in range(seq_len):
            mask = mask[:, -self.max_seq_len :]
            x_lift = out_lift[:, -self.max_seq_len :]
            x_pitch = out_pitch[:, -self.max_seq_len :]
            x_rhythm = out_rhythm[:, -self.max_seq_len :]

            rhythmsp, pitchsp, liftsp, notesp, _ = self.net(
                x_rhythm, x_pitch, x_lift, mask=mask, **kwargs
            )

            filtered_lift_logits = top_k(liftsp[:, -1, :], thres=filter_thres)
            filtered_pitch_logits = top_k(pitchsp[:, -1, :], thres=filter_thres)
            filtered_rhythm_logits = top_k(rhythmsp[:, -1, :], thres=filter_thres)

            lift_probs = F.softmax(filtered_lift_logits / temperature, dim=-1)
            pitch_probs = F.softmax(filtered_pitch_logits / temperature, dim=-1)
            rhythm_probs = F.softmax(filtered_rhythm_logits / temperature, dim=-1)

            lift_sample = torch.multinomial(lift_probs, 1)
            pitch_sample = torch.multinomial(pitch_probs, 1)
            rhythm_sample = torch.multinomial(rhythm_probs, 1)

            out_lift = torch.cat((out_lift, lift_sample), dim=-1)
            out_pitch = torch.cat((out_pitch, pitch_sample), dim=-1)
            out_rhythm = torch.cat((out_rhythm, rhythm_sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if (
                eos_token is not None
                and (torch.cumsum(out_rhythm == eos_token, 1)[:, -1] >= 1).all()
            ):
                break

            lift_token = detokenize(lift_sample, self.lifttokenizer)
            pitch_token = detokenize(pitch_sample, self.pitchtokenizer)
            rhythm_token = detokenize(rhythm_sample, self.rhythmtokenizer)
            merger.add_symbol(rhythm_token[0][0], pitch_token[0][0], lift_token[0][0])

        out_lift = out_lift[:, t:]
        out_pitch = out_pitch[:, t:]
        out_rhythm = out_rhythm[:, t:]

        self.net.train(was_training)
        return [merger.complete()]

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

        rhythmsp, pitchsp, liftsp, notesp, x = self.net(
            rhythmsi, pitchsi, liftsi, **kwargs
        )  # this calls ScoreTransformerWrapper.forward

        loss_consist = self.calConsistencyLoss(rhythmsp, pitchsp, liftsp, notesp, mask)
        loss_rhythm = self.masked_logits_cross_entropy(rhythmsp, rhythmso, mask)
        loss_pitch = self.masked_logits_cross_entropy(pitchsp, pitchso, mask)
        loss_lift = self.masked_logits_cross_entropy(liftsp, liftso, mask)
        loss_note = self.masked_logits_cross_entropy(notesp, noteso, mask)
        # From the TR OMR paper equation 2, we use however different values for alpha and beta
        alpha = 0.2
        beta = 1 - alpha
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
            num_note_tokens=config.num_note_tokens,
            num_rhythm_tokens=config.num_rhythm_tokens,
            num_pitch_tokens=config.num_pitch_tokens,
            num_lift_tokens=config.num_lift_tokens,
            max_seq_len=config.max_seq_len,
            emb_dim=config.decoder_dim,
            attn_layers=Decoder(
                dim=config.decoder_dim,
                depth=config.decoder_depth,
                heads=config.decoder_heads,
                **config.decoder_args.to_dict(),
            ),
        ),
        config=config,
        noteindexes=config.noteindexes,
    )


def detokenize(tokens: torch.Tensor, tokenizer: Any) -> list[list[str]]:
    toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
    for b in range(len(toks)):
        for i in reversed(range(len(toks[b]))):
            if toks[b][i] is None:
                toks[b][i] = ""
            toks[b][i] = toks[b][i].replace("Ġ", " ").strip()
            if toks[b][i] in (["[BOS]", "[EOS]", "[PAD]"]):
                del toks[b][i]
    return toks


def tokenize(symbols: list[str], vocab: Any, default_token: int, vocab_name: str) -> list[int]:

    result = []
    for symbol in symbols:
        if symbol in vocab:
            result.append(vocab[symbol])
        else:
            print("Warning: " + symbol + " not in " + vocab_name + " vocabulary")
            result.append(default_token)
    return result
