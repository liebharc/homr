from math import ceil
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    Decoder,
    TokenEmbedding,
)

from homr.transformer.configs import Config
from homr.transformer.vocabulary import SplitSymbol, rhythm_to_category


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
        self.articulation_emb = TokenEmbedding(
            config.decoder_dim, config.num_articulation_tokens, l2norm_embed=l2norm_embed
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
        self.to_logits_articulations = nn.Linear(dim, config.num_articulation_tokens)

    def init_(self) -> None:
        if self.l2norm_embed:
            nn.init.normal_(self.lift_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pitch_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.rhythm_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.articulation_emb.emb.weight, std=1e-5)
            return

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        x = (
            self.rhythm_emb(rhythms)
            + self.pitch_emb(pitchs)
            + self.lift_emb(lifts)
            + self.articulation_emb(articulations)
            + self.pos_emb(rhythms)
        )

        x = self.post_emb_norm(x)
        x = self.project_emb(x)

        x = self.attn_layers(x, mask=mask, return_hiddens=False, **kwargs)

        out_lifts = self.to_logits_lift(x)
        out_pitchs = self.to_logits_pitch(x)
        out_rhythms = self.to_logits_rhythm(x)
        out_notes = self.to_logits_note(x)
        out_articulations = self.to_logits_articulations(x)
        return out_rhythms, out_pitchs, out_lifts, out_notes, out_articulations, x


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
        config: Config,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.pad_value = (config.pad_token,)
        self.ignore_index = ignore_index
        self.config = config
        self.net = transformer
        self.max_seq_len = config.max_seq_len
        self.eos_token = config.eos_token

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}
        self.inv_articulation_vocab = {v: k for k, v in config.articulation_vocab.items()}

        note_mask = torch.zeros(config.num_rhythm_tokens)
        for index, rhythm_symbol in enumerate(config.rhythm_vocab.keys()):
            if rhythm_to_category(rhythm_symbol) == "note":
                note_mask[index] = 1
        self.note_mask = nn.Parameter(note_mask)

        # Weight the actual lift tokens (so neither nonote nor null) higher
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        nonote_tokens: torch.Tensor,
        temperature: float = 1.0,
        filter_thres: float = 0.7,
        **kwargs: Any,
    ) -> list[SplitSymbol]:
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out_rhythm = start_tokens
        out_pitch = nonote_tokens
        out_lift = nonote_tokens
        out_articulations = nonote_tokens
        mask = kwargs.pop("mask", None)

        if mask is None:
            mask = torch.full_like(out_rhythm, True, dtype=torch.bool, device=out_rhythm.device)

        symbols: list[SplitSymbol] = []

        for _ in range(self.max_seq_len):
            mask = mask[:, -self.max_seq_len :]
            x_lift = out_lift[:, -self.max_seq_len :]
            x_pitch = out_pitch[:, -self.max_seq_len :]
            x_rhythm = out_rhythm[:, -self.max_seq_len :]
            x_articulations = out_articulations[:, -self.max_seq_len :]

            rhythmsp, pitchsp, liftsp, _notesp, articulationsp, _ = self.net(
                rhythms=x_rhythm,
                pitchs=x_pitch,
                lifts=x_lift,
                mask=mask,
                articulations=x_articulations,
                **kwargs,
            )

            filtered_lift_logits = top_k(liftsp[:, -1, :], thres=filter_thres)
            filtered_pitch_logits = top_k(pitchsp[:, -1, :], thres=filter_thres)
            filtered_rhythm_logits = top_k(rhythmsp[:, -1, :], thres=filter_thres)
            filtered_articulations_logits = top_k(articulationsp[:, -1, :], thres=filter_thres)

            current_temperature = temperature
            retry = True
            attempt = 0
            max_attempts = 5

            while retry:
                lift_probs = F.softmax(filtered_lift_logits / current_temperature, dim=-1)
                pitch_probs = F.softmax(filtered_pitch_logits / current_temperature, dim=-1)
                rhythm_probs = F.softmax(filtered_rhythm_logits / current_temperature, dim=-1)
                articulation_probs = F.softmax(
                    filtered_articulations_logits / current_temperature, dim=-1
                )

                lift_sample = torch.multinomial(lift_probs, 1)
                pitch_sample = torch.multinomial(pitch_probs, 1)
                rhythm_sample = torch.multinomial(rhythm_probs, 1)
                articulation_sample = torch.multinomial(articulation_probs, 1)

                lift_token = detokenize(lift_sample, self.inv_lift_vocab)
                pitch_token = detokenize(pitch_sample, self.inv_pitch_vocab)
                rhythm_token = detokenize(rhythm_sample, self.inv_rhythm_vocab)
                articulation_token = detokenize(articulation_sample, self.inv_articulation_vocab)

                is_eos = len(rhythm_token)
                if is_eos == 0:
                    break

                symbol = SplitSymbol(
                    rhythm=rhythm_token[0],
                    pitch=pitch_token[0],
                    lift=lift_token[0],
                    articulation=articulation_token[0],
                )

                current_temperature *= 3.5
                attempt += 1
                retry = not symbol.is_valid() and attempt < max_attempts
                if not retry:
                    symbols.append(symbol)

            out_lift = torch.cat((out_lift, lift_sample), dim=-1)
            out_pitch = torch.cat((out_pitch, pitch_sample), dim=-1)
            out_rhythm = torch.cat((out_rhythm, rhythm_sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if (
                self.eos_token is not None
                and (torch.cumsum(out_rhythm == self.eos_token, 1)[:, -1] >= 1).all()
            ):
                break

        out_lift = out_lift[:, t:]
        out_pitch = out_pitch[:, t:]
        out_rhythm = out_rhythm[:, t:]

        self.net.train(was_training)
        return symbols

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        notes: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        liftsi = lifts[:, :-1]
        liftso = lifts[:, 1:]
        articulationsi = articulations[:, :-1]
        articulationso = articulations[:, 1:]
        pitchsi = pitchs[:, :-1]
        pitchso = pitchs[:, 1:]
        rhythmsi = rhythms[:, :-1]
        rhythmso = rhythms[:, 1:]
        noteso = notes[:, 1:]

        if mask.shape[1] == rhythms.shape[1]:
            mask = mask[:, :-1]

        rhythmsp, pitchsp, liftsp, notesp, articulationsp, x = self.net(
            rhythms=rhythmsi,
            pitchs=pitchsi,
            lifts=liftsi,
            articulations=articulationsi,
            mask=mask,
            **kwargs,
        )  # this calls ScoreTransformerWrapper.forward

        loss_consist = self.calConsistencyLoss(
            rhythmsp, pitchsp, liftsp, notesp, articulationsp, mask
        )
        loss_rhythm = self.masked_logits_cross_entropy(rhythmsp, rhythmso, mask)
        loss_pitch = self.masked_logits_cross_entropy(pitchsp, pitchso, mask)
        loss_lift = self.masked_logits_cross_entropy(liftsp, liftso, mask)
        loss_articulations = self.masked_logits_cross_entropy(articulationsp, articulationso, mask)
        loss_note = self.masked_logits_cross_entropy(notesp, noteso, mask)
        # From the TR OMR paper equation 2, we use however different values for alpha and beta
        alpha = 0.1
        beta = 1
        loss_sum = loss_rhythm + loss_pitch + loss_lift + loss_note + loss_articulations
        loss = alpha * loss_sum + beta * loss_consist

        return {
            "loss_rhythm": loss_rhythm,
            "loss_pitch": loss_pitch,
            "loss_lift": loss_lift,
            "loss_consist": loss_consist,
            "loss_note": loss_note,
            "loss_articulations": loss_articulations,
            "loss": loss,
        }

    def calConsistencyLoss(
        self,
        rhythmsp: torch.Tensor,
        pitchsp: torch.Tensor,
        liftsp: torch.Tensor,
        notesp: torch.Tensor,
        articulationsp: torch.Tensor,
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

        articulationsp_soft = torch.softmax(articulationsp, dim=2)
        articulationsp_note = torch.sum(articulationsp_soft[:, :, 1:], dim=2) * mask

        loss = (
            gamma
            * (
                F.l1_loss(rhythmsp_note, note_flag, reduction="none")
                + F.l1_loss(note_flag, liftsp_note, reduction="none")
                + F.l1_loss(note_flag, pitchsp_note, reduction="none")
                + F.l1_loss(note_flag, articulationsp_note, reduction="none")
            )
            / 4.0
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
        get_score_wrapper(config),
        config=config,
    )


def get_score_wrapper(config: Config) -> ScoreTransformerWrapper:
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
