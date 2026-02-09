from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol, has_rhythm_symbol_a_position
from training.architecture.transformer.custom_x_transformer import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    CustomDecoder,
    Intermediates,
    LayerIntermediates,
    TokenEmbedding,
)


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
        # The transformer operates on rotated images: height and width are swapped
        self.attention_dim = config.max_width * config.max_height // config.patch_size**2 + 1
        self.attention_width = config.max_height // config.patch_size
        self.attention_height = config.max_width // config.patch_size
        self.patch_size = config.patch_size

        self.attn_layers = attn_layers
        self.post_emb_norm = nn.LayerNorm(dim)

        self.to_logits_lift = nn.Linear(dim, config.num_lift_tokens)
        self.to_logits_pitch = nn.Linear(dim, config.num_pitch_tokens)
        self.to_logits_rhythm = nn.Linear(dim, config.num_rhythm_tokens)
        self.to_logits_position = nn.Linear(dim, config.num_position_tokens)
        self.to_logits_articulations = nn.Linear(dim, config.num_articulation_tokens)

        self.init_()

    def init_(self) -> None:
        if self.l2norm_embed:
            nn.init.normal_(self.lift_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pitch_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.rhythm_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.articulation_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
        else:
            # Use transformer standard initialization (std=0.02)
            # This provides stronger gradients than 1e-5 for faster convergence
            nn.init.normal_(self.lift_emb.emb.weight, std=0.02)
            nn.init.normal_(self.pitch_emb.emb.weight, std=0.02)
            nn.init.normal_(self.rhythm_emb.emb.weight, std=0.02)
            nn.init.normal_(self.articulation_emb.emb.weight, std=0.02)
            nn.init.normal_(self.pos_emb.emb.weight, std=0.02)

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        context: torch.Tensor | None = None,
        cache_len: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_center_of_attention: bool = False,
        **kwargs: torch.Tensor,
    ) -> Any:
        cache = kwargs.pop("cache", None)
        if cache is None:
            x = (
                self.rhythm_emb(rhythms)
                + self.pitch_emb(pitchs)
                + self.lift_emb(lifts)
                + self.articulation_emb(articulations)
                + self.pos_emb(rhythms)
            )

            x = self.post_emb_norm(x)

            if return_center_of_attention:
                x, hiddens = self.attn_layers(x, mask=mask, return_hiddens=True, **kwargs)
                attention = self.get_center_of_attention(hiddens.attn_intermediates)
            else:
                x = self.attn_layers(x, mask=mask, return_hiddens=False, context=context, **kwargs)
                attention = None

            out_lifts = self.to_logits_lift(x)
            out_pitchs = self.to_logits_pitch(x)
            out_rhythms = self.to_logits_rhythm(x)
            out_articulations = self.to_logits_articulations(x)
            out_positions = self.to_logits_position(x)
            return (
                out_rhythms,
                out_pitchs,
                out_lifts,
                out_positions,
                out_articulations,
                x,
                attention,
                None,
            )

        else:
            x = (
                self.rhythm_emb(rhythms)
                + self.pitch_emb(pitchs)
                + self.lift_emb(lifts)
                + self.articulation_emb(articulations)
                + self.pos_emb(rhythms, offset=cache_len)
            )

            x = self.post_emb_norm(x)

            # reconstruct x_transformers LayerIntermediates from the input_cache
            inters = []
            for i in range(0, 32, 2):
                inters.append(Intermediates(cached_kv=(cache[i], cache[i + 1])))

            cache_input = LayerIntermediates(attn_intermediates=inters, cache_length=cache_len)

            x, cache = self.attn_layers(
                x, cache=cache_input, mask=mask, return_hiddens=True, context=context
            )

            # get the kv cache tensors from the LayerIntermediates class
            # the cache is built up like this:
            # LayerIntermediates(atten_intermediates=Intermediates(cached_kv=(cache_k, cache_v)))
            # cache is alternating between shapes (batch, 8, seq_len, 64) and (batch, 8, 1281, 64)
            # 8 probably corresponds to the number of decoder_heads
            # 1281 is the same as the encoder output
            cache_out = []
            attn_inters = cache.attn_intermediates
            if return_center_of_attention:
                attention = self.get_center_of_attention(attn_inters)
            else:
                attention = None
            for i in range(16):  # 16x2
                k, v = attn_inters[i].cached_kv
                cache_out.append(k)
                cache_out.append(v)

            out_lifts = self.to_logits_lift(x)
            out_pitchs = self.to_logits_pitch(x)
            out_rhythms = self.to_logits_rhythm(x)
            out_articulations = self.to_logits_articulations(x)
            out_positions = self.to_logits_position(x)
            return (
                out_rhythms,
                out_pitchs,
                out_lifts,
                out_positions,
                out_articulations,
                x,
                attention,
                cache_out,
            )

    def get_center_of_attention(self, intermediates: list[Any]) -> torch.Tensor:
        """
        Calculates the center of attention. It uses the attention weights,
        performs a power scaling to give more weight to the peaks and then
        calculates the center of mass to get the focus point of the attention.
        """

        # Only use the last 3 layers as the later layers contain the strongest
        # alignment between attention and semantic object position.
        filtered_intermediate = [
            intermediates[-5].post_softmax_attn[:, :, -1, :],
            intermediates[-3].post_softmax_attn[:, :, -1, :],
            intermediates[-1].post_softmax_attn[:, :, -1, :],
        ]

        attention_all_layers = torch.mean(torch.stack(filtered_intermediate), dim=0)
        attention_all_layers = attention_all_layers.squeeze(0).squeeze(1)
        attention_all_layers = attention_all_layers.mean(dim=0)
        h, w = self.attention_height, self.attention_width

        image_token_count = h * w
        image_attention = attention_all_layers[0:image_token_count]

        image_attention_2d = image_attention.reshape(h, w)

        power = 4.0
        weights = torch.clamp(image_attention_2d, min=1e-4).pow(power)

        y_coords = torch.linspace(0.5, h - 0.5, h, device=weights.device, dtype=weights.dtype)
        x_coords = torch.linspace(0.5, w - 0.5, w, device=weights.device, dtype=weights.dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        total_mass = weights.sum()
        row = (weights * yy).sum() / total_mass
        col = (weights * xx).sum() / total_mass

        center_of_attention = torch.stack(
            [
                col * self.patch_size,
                row * self.patch_size,
            ]
        )

        return center_of_attention


class ScoreDecoder(nn.Module):
    def __init__(self, transformer: ScoreTransformerWrapper, config: Config):
        super().__init__()
        self.pad_value = (config.pad_token,)
        self.ignore_index = config.pad_token
        self.config = config
        self.net = transformer
        self.max_seq_len = config.max_seq_len
        self.eos_token = config.eos_token

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}
        self.inv_articulation_vocab = {v: k for k, v in config.articulation_vocab.items()}
        self.inv_position_vocab = {v: k for k, v in config.position_vocab.items()}

        note_mask = torch.zeros(config.num_rhythm_tokens)
        for index, rhythm_symbol in enumerate(config.rhythm_vocab.keys()):
            if has_rhythm_symbol_a_position(rhythm_symbol):
                note_mask[index] = 1
        self.note_mask = nn.Parameter(note_mask)

        # Weight the actual lift tokens (so neither nonote nor null) higher
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        nonote_tokens: torch.Tensor,
        **kwargs: Any,
    ) -> list[EncodedSymbol]:
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
        context_first = kwargs.pop("context")
        context_later = context_first[:, :0]

        if mask is None:
            # the mask is always (True, True) because the x_ are always (1, 1)
            # and contain only the last token
            # the information about the rest of the tokens gets passed via the kv cache
            mask = torch.ones((1, 1), dtype=torch.bool, device=self.device)

        symbols: list[EncodedSymbol] = []

        cache = init_cache(0, self.device)[0]

        for step in range(self.max_seq_len):
            x_lift = out_lift[:, -1:]
            x_pitch = out_pitch[:, -1:]
            x_rhythm = out_rhythm[:, -1:]
            x_articulations = out_articulations[:, -1:]

            if step == 0:
                context = context_first
            else:
                context = context_later

            rhythmsp, pitchsp, liftsp, positionsp, articulationsp, _, _, cache = self.net(
                rhythms=x_rhythm,
                pitchs=x_pitch,
                lifts=x_lift,
                articulations=x_articulations,
                context=context,
                cache_len=torch.Tensor([step]).to(self.device).long(),
                mask=mask,
                cache=cache,
                return_center_of_attention=False,
                **kwargs,
            )

            # Greedy decoding: pick the highest logit directly for each output
            rhythm_sample = rhythmsp[:, -1, :].argmax(dim=-1, keepdim=True)
            pitch_sample = pitchsp[:, -1, :].argmax(dim=-1, keepdim=True)
            lift_sample = liftsp[:, -1, :].argmax(dim=-1, keepdim=True)
            articulation_sample = articulationsp[:, -1, :].argmax(dim=-1, keepdim=True)
            position_sample = positionsp[:, -1, :].argmax(dim=-1, keepdim=True)

            lift_token = detokenize(lift_sample, self.inv_lift_vocab)
            pitch_token = detokenize(pitch_sample, self.inv_pitch_vocab)
            rhythm_token = detokenize(rhythm_sample, self.inv_rhythm_vocab)
            articulation_token = detokenize(articulation_sample, self.inv_articulation_vocab)
            position_token = detokenize(position_sample, self.inv_position_vocab)

            if rhythm_sample[0][0] == self.eos_token:
                break

            symbol = EncodedSymbol(
                rhythm=rhythm_token[0],
                pitch=pitch_token[0],
                lift=lift_token[0],
                articulation=articulation_token[0],
                position=position_token[0],
            )
            symbols.append(symbol)

            out_lift = torch.cat((out_lift, lift_sample), dim=-1)
            out_pitch = torch.cat((out_pitch, pitch_sample), dim=-1)
            out_rhythm = torch.cat((out_rhythm, rhythm_sample), dim=-1)
            out_articulations = torch.cat((out_articulations, articulation_sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

        self.net.train(was_training)
        return symbols

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        liftsi = lifts[:, :-1]
        liftso = lifts[:, 1:]
        articulationsi = articulations[:, :-1]
        articulationso = articulations[:, 1:]
        pitchsi = pitchs[:, :-1]
        pitchso = pitchs[:, 1:]
        rhythmsi = rhythms[:, :-1]
        rhythmso = rhythms[:, 1:]
        positionso = positions[:, 1:]

        if mask.shape[1] == rhythms.shape[1]:
            mask = mask[:, :-1]

        rhythmsp, pitchsp, liftsp, positionsp, articulationsp, x, _attention, _cache = self.net(
            rhythms=rhythmsi,
            pitchs=pitchsi,
            lifts=liftsi,
            articulations=articulationsi,
            mask=mask,
            cache=None,
            return_center_of_attention=False,
            **kwargs,
        )  # this calls ScoreTransformerWrapper.forward

        # From the TR OMR paper equation 2, we use however different values for alpha and beta
        alpha = 0.3125
        beta = 1
        loss_consist = beta * self.calConsistencyLoss(
            rhythmsp, pitchsp, liftsp, positionsp, articulationsp, mask
        )
        loss_rhythm = alpha * self.cross_entropy(rhythmsp, rhythmso, label_smoothing=0.1)
        loss_pitch = alpha * self.cross_entropy(pitchsp, pitchso)
        loss_lift = alpha * self.cross_entropy(liftsp, liftso)
        loss_articulations = alpha * self.cross_entropy(articulationsp, articulationso)
        loss_position = alpha * self.cross_entropy(positionsp, positionso)
        loss = (
            loss_rhythm + loss_pitch + loss_lift + loss_articulations + loss_position + loss_consist
        )

        return {
            "loss_rhythm": loss_rhythm,
            "loss_pitch": loss_pitch,
            "loss_lift": loss_lift,
            "loss_consist": loss_consist,
            "loss_position": loss_position,
            "loss_articulations": loss_articulations,
            "loss": loss,
            "logits": (rhythmsp, pitchsp, liftsp, positionsp, articulationsp),
        }

    def calConsistencyLoss(
        self,
        rhythmsp: torch.Tensor,
        pitchsp: torch.Tensor,
        liftsp: torch.Tensor,
        positionsp: torch.Tensor,
        articulationsp: torch.Tensor,
        mask: torch.Tensor,
        gamma: int = 10,
    ) -> torch.Tensor:
        positionsp_soft = torch.softmax(positionsp, dim=2)
        positionsp_note = torch.sum(positionsp_soft[:, :, 1:], dim=2) * mask

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
                F.l1_loss(rhythmsp_note, positionsp_note, reduction="none")
                + F.l1_loss(positionsp_note, liftsp_note, reduction="none")
                + F.l1_loss(positionsp_note, pitchsp_note, reduction="none")
                + F.l1_loss(positionsp_note, articulationsp_note, reduction="none")
            )
            / 4.0
        )

        # Apply the mask to the loss and average over the non-masked elements
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def cross_entropy(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        label_smoothing: float = 0.0,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits.transpose(1, 2),
            target,
            reduction="mean",
            weight=weights,
            ignore_index=self.ignore_index,
            label_smoothing=label_smoothing,
        )


def init_cache(
    cache_len: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[str], list[str], dict[str, dict[int, str]], int]:
    cache = []
    input_names = []
    output_names = []
    dynamic = {}
    for i in range(32):
        cache.append(torch.zeros((1, 8, cache_len, 64), dtype=torch.float32).to(device))
        input_names.append(f"cache_in{i}")
        output_names.append(f"cache_out{i}")
        dynamic[f"cache_in{i}"] = {2: "seq_len"}
    return cache, input_names, output_names, dynamic, cache_len


def get_decoder(config: Config) -> ScoreDecoder:
    return ScoreDecoder(
        get_score_wrapper(config),
        config=config,
    )


def get_score_wrapper(config: Config, attn_flash: bool = True) -> ScoreTransformerWrapper:
    return ScoreTransformerWrapper(
        config=config,
        attn_layers=CustomDecoder(
            dim=config.decoder_dim,
            depth=config.decoder_depth,
            heads=config.decoder_heads,
            attn_flash=attn_flash,
            **config.decoder_args.to_dict(),
        ),
    )


def detokenize(tokens: torch.Tensor, vocab: Any) -> list[str]:
    toks = [vocab[tok.item()] for tok in tokens]
    toks = [t for t in toks if t not in ("[BOS]", "[EOS]", "[PAD]")]
    return toks
