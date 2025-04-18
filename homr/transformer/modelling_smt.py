import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_
from transformers import ConvNextConfig, ConvNextModel, PreTrainedModel  # type: ignore
from transformers.modeling_outputs import (  # type: ignore
    CausalLMOutputWithCrossAttentions,
)

from homr.transformer.configs import Config


class PositionalEncoding2D(nn.Module):

    def __init__(self, dim: int, h_max: int, w_max: int) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add 2D positional encoding to x
        x: (B, C, H, W)
        """
        return x + self.pe[:, :, : x.size(2), : x.size(3)]

    def get_pe_by_size(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        return self.pe[:, :, :h, :w].to(device)


class PositionalEncoding1D(nn.Module):

    def __init__(self, dim: int, len_max: int) -> None:
        super().__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.zeros(
            (1, dim, len_max),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            requires_grad=False,
        )

        div = torch.exp(
            -torch.arange(0.0, dim, 2) / dim * torch.log(torch.tensor(10000.0))
        ).unsqueeze(1)
        l_pos = torch.arange(0.0, len_max)
        self.pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0)
        self.pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0)

    def forward(self, x: torch.Tensor, start: int) -> torch.Tensor:
        """
        Add 1D positional encoding to x
        x: (B, C, L)
        start: index for x[:,:, 0]
        """
        if isinstance(start, int):
            return x + self.pe[:, :, start : start + x.size(2)].to(x.device)
        else:
            for i in range(x.size(0)):
                x[i] = x[i] + self.pe[0, :, start[i] : start[i] + x.size(2)]
            return x


class MHA(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int | None = None,
        dropout: float = 0,
        proj_value: bool = True,
    ) -> None:
        super().__init__()

        self.proj_value = proj_value
        self.lq = nn.Linear(embedding_dim, embedding_dim)
        self.lk = nn.Linear(embedding_dim, embedding_dim)
        if proj_value:
            self.lv = nn.Linear(embedding_dim, embedding_dim)

        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale_factor = float(self.head_dim) ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_pad_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        get_weights: bool = True,
    ) -> torch.Tensor:

        target_len, b, c = query.size()
        source_len = key.size(0)

        q = self.lq(query)
        k = self.lk(key)
        v = self.lv(value) if self.proj_value else value

        q = torch.reshape(q, (target_len, b * self.num_heads, self.head_dim)).transpose(0, 1)
        k = torch.reshape(k, (source_len, b * self.num_heads, self.head_dim)).transpose(0, 1)
        v = torch.reshape(v, (source_len, b * self.num_heads, self.head_dim)).transpose(0, 1)

        attn_output_weigths = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                attn_output_weigths.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weigths += attn_mask

        if key_pad_mask is not None:
            attn_output_weigths = attn_output_weigths.view(
                b, self.num_heads, target_len, source_len
            )
            attn_output_weigths = attn_output_weigths.masked_fill(
                key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_output_weigths = attn_output_weigths.view(
                b * self.num_heads, target_len, source_len
            )

        attn_output_weigths_raw = self.softmax(attn_output_weigths)
        attn_output_weigths = self.dropout(attn_output_weigths_raw)
        attn_output = torch.bmm(attn_output_weigths, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(target_len, b, c)
        attn_output = self.out_proj(attn_output)

        if get_weights:
            attn_output_weigths_raw = attn_output_weigths_raw.view(
                b, self.num_heads, target_len, source_len
            )
            return attn_output, attn_output_weigths_raw.sum(dim=1) / self.num_heads

        return attn_output

    def init_weights(self):
        xavier_uniform_(self.in_proj_q.weight)
        xavier_uniform_(self.in_proj_k.weight)
        if self.proj_value:
            xavier_uniform_(self.in_proj_v.weight)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, dim_ff) -> None:
        super().__init__()
        self.d_model = d_model
        self.ff = dim_ff

        self.input_attention = MHA(
            embedding_dim=self.d_model, num_heads=4, proj_value=True, dropout=0.1
        )

        self.norm1 = nn.LayerNorm(self.d_model)

        self.cross_attention = MHA(
            embedding_dim=self.d_model, num_heads=4, proj_value=True, dropout=0.1
        )

        self.ffNet = nn.Sequential(
            nn.Linear(self.d_model, self.ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.ff, self.d_model),
        )

        self.dropout = nn.Dropout(0.1)

        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)

    def set_lm_mode(self):
        for parameter in self.cross_attention.parameters():
            parameter.requires_grad = False

        for parameter in self.norm2.parameters():
            parameter.requires_grad = False

    def set_transcription_mode(self):
        for parameter in self.cross_attention.parameters():
            parameter.requires_grad = True

        for parameter in self.norm2.parameters():
            parameter.requires_grad = True

    def forward(
        self,
        tgt,
        memory_key,
        memory_value=None,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        predict_n_last_only=None,
    ):

        if memory_value is None:
            memory_value = memory_key

        mha_q = tgt[-predict_n_last_only:] if predict_n_last_only else tgt

        tgt2, weights_input = self.input_attention(
            mha_q, tgt, tgt, attn_mask=tgt_mask, key_pad_mask=tgt_key_padding_mask, get_weights=True
        )
        tgt = mha_q + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        att_query = tgt

        tgt2, weights_cross = self.cross_attention(
            att_query,
            memory_key,
            memory_value,
            attn_mask=memory_mask,
            key_pad_mask=memory_key_padding_mask,
            get_weights=True,
        )

        tgt = att_query + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ffNet(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt, weights_input, weights_cross


class DecoderStack(nn.Module):

    def __init__(self, num_dec_layers, d_model, dim_ff) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, dim_ff=dim_ff) for _ in range(num_dec_layers)]
        )

    def set_lm_mode(self):
        for layer in self.layers:
            layer.set_lm_mode()

    def set_transcription_mode(self):
        for layer in self.layers:
            layer.set_transcription_mode()

    def forward(
        self,
        tgt,
        memory_key,
        memory_value,
        tgt_mask,
        memory_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
        use_cache=False,
        cache=None,
        predict_last_n_only=False,
        keep_all_weights=True,
    ):

        output = tgt
        cache_t = []
        all_weights = {"self": [], "mix": []}

        for i, dec_layer in enumerate(self.layers):
            output, weights_self, weights_cross = dec_layer(
                output,
                memory_key=memory_key,
                memory_value=memory_value,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                predict_n_last_only=predict_last_n_only,
            )

            if use_cache:
                cache_t.append(output)
                if cache is not None:
                    output = torch.cat([cache[i], output], dim=0)

            if keep_all_weights:
                all_weights["self"].append(weights_self)
                all_weights["mix"].append(weights_cross)

        if use_cache:
            cache = (
                torch.cat([cache, torch.stack(cache_t, dim=0)], dim=1)
                if cache is not None
                else torch.stack(cache_t, dim=0)
            )

        if predict_last_n_only:
            output = output[-predict_last_n_only:]

        if keep_all_weights:
            return output, all_weights, cache

        return output, weights_cross, cache


class Decoder(nn.Module):
    def __init__(
        self, d_model, dim_ff, n_layers, maxlen, out_categories, attention_window=100
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dec_attn_win = attention_window
        self.positional_1D = PositionalEncoding1D(d_model, maxlen)

        self.decoder = DecoderStack(num_dec_layers=n_layers, d_model=d_model, dim_ff=dim_ff)

        self.embedding = nn.Embedding(num_embeddings=out_categories, embedding_dim=d_model)

        self.end_relu = nn.ReLU()

        self.out_layer = nn.Conv1d(d_model, out_categories, kernel_size=1)

    def set_lm_mode(self):
        self.decoder.set_lm_mode()

    def set_transcription_mode(self):
        self.decoder.set_transcription_mode()

    def forward(
        self,
        raw_features_1D,
        enhanced_features_1D,
        tokens,
        reduced_size,
        token_len,
        features_size,
        hidden_predict=None,
        num_pred=None,
        cache=None,
        keep_all_weights=True,
    ):

        device = raw_features_1D.device

        pos_tokens = self.embedding(tokens).permute(0, 2, 1)

        pos_tokens = self.positional_1D(pos_tokens, start=0)
        pos_tokens = pos_tokens.permute(2, 0, 1).contiguous()

        if num_pred is None:
            num_pred = tokens.size(1)

        if self.dec_attn_win > 1 and cache is not None:
            cache = cache[:, -self.dec_attn_win - 1]
        else:
            cache = None

        num_tokens_to_keep = (
            num_pred
            if self.dec_attn_win is None
            else min([num_pred + self.dec_attn_win - 1, pos_tokens.size(0), token_len[0]])
        )
        pos_tokens = pos_tokens[-num_tokens_to_keep:]

        target_mask = self.generate_target_mask(tokens.size(1), device=device)
        memory_mask = None

        key_target_mask = self.generate_token_mask(token_len, tokens.size(), device)
        key_memory_mask = None  # self.generate_enc_mask(reduced_size, features_size, device)

        target_mask = target_mask[-num_pred:, -num_tokens_to_keep:]
        key_target_mask = key_target_mask[:, -num_tokens_to_keep:]

        output, weights, cache = self.decoder(
            pos_tokens,
            memory_key=enhanced_features_1D,
            memory_value=raw_features_1D,
            tgt_mask=target_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=key_target_mask,
            memory_key_padding_mask=key_memory_mask,
            use_cache=True,
            cache=cache,
            predict_last_n_only=num_pred,
            keep_all_weights=keep_all_weights,
        )

        dpoutput = self.dropout(self.end_relu(output))

        predictions = self.out_layer(dpoutput.permute(1, 2, 0).contiguous())

        if not keep_all_weights:
            weights = torch.sum(weights, dim=1, keepdim=True).reshape(
                -1, 1, features_size[2], features_size[3]
            )

        return output, predictions, hidden_predict, cache, weights

    def generate_enc_mask(self, batch_reduced_size, total_size, device):
        batch_size, _, h_max, w_max = total_size
        mask = torch.ones((batch_size, h_max, w_max), dtype=torch.bool, device=device)
        for i, (h, w) in enumerate(batch_reduced_size):
            mask[i, :h, :w] = False
        return torch.flatten(mask, start_dim=1, end_dim=2)

    def generate_token_mask(self, token_len, total_size, device):
        batch_size, len_mask = total_size
        mask = torch.zeros((batch_size, len_mask), dtype=torch.bool, device=device)
        for i, len_ in enumerate(token_len):
            mask[i, :len_] = False

        return mask

    def generate_target_mask(self, target_len, device):
        if self.dec_attn_win == 1:
            return torch.triu(
                torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=1
            )
        else:
            return torch.logical_not(
                torch.logical_and(
                    torch.tril(
                        torch.ones((target_len, target_len), dtype=torch.bool, device=device),
                        diagonal=0,
                    ),
                    torch.triu(
                        torch.ones((target_len, target_len), dtype=torch.bool, device=device),
                        diagonal=-self.dec_attn_win + 1,
                    ),
                )
            )


class SMTOutput(CausalLMOutputWithCrossAttentions):
    """This is a nice output wrapper"""


class SMTModelForCausalLM(PreTrainedModel):
    config_class = Config

    def __init__(self, config: Config):
        super().__init__(config)
        next_config = ConvNextConfig(
            num_channels=config.channels,
            num_stages=3,
            hidden_sizes=[64, 128, 256],
            depths=[3, 3, 9],
        )
        self.encoder = ConvNextModel(next_config)

        self.note_decoder = Decoder(
            d_model=config.decoder_dim,
            dim_ff=config.decoder_dim,
            n_layers=config.decoder_depth,
            maxlen=config.max_seq_len,
            out_categories=config.num_note_tokens,
            attention_window=config.max_seq_len + 1,
        )

        self.rhythm_decoder = Decoder(
            d_model=config.decoder_dim,
            dim_ff=config.decoder_dim,
            n_layers=config.decoder_depth,
            maxlen=config.max_seq_len,
            out_categories=config.num_rhythm_tokens,
            attention_window=config.max_seq_len + 1,
        )

        self.pitch_decoder = Decoder(
            d_model=config.decoder_dim,
            dim_ff=config.decoder_dim,
            n_layers=config.decoder_depth,
            maxlen=config.max_seq_len,
            out_categories=config.num_pitch_tokens,
            attention_window=config.max_seq_len + 1,
        )

        self.lift_decoder = Decoder(
            d_model=config.decoder_dim,
            dim_ff=config.decoder_dim,
            n_layers=config.decoder_depth,
            maxlen=config.max_seq_len,
            out_categories=config.num_lift_tokens,
            attention_window=config.max_seq_len + 1,
        )

        self.positional_2D = PositionalEncoding2D(
            config.decoder_dim, config.max_width, config.max_height
        )

        self.padding_token = config.pad_token
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token)

        self.maxlen = config.max_seq_len
        self.config = config

        note_mask = torch.zeros(config.num_rhythm_tokens)
        note_mask[config.noteindexes] = 1
        self.note_mask = nn.Parameter(note_mask)

    def forward_encoder(self, x):
        return self.encoder(pixel_values=x).last_hidden_state

    def forward_decoder(self, encoder_output, rhythms_seq, pitchs_seq, lifts_seq, note_seq) -> tuple[SMTOutput, SMTOutput, SMTOutput, SMTOutput]:
        b, _, _, _ = encoder_output.size()
        reduced_size = [s.shape[:2] for s in encoder_output]
        ylens = [len(sample) for sample in rhythms_seq]

        pos_features = self.positional_2D(encoder_output)
        features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(2, 0, 1)
        enhanced_features = features
        enhanced_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
        note_output = self.wrap_decoder_result(
            self.note_decoder(
                features,
                enhanced_features,
                note_seq[:, :],
                reduced_size,
                [max(ylens) for _ in range(b)],
                encoder_output.size(),
                cache=None,
                keep_all_weights=True,
            )
        )
        rhythms_output = self.wrap_decoder_result(
            self.rhythm_decoder(
                features,
                enhanced_features,
                rhythms_seq[:, :],
                reduced_size,
                [max(ylens) for _ in range(b)],
                encoder_output.size(),
                cache=None,
                keep_all_weights=True,
            )
        )
        pitch_output = self.wrap_decoder_result(
            self.pitch_decoder(
                features,
                enhanced_features,
                pitchs_seq[:, :],
                reduced_size,
                [max(ylens) for _ in range(b)],
                encoder_output.size(),
                cache=None,
                keep_all_weights=True,
            )
        )
        lift_output = self.wrap_decoder_result(
            self.lift_decoder(
                features,
                enhanced_features,
                lifts_seq[:, :],
                reduced_size,
                [max(ylens) for _ in range(b)],
                encoder_output.size(),
                cache=None,
                keep_all_weights=True,
            )
        )
        return note_output, rhythms_output, pitch_output, lift_output

    def wrap_decoder_result(self, result) -> SMTOutput:
        output, predictions, _, _, weights = result
        return SMTOutput(
            logits=predictions.permute(0, 2, 1),
            hidden_states=output,
            attentions=weights["self"],
            cross_attentions=weights["mix"],
        )

    def forward(self, inputs, rhythms_seq, pitchs_seq, lifts_seq, note_seq, mask):
        inputs = self.forward_encoder(inputs)
        note_output, rhythms_output, pitch_output, lift_output = self.forward_decoder(
            inputs, rhythms_seq, pitchs_seq, lifts_seq, note_seq
        )

        loss_consist = self.calConsistencyLoss(
            rhythms_output.logits, pitch_output.logits, lift_output.logits, note_output.logits, mask
        )
        
        loss_rhythm = self.masked_logits_cross_entropy(rhythms_output.logits, rhythms_seq, mask)
        loss_pitch = self.masked_logits_cross_entropy(pitch_output.logits, pitchs_seq, mask)
        loss_lift = self.masked_logits_cross_entropy(lift_output.logits, lifts_seq, mask)
        loss_note = self.masked_logits_cross_entropy(note_output.logits, note_seq, mask)
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
        loss = F.cross_entropy(logits.transpose(1, 2), target, reduction="none", weight=weights)

        # As reduction is "none", we can apply the mask to the loss
        # and this way we ignore the loss for the padded tokens
        loss = loss * mask
        loss = loss.sum() / mask.sum()

        return loss

    def predict(self, input_tensor, convert_to_str=False):
        predicted_sequence = (
            torch.from_numpy(np.asarray([self.config.bos_token]))
            .to(input_tensor.device)
            .unsqueeze(0)
        )
        encoder_output = self.forward_encoder(input_tensor)
        text_sequence = []
        for _ in range(self.maxlen - predicted_sequence.shape[-1]):
            note_output, rhythms_output, pitch_output, lift_output = self.forward_decoder(
                encoder_output, predicted_sequence.long()
            )
            # TODO merge decoder outputs
            predicted_token = torch.argmax(rhythms_output.logits[:, :, -1]).item()
            predicted_sequence = torch.cat(
                [
                    predicted_sequence,
                    torch.argmax(rhythms_output.logits[:, :, -1], dim=1, keepdim=True),
                ],
                dim=1,
            )
            if convert_to_str:
                predicted_token = f"{predicted_token}"
            if predicted_token == self.config.eos_token:
                break
            text_sequence.append(self.config.rhythm_vocab[predicted_token])

        return text_sequence, rhythms_output
