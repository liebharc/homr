from math import ceil
from typing import Any

import numpy as np
import onnxruntime as ort

from homr.results import TransformerChord
from homr.transformer.configs import Config
from homr.transformer.split_merge_symbols import SymbolMerger
from homr.transformer.utils import softmax
from homr.type_definitions import NDArray


class ScoreDecoder:
    def __init__(
        self,
        transformer: ort.InferenceSession,
        config: Config,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.net = transformer
        self.max_seq_len = config.max_seq_len

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}

    def generate(  # noqa: PLR0915
        self,
        start_tokens: NDArray,
        nonote_tokens: NDArray,
        seq_len: int = 256,
        eos_token: int | None = None,
        temperature: float = 1.0,
        filter_thres: float = 0.7,
        **kwargs: Any,
    ) -> list[TransformerChord]:
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        out_rhythm = start_tokens
        out_pitch = nonote_tokens
        out_lift = nonote_tokens
        merger = SymbolMerger()

        for _position_in_seq in range(seq_len):
            x_lift = out_lift[:, -self.max_seq_len :]
            x_pitch = out_pitch[:, -self.max_seq_len :]
            x_rhythm = out_rhythm[:, -self.max_seq_len :]
            context = kwargs["context"]

            inputs = {"rhythms": x_rhythm, "pitchs": x_pitch, "lifts": x_lift, "context": context}

            rhythmsp, pitchsp, liftsp = self.net.run(
                output_names=["out_rhythms", "out_pitchs", "out_lifts"],  # noqa: E501
                input_feed=inputs,
            )

            filtered_lift_logits = top_k(liftsp[:, -1, :], thres=filter_thres)
            filtered_pitch_logits = top_k(pitchsp[:, -1, :], thres=filter_thres)
            filtered_rhythm_logits = top_k(rhythmsp[:, -1, :], thres=filter_thres)

            current_temperature = temperature
            retry = True
            attempt = 0
            max_attempts = 5
            while retry and attempt < max_attempts:
                lift_probs = softmax(filtered_lift_logits / current_temperature, dim=-1)
                pitch_probs = softmax(filtered_pitch_logits / current_temperature, dim=-1)
                rhythm_probs = softmax(filtered_rhythm_logits / current_temperature, dim=-1)

                lift_sample = np.array([[lift_probs.argmax()]])
                pitch_sample = np.array([[pitch_probs.argmax()]])
                rhythm_sample = np.array([[rhythm_probs.argmax()]])

                sorted_indices = np.argsort(rhythm_probs)[:, ::-1]
                sorted_probs = np.take_along_axis(rhythm_probs, sorted_indices, axis=1)

                rhythm_confidence = sorted_probs[0, 0].item()
                alternative_confidence = sorted_probs[0, 1].item()

                top_token_id = np.expand_dims(sorted_indices[0, 0], axis=0)
                alt_token_id = np.expand_dims(sorted_indices[0, 1], axis=0)

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

            out_lift = np.concatenate((out_lift, lift_sample), axis=-1)
            out_pitch = np.concatenate((out_pitch, pitch_sample), axis=-1)
            out_rhythm = np.concatenate((out_rhythm, rhythm_sample), axis=-1)

            if eos_token is not None and (np.cumsum(out_rhythm == eos_token, 1)[:, -1] >= 1).all():
                break

        out_lift = out_lift[:, t:]
        out_pitch = out_pitch[:, t:]
        out_rhythm = out_rhythm[:, t:]

        return merger.complete()


def top_k(logits: NDArray, thres: float = 0.9) -> NDArray:
    """Numpy implementation matching torch's top_k behavior"""
    k = ceil((1 - thres) * logits.shape[-1])

    # Get top k elements
    flat_logits = logits.ravel()
    indices = np.argpartition(flat_logits, -k)[-k:]  # Get indices of top k elements
    indices = indices[np.argsort(-flat_logits[indices])]  # Sort them in descending order
    values = flat_logits[indices]  # Get the corresponding values

    # Create output array with -inf
    output = np.full_like(logits, -np.inf)

    # Scatter the topk values back into the output array
    # For multi-dimensional arrays, we need to convert flat indices to multi-indices
    if logits.ndim > 1:
        multi_indices = np.unravel_index(indices, logits.shape)
        output[multi_indices] = values
    else:
        output[indices] = values

    return output


def detokenize(tokens: NDArray, vocab: Any) -> list[str]:
    toks = [vocab[tok.item()] for tok in tokens]
    toks = [t for t in toks if t not in ("[BOS]", "[EOS]", "[PAD]")]
    return toks


def get_decoder(config: Config, path: str, use_gpu: bool) -> ScoreDecoder:
    """
    Returns Tromr's Decoder
    """
    if use_gpu:
        try:
            onnx_transformer = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
        except Exception:
            onnx_transformer = ort.InferenceSession(path)

    else:
        onnx_transformer = ort.InferenceSession(path)

    return ScoreDecoder(onnx_transformer, config=config)
