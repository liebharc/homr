from math import ceil
from typing import Any

import numpy as np
import onnxruntime as ort

from homr.transformer.configs import Config
from homr.transformer.utils import softmax
from homr.transformer.vocabulary import EncodedSymbol
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
        self.eos_token = config.eos_token

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}
        self.inv_articulation_vocab = {v: k for k, v in config.articulation_vocab.items()}
        self.inv_position_vocab = {v: k for k, v in config.position_vocab.items()}
        self.state_vocab = config.state_vocab

    def generate(
        self,
        start_tokens: NDArray,
        nonote_tokens: NDArray,
        temperature: float = 1.0,
        filter_thres: float = 0.7,
        **kwargs: Any,
    ) -> list[EncodedSymbol]:
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        out_rhythm = start_tokens
        out_pitch = nonote_tokens
        out_lift = nonote_tokens
        out_articulations = nonote_tokens
        key = "keySignature_0"
        clef_upper = "clef_G2"
        clef_lower = "clef_F4"
        states = np.array([[self.state_vocab[f"{key}+{clef_upper}+{clef_lower}"]]])

        symbols: list[EncodedSymbol] = []

        for _ in range(self.max_seq_len):
            x_lift = out_lift[:, -self.max_seq_len :]
            x_pitch = out_pitch[:, -self.max_seq_len :]
            x_rhythm = out_rhythm[:, -self.max_seq_len :]
            x_articulations = out_articulations[:, -self.max_seq_len :]
            context = kwargs["context"]

            inputs = {
                "rhythms": x_rhythm,
                "pitchs": x_pitch,
                "lifts": x_lift,
                "articulations": x_articulations,
                "states": states,
                "context": context,
            }

            rhythmsp, pitchsp, liftsp, positionsp, articulationsp = self.net.run(
                output_names=[
                    "out_rhythms",
                    "out_pitchs",
                    "out_lifts",
                    "out_positions",
                    "out_articulations",
                ],
                input_feed=inputs,
            )

            filtered_lift_logits = top_k(liftsp[:, -1, :], thres=filter_thres)
            filtered_pitch_logits = top_k(pitchsp[:, -1, :], thres=filter_thres)
            filtered_rhythm_logits = top_k(rhythmsp[:, -1, :], thres=filter_thres)
            filtered_articulations_logits = top_k(articulationsp[:, -1, :], thres=filter_thres)
            filtered_positions_logits = top_k(positionsp[:, -1, :], thres=filter_thres)

            lift_probs = softmax(filtered_lift_logits / temperature, dim=-1)
            pitch_probs = softmax(filtered_pitch_logits / temperature, dim=-1)
            rhythm_probs = softmax(filtered_rhythm_logits / temperature, dim=-1)
            articulation_probs = softmax(filtered_articulations_logits / temperature, dim=-1)
            positions_probs = softmax(filtered_positions_logits / temperature, dim=-1)

            lift_sample = np.array([[lift_probs.argmax()]])
            pitch_sample = np.array([[pitch_probs.argmax()]])
            rhythm_sample = np.array([[rhythm_probs.argmax()]])
            articulation_sample = np.array([[articulation_probs.argmax()]])
            position_sample = np.array([[positions_probs.argmax()]])

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
            if symbol.rhythm.startswith("keySignature"):
                key = symbol.rhythm
            elif symbol.rhythm.startswith("clef"):
                if symbol.position == "upper":
                    clef_upper = symbol.rhythm
                else:
                    clef_lower = symbol.rhythm
            states = np.concatenate(
                (states, np.array([[self.state_vocab[f"{key}+{clef_upper}+{clef_lower}"]]])),
                axis=-1,
            )

            out_lift = np.concatenate((out_lift, lift_sample), axis=-1)
            out_pitch = np.concatenate((out_pitch, pitch_sample), axis=-1)
            out_rhythm = np.concatenate((out_rhythm, rhythm_sample), axis=-1)
            out_articulations = np.concatenate((out_articulations, articulation_sample), axis=-1)

        return symbols


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


def detokenize(tokens: NDArray, vocab: dict[int, str]) -> list[str]:
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
