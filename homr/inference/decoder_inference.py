from typing import Any
import numpy as np
from math import ceil
import onnxruntime as ort

from homr.inference.transformer.configs import Config
from homr.inference.transformer.split_merge_symbols import SymbolMerger

from homr.inference.utils import softmax

class ScoreDecoder():
    def __init__(
        self,
        transformer,
        config: Config,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.pad_value = (config.pad_token,)
        self.ignore_index = ignore_index
        self.config = config
        self.net = transformer
        self.max_seq_len = config.max_seq_len

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}

    def generate(  # noqa: PLR0915
        self,
        start_tokens: np.ndarray,
        nonote_tokens: np.ndarray,
        seq_len: int=256,
        eos_token: int | None = None,
        temperature: float = 1.0,
        filter_thres: float = 0.7,
        **kwargs: Any,
    ) -> list[str]:
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
            context = kwargs['context']
            
            inputs = {"rhythms": x_rhythm,
                    "pitchs": x_pitch,
                    "lifts": x_lift,
                    "context": context
                }

            rhythmsp, pitchsp, liftsp = self.net.run(output_names=["out_rhythms", "out_pitchs", "out_lifts"],input_feed=inputs)

            filtered_lift_logits = top_k(liftsp[:, -1, :],   thres=filter_thres)
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

                lift_token = detokenize(lift_sample, self.inv_lift_vocab)
                pitch_token = detokenize(pitch_sample, self.inv_pitch_vocab)
                rhythm_token = detokenize(rhythm_sample, self.inv_rhythm_vocab)

                is_eos = len(rhythm_token)
                if is_eos == 0:
                    break
                retry = merger.add_symbol(rhythm_token[0], pitch_token[0], lift_token[0])
                current_temperature *= 3.5
                attempt += 1


            out_lift = np.concatenate((out_lift, lift_sample), axis=-1)
            out_pitch = np.concatenate((out_pitch, pitch_sample), axis=-1)
            out_rhythm = np.concatenate((out_rhythm, rhythm_sample), axis=-1)

            if (
                eos_token is not None
                and (np.cumsum(out_rhythm == eos_token, 1)[:, -1] >= 1).all()
                ):
                break

        out_lift = out_lift[:, t:]
        out_pitch = out_pitch[:, t:]
        out_rhythm = out_rhythm[:, t:]

        return merger.complete()



def top_k(logits: np.ndarray, thres: float = 0.9) -> np.ndarray:
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


def detokenize(tokens: np.ndarray, vocab: Any) -> list[str]:
    toks = [vocab[tok.item()] for tok in tokens]
    toks = [t for t in toks if t not in ("[BOS]", "[EOS]", "[PAD]")]
    return toks

def get_decoder(config: Config, path):
    onnx_transformer = ort.InferenceSession(path)
    return ScoreDecoder(onnx_transformer, config=config)