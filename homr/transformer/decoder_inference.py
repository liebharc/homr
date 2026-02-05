from typing import Any

import numpy as np
import onnxruntime as ort

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray


class ScoreDecoder:
    def __init__(
        self,
        transformer: ort.InferenceSession,
        fp16: bool,
        use_gpu: bool,
        config: Config,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.net = transformer
        self.io_binding = self.net.io_binding()
        self.max_seq_len = config.max_seq_len
        self.eos_token = config.eos_token

        self.inv_rhythm_vocab = {v: k for k, v in config.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in config.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in config.lift_vocab.items()}
        self.inv_articulation_vocab = {v: k for k, v in config.articulation_vocab.items()}
        self.inv_position_vocab = {v: k for k, v in config.position_vocab.items()}

        self.fp16 = fp16
        self.use_gpu = use_gpu
        self.device_id = 0
        self.output_names = [
            "out_rhythms",
            "out_pitchs",
            "out_lifts",
            "out_positions",
            "out_articulations",
            "attention",
        ]

    def generate(
        self,
        start_tokens: NDArray,
        nonote_tokens: NDArray,
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
        cache, kv_input_names, kv_output_names = self.init_cache()
        output_names = self.output_names + kv_output_names
        context = kwargs["context"]
        context_reduced = kwargs["context"][:, :1]

        symbols: list[EncodedSymbol] = []

        for step in range(self.max_seq_len):
            x_lift = out_lift[:, -1:]  # for all: shape=(1,1)
            x_pitch = out_pitch[:, -1:]
            x_rhythm = out_rhythm[:, -1:]
            x_articulations = out_articulations[:, -1:]

            # after the first step we don't pass the full context into the decoder
            # x_transformers uses [:, :0] to split the context
            # which caused a Reshape error when loading the onnx model
            context = context if step == 0 else context_reduced

            # Bind Inputs
            self.io_binding.bind_cpu_input("rhythms", x_rhythm)
            self.io_binding.bind_cpu_input("pitchs", x_pitch)
            self.io_binding.bind_cpu_input("lifts", x_lift)
            self.io_binding.bind_cpu_input("articulations", x_articulations)
            self.io_binding.bind_cpu_input("context", context)
            self.io_binding.bind_cpu_input("cache_len", np.array([step], dtype=np.int64))
            for name, cache_val in zip(kv_input_names, cache, strict=True):
                self.io_binding.bind_ortvalue_input(name, cache_val)

            # Bind Outputs
            for name in output_names:
                self.io_binding.bind_output(name, "cuda" if self.use_gpu else "cpu", self.device_id)

            # Run inference
            self.net.run_with_iobinding(iobinding=self.io_binding)

            # Get outputs
            outputs = self.io_binding.get_outputs()
            cache = outputs[6:]

            # Greedy decoding: pick the highest logit directly for each output
            rhythmsp = outputs[0].numpy()
            pitchsp = outputs[1].numpy()
            liftsp = outputs[2].numpy()
            positionsp = outputs[3].numpy()
            articulationsp = outputs[4].numpy()
            attention = outputs[5].numpy()

            rhythm_sample = np.array([[rhythmsp[:, -1, :].argmax()]])
            pitch_sample = np.array([[pitchsp[:, -1, :].argmax()]])
            lift_sample = np.array([[liftsp[:, -1, :].argmax()]])
            articulation_sample = np.array([[articulationsp[:, -1, :].argmax()]])
            position_sample = np.array([[positionsp[:, -1, :].argmax()]])

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
                coordinates=attention,
            )
            symbols.append(symbol)

            out_lift = np.concatenate((out_lift, lift_sample), axis=-1)
            out_pitch = np.concatenate((out_pitch, pitch_sample), axis=-1)
            out_rhythm = np.concatenate((out_rhythm, rhythm_sample), axis=-1)
            out_articulations = np.concatenate((out_articulations, articulation_sample), axis=-1)

        return symbols

    def init_cache(self, cache_len: int = 0) -> tuple[list[NDArray], list[str], list[str]]:
        cache = []
        input_names = []
        output_names = []
        for i in range(32):
            if self.fp16:  # the cache needs to be fp16 as well
                cache.append(
                    ort.OrtValue.ortvalue_from_numpy(
                        np.zeros((1, 8, cache_len, 64), dtype=np.float16),
                        "cuda" if self.use_gpu else "cpu",
                        self.device_id,
                    )
                )
            else:
                cache.append(
                    ort.OrtValue.ortvalue_from_numpy(
                        np.zeros((1, 8, cache_len, 64), dtype=np.float32),
                        "cuda" if self.use_gpu else "cpu",
                        self.device_id,
                    )
                )
            input_names.append(f"cache_in{i}")
            output_names.append(f"cache_out{i}")
        return cache, input_names, output_names


def detokenize(tokens: NDArray, vocab: dict[int, str]) -> list[str]:
    toks = [vocab[tok.item()] for tok in tokens]
    toks = [t for t in toks if t not in ("[BOS]", "[EOS]", "[PAD]")]
    return toks


def get_decoder(config: Config) -> ScoreDecoder:
    """
    Returns Tromr's Decoder
    """
    use_gpu = False
    if config.use_gpu_inference:
        try:
            onnx_transformer = ort.InferenceSession(
                config.filepaths.decoder_path_fp16, providers=["CUDAExecutionProvider"]
            )
            fp16 = True
            use_gpu = True
        except Exception as ex:
            eprint(ex)
            eprint("Going on without GPU support")
            onnx_transformer = ort.InferenceSession(config.filepaths.decoder_path_fp16)
            fp16 = True

    else:
        onnx_transformer = ort.InferenceSession(config.filepaths.decoder_path)
        fp16 = False

    return ScoreDecoder(onnx_transformer, fp16, use_gpu, config=config)
