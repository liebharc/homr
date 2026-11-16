"""
Decoding with the batched decoder (decoder_*_dyn.onnx: dynamic batch and beam_index).

Every staff is read from two crops (see staff_parsing). One decoder call per step runs
all rows: the ensemble (both crops' log-probs averaged, both rows fed its token) and a
greedy reading per crop. The reading whose bars fit best wins (bar_fit). Bars which still
don't fit are decoded again by a small beam search which only accepts fitting bars
(_repair_bars).

The decoder graph takes `beam_index`, the cache row which each output row continues, and
cuts the cache to `cache_len`. So beams can be reordered and decoding can continue from
an earlier step without copying the cache.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import onnxruntime as ort

from homr.onnx_providers import gpu_providers
from homr.simple_logging import eprint
from homr.transformer.bar_fit import (
    StaffBars,
    bar_durations,
    bar_fit,
    fits,
    has_unfit_bar,
    header,
    is_bar_end,
)
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray

HEADS = ("rhythm", "pitch", "lift", "position", "articulation", "slur")
FED = ("rhythm", "pitch", "lift", "articulation", "slur")  # the decoder's token inputs
INPUT_NAMES = {
    "rhythm": "rhythms",
    "pitch": "pitchs",
    "lift": "lifts",
    "articulation": "articulations",
    "slur": "slurs",
}
OUTPUT_NAMES = [
    "out_rhythms",
    "out_pitchs",
    "out_lifts",
    "out_positions",
    "out_articulations",
    "out_slurs",
    "attention",
]
START = {"rhythm": 1, "pitch": 0, "lift": 0, "articulation": 0, "slur": 0}

# A repaired bar may be at most this much less likely (log-prob, about 10x) than the bar
# it replaces.
REPAIR_DELTA = 2.3
REPAIR_WIDTH = 4  # beams
REPAIR_RHYTHMS = 3  # rhythm alternatives per beam and step, other heads take the best

Token = dict[str, int]
LogProbs = dict[str, NDArray]


def _log_softmax(x: NDArray) -> NDArray:
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def _average(log_probs: list[LogProbs]) -> LogProbs:
    return {h: np.mean([lp[h] for lp in log_probs], axis=0) for h in HEADS}


def _score(log_probs: LogProbs, token: Token) -> float:
    return float(sum(log_probs[h][token[h]] for h in HEADS))


def _fed(token: Token) -> Token:
    return {h: token[h] for h in FED}


@dataclass
class _Decoded:
    token: Token
    attention: NDArray
    score: float
    log_probs: LogProbs | None  # None: unknown (inside a repaired bar)


@dataclass
class _Cache:
    """Decoder cache and the rows in it which belong to one reading (one row per crop)."""

    values: list[Any]
    rows: list[int]


class ScoreDecoder:
    def __init__(
        self,
        transformer: ort.InferenceSession,
        fp16: bool,
        use_gpu: bool,
        config: Config,
    ):
        self.config = config
        self.net = transformer
        self.max_seq_len = config.max_seq_len
        self.eos_token = config.eos_token
        self.fp16 = fp16
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu else "cpu"
        self.device_id = 0
        self.inv_vocab = {
            "rhythm": {v: k for k, v in config.rhythm_vocab.items()},
            "pitch": {v: k for k, v in config.pitch_vocab.items()},
            "lift": {v: k for k, v in config.lift_vocab.items()},
            "articulation": {v: k for k, v in config.articulation_vocab.items()},
            "slur": {v: k for k, v in config.slur_vocab.items()},
            "position": {v: k for k, v in config.position_vocab.items()},
        }
        n_cache = config.decoder_depth * 4
        self.cache_inputs = [f"cache_in{i}" for i in range(n_cache)]
        self.outputs = OUTPUT_NAMES + [f"cache_out{i}" for i in range(n_cache)]
        self.calls = 0

    def generate(self, contexts: list[NDArray]) -> list[EncodedSymbol]:
        """contexts: the encoder outputs of the crops of one staff."""
        self.calls = 0
        n_views = len(contexts)
        reduced = [np.ascontiguousarray(c[:, :1]) for c in contexts]
        views = (contexts, reduced)
        if n_views == 1:
            rows, readings = [0], [[0]]
        else:
            # ensemble rows (one per crop) and a greedy row per crop
            rows = list(range(n_views)) * 2
            readings = [list(range(n_views))] + [[n_views + v] for v in range(n_views)]
        previous = [dict(START) for _ in rows]
        decoded: list[list[_Decoded]] = [[] for _ in readings]
        done = [False] * len(readings)
        cache: list[Any] | None = None
        for step in range(self.max_seq_len):
            log_probs, attention, cache = self._run(
                previous, rows, views, cache, step, list(range(len(rows)))
            )
            for r, reading_rows in enumerate(readings):
                if done[r]:
                    continue
                lp = _average([log_probs[row] for row in reading_rows])
                token = {h: int(lp[h].argmax()) for h in HEADS}
                if token["rhythm"] == self.eos_token:
                    done[r] = True
                    continue
                decoded[r].append(
                    _Decoded(token, attention[reading_rows[0]], _score(lp, token), lp)
                )
                for row in reading_rows:
                    previous[row] = _fed(token)
            if all(done):
                break
        candidates = [self._symbols(d) for d in decoded]
        fit = [bar_fit(c) for c in candidates]
        best = max(range(len(candidates)), key=lambda i: (fit[i], i == 0))
        if cache is None or not has_unfit_bar(candidates[best]):
            return candidates[best]
        crop_rows = [rows[row] for row in readings[best]]
        return self._repair_bars(decoded[best], crop_rows, views, _Cache(cache, readings[best]))

    def _run(
        self,
        tokens: list[Token],
        row_views: list[int],
        views: tuple[list[NDArray], list[NDArray]],
        cache: list[Any] | None,
        step: int,
        beam_index: list[int],
    ) -> tuple[list[LogProbs], NDArray, list[Any]]:
        """One decoder step. Row i continues cache row beam_index[i]."""
        binding = self.net.io_binding()
        contexts = views[0] if step == 0 else views[1]
        keep = {INPUT_NAMES[h]: np.array([[t[h]] for t in tokens], dtype=np.int64) for h in FED}
        keep["cache_len"] = np.array([step], dtype=np.int64)
        keep["beam_index"] = np.array(beam_index, dtype=np.int64)
        keep["context"] = np.ascontiguousarray(np.concatenate([contexts[v] for v in row_views]))
        for name, value in keep.items():
            binding.bind_cpu_input(name, value)
        if cache is None:
            heads = self.config.decoder_heads
            empty = np.zeros(
                (len(tokens), heads, 0, self.config.decoder_dim // heads),
                dtype=np.float16 if self.fp16 else np.float32,
            )
            cache = [ort.OrtValue.ortvalue_from_numpy(empty, self.device, self.device_id)] * len(
                self.cache_inputs
            )
        for name, value in zip(self.cache_inputs, cache, strict=True):
            binding.bind_ortvalue_input(name, value)
        for name in self.outputs:
            binding.bind_output(name, self.device, self.device_id)
        self.net.run_with_iobinding(iobinding=binding)
        self.calls += 1
        outputs = binding.get_outputs()
        logits = [outputs[i].numpy()[:, -1] for i in range(len(HEADS))]
        log_probs = [
            {h: _log_softmax(logits[k][row]) for k, h in enumerate(HEADS)}
            for row in range(len(tokens))
        ]
        return log_probs, outputs[len(HEADS)].numpy(), list(outputs[len(HEADS) + 1 :])

    def _symbol(self, token: Token, attention: NDArray) -> EncodedSymbol:
        return EncodedSymbol(
            rhythm=self.inv_vocab["rhythm"][token["rhythm"]],
            pitch=self.inv_vocab["pitch"][token["pitch"]],
            lift=self.inv_vocab["lift"][token["lift"]],
            articulation=self.inv_vocab["articulation"][token["articulation"]],
            slur=self.inv_vocab["slur"][token["slur"]],
            position=self.inv_vocab["position"][token["position"]],
            coordinates=(float(attention[0]), float(attention[1])),
        )

    def _symbols(self, decoded: list[_Decoded]) -> list[EncodedSymbol]:
        return [self._symbol(d.token, d.attention) for d in decoded]

    def _decode_from(
        self,
        previous: Token,
        crop_rows: list[int],
        views: tuple[list[NDArray], list[NDArray]],
        cache: _Cache,
        step: int,
        prefix: list[_Decoded],
    ) -> tuple[list[_Decoded], _Cache]:
        """Greedy continuation of prefix, the cache holds the state after step - 1."""
        result = list(prefix)
        n_views = len(crop_rows)
        values, beam_index = cache.values, cache.rows
        while step < self.max_seq_len:
            log_probs, attention, values = self._run(
                [previous] * n_views, crop_rows, views, values, step, beam_index
            )
            beam_index = list(range(n_views))
            lp = _average(log_probs)
            token = {h: int(lp[h].argmax()) for h in HEADS}
            if token["rhythm"] == self.eos_token:
                break
            result.append(_Decoded(token, attention[0], _score(lp, token), lp))
            previous, step = _fed(token), step + 1
        return result, _Cache(values, list(range(n_views)))

    def _repair_bars(
        self,
        decoded: list[_Decoded],
        crop_rows: list[int],
        views: tuple[list[NDArray], list[NDArray]],
        cache: _Cache,
    ) -> list[EncodedSymbol]:
        """
        Every closed bar which doesn't fit is decoded again from its start with a beam
        search (top rhythms, other heads take the best) that only accepts fitting bars.
        The best accepted bar replaces the original one if its log-prob is at most
        REPAIR_DELTA lower, and the rest of the staff is decoded again after it.
        """
        n_views = len(crop_rows)
        next_bar = 0
        while True:
            symbols = self._symbols(decoded)
            bars = StaffBars(symbols)
            if not bars.checkable():
                break
            target = bars.first_unfit(next_bar)
            if target is None:
                break
            next_bar = target + 1
            start, end = bars.bars[target]
            start_log_probs = decoded[start].log_probs
            if start_log_probs is None:
                continue
            original_score = sum(decoded[j].score for j in range(start, end + 1))
            bar_header = header(symbols[start:end])
            # beams: (score, path, log-probs of the next token, its attention)
            beams: list[tuple[float, list[_Decoded], LogProbs, NDArray]] = [
                (0.0, [], start_log_probs, decoded[start].attention)
            ]
            beam_cache = cache.values
            # cache rows of each beam; the first step cuts the staff's cache back to the bar
            # start through cache_len
            rows_of = [cache.rows]
            step = start
            accepted: list[tuple[float, list[_Decoded], _Cache, int]] = []
            for _ in range(end - start + 1 + 4):
                expanded = []
                for b, (score, path, lp, attention) in enumerate(beams):
                    best = {h: int(lp[h].argmax()) for h in HEADS}
                    for rhythm in np.argsort(-lp["rhythm"])[:REPAIR_RHYTHMS]:
                        token = dict(best, rhythm=int(rhythm))
                        expanded.append((score + _score(lp, token), path, token, attention, b))
                expanded.sort(key=lambda e: -e[0])
                survivors: list[tuple[float, list[_Decoded], Token, int]] = []
                for score, path, token, attention, b in expanded:
                    if len(survivors) >= REPAIR_WIDTH:
                        break
                    if token["rhythm"] == self.eos_token:
                        continue
                    path_with = [*path, _Decoded(token, attention, score, None)]
                    if is_bar_end(self.inv_vocab["rhythm"][token["rhythm"]]):
                        bar = self._symbols(path)
                        upper, lower = bar_durations(bar)
                        if (
                            upper > 0
                            and header(bar) == bar_header
                            and fits(upper, lower, bars.typical, bars.has_lower, target == 0)
                        ):
                            accepted.append(
                                (score, path_with, _Cache(beam_cache, rows_of[b]), step)
                            )
                        continue
                    if step + 1 >= self.max_seq_len:
                        continue
                    survivors.append((score, path_with, token, b))
                if not survivors:
                    break
                beam_index = [row for _, _, _, b in survivors for row in rows_of[b]]
                log_probs, attention_rows, beam_cache = self._run(
                    [_fed(token) for _, _, token, _ in survivors for _ in range(n_views)],
                    crop_rows * len(survivors),
                    views,
                    beam_cache,
                    step + 1,
                    beam_index,
                )
                step += 1
                beams = [
                    (
                        score,
                        path,
                        _average(log_probs[i * n_views : (i + 1) * n_views]),
                        attention_rows[i * n_views],
                    )
                    for i, (score, path, _, _) in enumerate(survivors)
                ]
                rows_of = [list(range(i * n_views, (i + 1) * n_views)) for i in range(len(beams))]
            if not accepted:
                continue
            score, path, bar_cache, bar_step = max(accepted, key=lambda a: a[0])
            if score < original_score - REPAIR_DELTA:
                continue
            new_bar = [_Decoded(d.token, d.attention, 0.0, None) for d in path]
            decoded, cache = self._decode_from(
                _fed(path[-1].token),
                crop_rows,
                views,
                bar_cache,
                bar_step + 1,
                decoded[:start] + new_bar,
            )
        return self._symbols(decoded)


def get_decoder(config: Config) -> ScoreDecoder:
    """
    Returns Tromr's Decoder
    """
    use_gpu = False
    if config.use_gpu_inference:
        try:
            providers, device = gpu_providers()
            onnx_transformer = ort.InferenceSession(
                config.filepaths.decoder_path_fp16, providers=providers
            )
            fp16 = True
            # Sometimes Ort falls automatically back to the CPU EP
            # if so we get an error due to the device selection in init_cache().
            # CoreML binds IO on the CPU (device == "cpu") even when the GPU/ANE runs the
            # compute, so we only flip use_gpu on when CUDA/ROCm device memory is in play.
            active = onnx_transformer.get_providers()
            if device == "cuda":
                for active_provider in active:
                    if active_provider in {"CUDAExecutionProvider", "ROCMExecutionProvider"}:
                        use_gpu = True
                        break
                if not use_gpu:
                    eprint(
                        "Onnxruntime is not using GPU and therefore falling back to CPU. "
                        "This is slow."
                    )

        except Exception as ex:
            eprint(ex)
            eprint("Going on without GPU support")
            onnx_transformer = ort.InferenceSession(config.filepaths.decoder_path_fp16)
            fp16 = True

    else:
        onnx_transformer = ort.InferenceSession(config.filepaths.decoder_path)
        fp16 = False

    return ScoreDecoder(onnx_transformer, fp16, use_gpu, config=config)
