"""
ONNX-only TrOMR wrapper for HOMR adversarial benchmark.

This wrapper is for benchmark Track B: cached HOMR-prepared staff images.

It runs:

    prepared staff image [256, 1280], float32 [0, 1]
        -> TrOMR normalization
        -> tromr_encoder.onnx
        -> tromr_decoder.onnx autoregressive multi-stream decoding
        -> list[EncodedSymbol]

It intentionally does NOT call:
    - Staff2Score.predict(...)
    - Encoder(...)
    - get_decoder(...)
    - parse_staff_tromr(...)
    - PyTorch
    - TensorFlow

Allowed HOMR Python usage here:
    - Config for constants/vocabulary
    - Vocabulary / EncodedSymbol for deterministic token decoding

Neural inference goes only through ONNX Runtime.
"""

from __future__ import annotations

import argparse
from html import parser
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnxruntime as ort

FORBIDDEN_HOMR_NEURAL_IMPORTS = [
    "homr.transformer.staff2score",
    "homr.transformer.encoder_inference",
    "homr.transformer.decoder_inference",
    "homr.staff_parsing_tromr",
]


def assert_no_forbidden_homr_neural_modules_loaded() -> None:
    loaded = [
        name
        for name in FORBIDDEN_HOMR_NEURAL_IMPORTS
        if name in sys.modules
    ]

    if loaded:
        raise RuntimeError(
            "Forbidden HOMR neural inference modules were loaded in ONNX benchmark mode:\n"
            + "\n".join(f"  - {name}" for name in loaded)
            + "\nThis benchmark wrapper must use tromr_encoder.onnx and "
                "tromr_decoder.onnx through ONNX Runtime only."
        )


# ---------------------------------------------------------------------
# Make project imports work when running from repository root.
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol


# ---------------------------------------------------------------------
# Small local Levenshtein implementation.
# This avoids requiring editdistance inside the wrapper itself.
# statistics_engine.py can still use editdistance elsewhere.
# ---------------------------------------------------------------------


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    """
    Compute Levenshtein edit distance between two token lists.
    """
    n = len(a)
    m = len(b)

    if n == 0:
        return m

    if m == 0:
        return n

    prev = list(range(m + 1))
    cur = [0] * (m + 1)

    for i in range(1, n + 1):
        cur[0] = i

        for j in range(1, m + 1):
            substitution_cost = 0 if a[i - 1] == b[j - 1] else 1

            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + substitution_cost,
            )

        prev, cur = cur, prev

    return prev[m]


def symbol_to_string(symbol: Any) -> str:
    """
    Convert an EncodedSymbol-like object or string into a stable SER token string.
    """
    if isinstance(symbol, str):
        return symbol

    if all(
        hasattr(symbol, attr)
        for attr in ("rhythm", "pitch", "lift", "articulation", "position")
    ):
        return " ".join(
            [
                str(symbol.rhythm),
                str(symbol.pitch),
                str(symbol.lift),
                str(symbol.articulation),
                str(symbol.position),
            ]
        )

    return str(symbol)


def symbols_to_strings(symbols: Iterable[Any]) -> list[str]:
    return [symbol_to_string(symbol) for symbol in symbols]


# ---------------------------------------------------------------------
# ONNX session setup
# ---------------------------------------------------------------------


def build_onnx_providers(use_cuda: bool = True) -> list[Any]:
    """
    Prefer CUDAExecutionProvider, fallback to CPUExecutionProvider.

    TensorRT is deliberately not used for benchmark consistency.
    """
    cuda_options = {
        "device_id": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "arena_extend_strategy": "kNextPowerOfTwo",
    }

    available = ort.get_available_providers()

    if use_cuda and "CUDAExecutionProvider" in available:
        return [
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",
        ]

    return ["CPUExecutionProvider"]


def make_session(model_path: Path, providers: list[Any]) -> ort.InferenceSession:
    """
    Create an ONNX Runtime session with quiet-ish logging.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    session_options = ort.SessionOptions()

    # 2 = WARNING. This hides most info logs but still shows real warnings/errors.
    session_options.log_severity_level = 2

    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
    )


# ---------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------


class HOMRBlackBoxWrapper:
    """
    ONNX-only TrOMR recognition wrapper.

    Main public methods:
        predict_prepared_staff(staff_image) -> list[EncodedSymbol]
        score_query(staff_image, target_symbols) -> float
        encoder_forward(staff_image) -> np.ndarray
        decoder_generate(context) -> list[EncodedSymbol]

    Compatibility aliases:
        predict_staff_crop(...)
        predict_sequence(...)
    """

    REQUIRED_DECODER_INPUTS = {
        "rhythms",
        "pitchs",
        "lifts",
        "articulations",
        "context",
        "cache_len",
    }

    REQUIRED_DECODER_OUTPUTS = {
        "out_rhythms",
        "out_pitchs",
        "out_lifts",
        "out_positions",
        "out_articulations",
        "attention",
    }

    def __init__(
        self,
        model_dir: str | os.PathLike[str] = "models/onnx",
        encoder_onnx_path: str | os.PathLike[str] | None = None,
        decoder_onnx_path: str | os.PathLike[str] | None = None,
        use_cuda: bool = True,
        max_decode_len: int | None = None,
        strict_shape: bool = True,
        strict_import_guard: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)

        self.encoder_onnx_path = (
            Path(encoder_onnx_path)
            if encoder_onnx_path is not None
            else self.model_dir / "tromr_encoder.onnx"
        )

        self.decoder_onnx_path = (
            Path(decoder_onnx_path)
            if decoder_onnx_path is not None
            else self.model_dir / "tromr_decoder.onnx"
        )

        self.use_cuda = bool(use_cuda)
        self.strict_shape = bool(strict_shape)
        self.strict_import_guard = bool(strict_import_guard)

        self.config = Config()

        if self.strict_import_guard:
            assert_no_forbidden_homr_neural_modules_loaded()

        self.max_decode_len = (
            int(max_decode_len)
            if max_decode_len is not None
            else int(self.config.max_seq_len)
        )

        self.providers = build_onnx_providers(use_cuda=self.use_cuda)

        self.encoder_sess = make_session(
            self.encoder_onnx_path,
            providers=self.providers,
        )

        self.decoder_sess = make_session(
            self.decoder_onnx_path,
            providers=self.providers,
        )

        self._extract_metadata()
        self._build_vocab_maps()
        self._validate_sessions()

    # -----------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------

    def _extract_metadata(self) -> None:
        self.encoder_inputs = self.encoder_sess.get_inputs()
        self.encoder_outputs = self.encoder_sess.get_outputs()

        self.decoder_inputs = self.decoder_sess.get_inputs()
        self.decoder_outputs = self.decoder_sess.get_outputs()

        if len(self.encoder_inputs) != 1:
            raise ValueError(
                f"Expected one encoder input, got {len(self.encoder_inputs)}."
            )

        if len(self.encoder_outputs) != 1:
            raise ValueError(
                f"Expected one encoder output, got {len(self.encoder_outputs)}."
            )

        self.encoder_input_name = self.encoder_inputs[0].name
        self.encoder_output_name = self.encoder_outputs[0].name
        self.encoder_input_shape = self.encoder_inputs[0].shape
        self.encoder_output_shape = self.encoder_outputs[0].shape

        self.decoder_input_names = [item.name for item in self.decoder_inputs]
        self.decoder_output_names = [item.name for item in self.decoder_outputs]

        self.cache_input_names = sorted(
            [name for name in self.decoder_input_names if name.startswith("cache_in")],
            key=lambda name: int(name.replace("cache_in", "")),
        )

        self.cache_output_names = sorted(
            [name for name in self.decoder_output_names if name.startswith("cache_out")],
            key=lambda name: int(name.replace("cache_out", "")),
        )

        self.primary_decoder_output_names = [
            "out_rhythms",
            "out_pitchs",
            "out_lifts",
            "out_positions",
            "out_articulations",
            "attention",
        ]

        self.all_decoder_output_names = (
            self.primary_decoder_output_names + self.cache_output_names
        )

    def _build_vocab_maps(self) -> None:
        vocab = self.config.vocab

        self.rhythm_vocab = vocab.rhythm
        self.pitch_vocab = vocab.pitch
        self.lift_vocab = vocab.lift
        self.articulation_vocab = vocab.articulation
        self.position_vocab = vocab.position

        self.inv_rhythm_vocab = {v: k for k, v in self.rhythm_vocab.items()}
        self.inv_pitch_vocab = {v: k for k, v in self.pitch_vocab.items()}
        self.inv_lift_vocab = {v: k for k, v in self.lift_vocab.items()}
        self.inv_articulation_vocab = {
            v: k for k, v in self.articulation_vocab.items()
        }
        self.inv_position_vocab = {v: k for k, v in self.position_vocab.items()}

        self.bos_token = int(self.config.bos_token)
        self.eos_token = int(self.config.eos_token)
        self.nonote_token = int(self.config.nonote_token)

    def _validate_sessions(self) -> None:
        missing_decoder_inputs = (
            self.REQUIRED_DECODER_INPUTS - set(self.decoder_input_names)
        )

        if missing_decoder_inputs:
            raise ValueError(
                "Decoder ONNX model is missing required inputs: "
                + ", ".join(sorted(missing_decoder_inputs))
            )

        missing_decoder_outputs = (
            self.REQUIRED_DECODER_OUTPUTS - set(self.decoder_output_names)
        )

        if missing_decoder_outputs:
            raise ValueError(
                "Decoder ONNX model is missing required outputs: "
                + ", ".join(sorted(missing_decoder_outputs))
            )

        if len(self.cache_input_names) != len(self.cache_output_names):
            raise ValueError(
                "Decoder cache input/output count mismatch: "
                f"{len(self.cache_input_names)} inputs vs "
                f"{len(self.cache_output_names)} outputs."
            )

        expected_cache_count = int(self.config.decoder_depth) * 4

        if len(self.cache_input_names) != expected_cache_count:
            raise ValueError(
                f"Expected {expected_cache_count} decoder cache tensors, "
                f"got {len(self.cache_input_names)}."
            )

        expected_encoder_input = [1, 1, self.config.max_height, self.config.max_width]

        actual_encoder_input = [
            dim if isinstance(dim, int) else dim
            for dim in self.encoder_input_shape
        ]

        if actual_encoder_input != expected_encoder_input:
            raise ValueError(
                f"Unexpected encoder input shape: {self.encoder_input_shape}. "
                f"Expected {expected_encoder_input}."
            )

        self._validate_decoder_vocab_sizes()

    def _validate_decoder_vocab_sizes(self) -> None:
        """
        Check decoder head sizes against HOMR vocabulary sizes.

        User's ONNX metadata should match:
            out_rhythms        -> 260
            out_pitchs         -> 72
            out_lifts          -> 7
            out_positions      -> 3
            out_articulations  -> 144
        """
        expected = {
            "out_rhythms": len(self.rhythm_vocab),
            "out_pitchs": len(self.pitch_vocab),
            "out_lifts": len(self.lift_vocab),
            "out_positions": len(self.position_vocab),
            "out_articulations": len(self.articulation_vocab),
        }

        actual: dict[str, int] = {}

        for output in self.decoder_outputs:
            if output.name not in expected:
                continue

            last_dim = output.shape[-1]

            if not isinstance(last_dim, int):
                raise ValueError(
                    f"Could not statically read vocab dimension for {output.name}: "
                    f"{output.shape}"
                )

            actual[output.name] = last_dim

        for name, expected_size in expected.items():
            actual_size = actual.get(name)

            if actual_size != expected_size:
                raise ValueError(
                    f"Decoder output {name} has size {actual_size}, "
                    f"but HOMR vocabulary has size {expected_size}."
                )

    # -----------------------------------------------------------------
    # Description / debugging
    # -----------------------------------------------------------------

    def describe(self) -> None:
        print("\n=== TrOMR encoder ===")
        print("providers:", self.encoder_sess.get_providers())

        print("inputs:")
        for item in self.encoder_inputs:
            print(" ", item.name, item.shape, item.type)

        print("outputs:")
        for item in self.encoder_outputs:
            print(" ", item.name, item.shape, item.type)

        print("\n=== TrOMR decoder ===")
        print("providers:", self.decoder_sess.get_providers())

        print("inputs:")
        for item in self.decoder_inputs:
            print(" ", item.name, item.shape, item.type)

        print("outputs:")
        for item in self.decoder_outputs:
            print(" ", item.name, item.shape, item.type)

        print("\n=== Vocabulary sizes ===")
        print("rhythm:", len(self.rhythm_vocab))
        print("pitch:", len(self.pitch_vocab))
        print("lift:", len(self.lift_vocab))
        print("position:", len(self.position_vocab))
        print("articulation:", len(self.articulation_vocab))

        print("\n=== Special tokens ===")
        print("BOS:", self.bos_token)
        print("EOS:", self.eos_token)
        print("nonote:", self.nonote_token)
        print("max_decode_len:", self.max_decode_len)

    # -----------------------------------------------------------------
    # Image preprocessing
    # -----------------------------------------------------------------

    def _apply_tromr_transform(self, staff_image: np.ndarray) -> np.ndarray:
        """
        Convert prepared staff image to TrOMR encoder input.

        Input:
            staff_image:
                [256, 1280] float32 [0, 1], or uint8 [0, 255].
                [256, 1280, 1] also accepted.

        Output:
            tensor:
                [1, 1, 256, 1280] float32, normalized.
        """
        arr = np.asarray(staff_image)

        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
            elif arr.shape[2] in (3, 4):
                # The benchmark cache should be grayscale already.
                # If a visual PNG slips through, use the first channel.
                arr = arr[:, :, 0]
            else:
                raise ValueError(f"Unsupported staff image shape: {arr.shape}")

        if arr.ndim != 2:
            raise ValueError(
                f"Expected prepared staff image [H, W] or [H, W, 1], got {arr.shape}"
            )

        expected_shape = (int(self.config.max_height), int(self.config.max_width))

        if self.strict_shape and tuple(arr.shape) != expected_shape:
            raise ValueError(
                f"Expected prepared staff image shape {expected_shape}, got {arr.shape}. "
                "This wrapper expects cached prepare_staff_image(...) outputs."
            )

        if tuple(arr.shape) != expected_shape:
            raise ValueError(
                f"Prepared staff image has shape {arr.shape}; expected {expected_shape}. "
                "Do not resize here during benchmark runs. Regenerate the cache instead."
            )

        arr = arr.astype(np.float32)

        # Cached .npy files are already [0, 1].
        # PNG/debug inputs may be uint8 [0, 255].
        if arr.max(initial=0.0) > 1.5:
            arr = arr / 255.0

        arr = np.clip(arr, 0.0, 1.0)

        arr = arr[np.newaxis, np.newaxis, :, :]

        arr = (arr - 0.7931) / 0.1738

        return np.ascontiguousarray(arr.astype(np.float32))

    # -----------------------------------------------------------------
    # Encoder
    # -----------------------------------------------------------------

    def encoder_forward(self, staff_image: np.ndarray) -> np.ndarray:
        """
        Run tromr_encoder.onnx.

        Returns:
            context [1, 1280, 512] float32.
        """
        x = self._apply_tromr_transform(staff_image)

        outputs = self.encoder_sess.run(
            [self.encoder_output_name],
            {self.encoder_input_name: x},
        )

        context = outputs[0]

        if context.shape != (1, 1280, 512):
            raise ValueError(
                f"Unexpected encoder context shape: {context.shape}. "
                "Expected [1, 1280, 512]."
            )

        return context.astype(np.float32, copy=False)

    # -----------------------------------------------------------------
    # Decoder
    # -----------------------------------------------------------------

    def _init_cache(self, cache_len: int = 0) -> list[np.ndarray]:
        """
        Initialize decoder KV cache tensors.

        The export has 32 cache tensors:
            config.decoder_depth * 4 = 8 * 4 = 32

        Each input cache tensor shape is:
            [1, decoder_heads, cache_len, decoder_dim / decoder_heads]
            [1, 8, cache_len, 64]
        """
        heads = int(self.config.decoder_heads)
        head_dim = int(self.config.decoder_dim) // heads

        cache = [
            np.zeros((1, heads, cache_len, head_dim), dtype=np.float32)
            for _ in self.cache_input_names
        ]

        return cache

    def _decode_id(self, token_id: int, inv_vocab: dict[int, str]) -> str:
        try:
            return inv_vocab[int(token_id)]
        except KeyError as exc:
            raise KeyError(
                f"Token id {token_id} not found in vocabulary of size {len(inv_vocab)}."
            ) from exc

    def _run_decoder_step(
        self,
        *,
        rhythm_token: np.ndarray,
        pitch_token: np.ndarray,
        lift_token: np.ndarray,
        articulation_token: np.ndarray,
        context: np.ndarray,
        cache_len: int,
        cache: list[np.ndarray],
    ) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
        """
        Run one autoregressive decoder step.
        """
        feed: dict[str, np.ndarray] = {
            "rhythms": rhythm_token.astype(np.int64),
            "pitchs": pitch_token.astype(np.int64),
            "lifts": lift_token.astype(np.int64),
            "articulations": articulation_token.astype(np.int64),
            "context": context.astype(np.float32),
            "cache_len": np.array([cache_len], dtype=np.int64),
        }

        if len(cache) != len(self.cache_input_names):
            raise ValueError(
                f"Expected {len(self.cache_input_names)} cache tensors, got {len(cache)}."
            )

        for name, value in zip(self.cache_input_names, cache):
            feed[name] = value.astype(np.float32, copy=False)

        raw_outputs = self.decoder_sess.run(
            self.all_decoder_output_names,
            feed,
        )

        named_outputs = {
            name: value
            for name, value in zip(self.all_decoder_output_names, raw_outputs)
        }

        next_cache = [
            named_outputs[name].astype(np.float32, copy=False)
            for name in self.cache_output_names
        ]

        return named_outputs, next_cache

    def decoder_generate(self, context: np.ndarray) -> list[EncodedSymbol]:
        """
        Greedy autoregressive decoding with the real HOMR multi-stream decoder.

        This mirrors homr.transformer.decoder_inference.ScoreDecoder.generate(...)
        but uses plain ONNX Runtime session.run and NumPy arrays.
        """
        if context.shape != (1, 1280, 512):
            raise ValueError(f"Expected context [1, 1280, 512], got {context.shape}")

        full_context = context.astype(np.float32, copy=False)

        # HOMR decoder uses context[:, :1] after the first step.
        # This is necessary for this ONNX export's cache behavior.
        reduced_context = full_context[:, :1, :]

        rhythm_token = np.array([[self.bos_token]], dtype=np.int64)
        pitch_token = np.array([[self.nonote_token]], dtype=np.int64)
        lift_token = np.array([[self.nonote_token]], dtype=np.int64)
        articulation_token = np.array([[self.nonote_token]], dtype=np.int64)

        cache = self._init_cache(cache_len=0)

        symbols: list[EncodedSymbol] = []

        for step in range(self.max_decode_len):
            step_context = full_context if step == 0 else reduced_context

            outputs, cache = self._run_decoder_step(
                rhythm_token=rhythm_token,
                pitch_token=pitch_token,
                lift_token=lift_token,
                articulation_token=articulation_token,
                context=step_context,
                cache_len=step,
                cache=cache,
            )

            out_rhythms = outputs["out_rhythms"]
            out_pitchs = outputs["out_pitchs"]
            out_lifts = outputs["out_lifts"]
            out_positions = outputs["out_positions"]
            out_articulations = outputs["out_articulations"]
            attention = outputs["attention"]

            rhythm_id = int(np.argmax(out_rhythms[:, -1, :], axis=-1)[0])
            pitch_id = int(np.argmax(out_pitchs[:, -1, :], axis=-1)[0])
            lift_id = int(np.argmax(out_lifts[:, -1, :], axis=-1)[0])
            position_id = int(np.argmax(out_positions[:, -1, :], axis=-1)[0])
            articulation_id = int(np.argmax(out_articulations[:, -1, :], axis=-1)[0])

            if rhythm_id == self.eos_token:
                break

            rhythm = self._decode_id(rhythm_id, self.inv_rhythm_vocab)
            pitch = self._decode_id(pitch_id, self.inv_pitch_vocab)
            lift = self._decode_id(lift_id, self.inv_lift_vocab)
            position = self._decode_id(position_id, self.inv_position_vocab)
            articulation = self._decode_id(
                articulation_id,
                self.inv_articulation_vocab,
            )

            attention_flat = np.asarray(attention).reshape(-1)

            coordinates: tuple[float, float] | None

            if attention_flat.size >= 2:
                coordinates = (
                    float(attention_flat[0]),
                    float(attention_flat[1]),
                )
            else:
                coordinates = None

            symbol = EncodedSymbol(
                rhythm=rhythm,
                pitch=pitch,
                lift=lift,
                articulation=articulation,
                position=position,
                coordinates=coordinates,
            )

            symbols.append(symbol)

            rhythm_token = np.array([[rhythm_id]], dtype=np.int64)
            pitch_token = np.array([[pitch_id]], dtype=np.int64)
            lift_token = np.array([[lift_id]], dtype=np.int64)
            articulation_token = np.array([[articulation_id]], dtype=np.int64)

        return symbols

    # -----------------------------------------------------------------
    # Public prediction/scoring API
    # -----------------------------------------------------------------

    def predict_prepared_staff(self, staff_image: np.ndarray) -> list[EncodedSymbol]:
        """
        Predict symbols from one cached prepared staff image.

        This is the main Track B interface.
        """
        context = self.encoder_forward(staff_image)
        return self.decoder_generate(context)

    def predict_staff_crop(self, staff_crop: np.ndarray) -> list[EncodedSymbol]:
        """
        Backward-compatible alias.

        Despite the old name, this expects a prepared staff image, not a raw crop.
        """
        return self.predict_prepared_staff(staff_crop)

    def predict_sequence(self, staff_image: np.ndarray) -> list[EncodedSymbol]:
        """
        Backward-compatible alias.
        """
        return self.predict_prepared_staff(staff_image)

    def score_query(
        self,
        staff_image: np.ndarray,
        target_symbols: list[Any],
    ) -> float:
        """
        Black-box score for Square Attack.

        Higher score = more divergent from target = better adversarial candidate.
        """
        pred_symbols = self.predict_prepared_staff(staff_image)

        pred_tokens = symbols_to_strings(pred_symbols)
        target_tokens = symbols_to_strings(target_symbols)

        denom = max(len(target_tokens), 1)
        distance = levenshtein_distance(pred_tokens, target_tokens)

        return float(distance / denom)

    def predict_file(self, path: str | os.PathLike[str]) -> list[EncodedSymbol]:
        """
        Convenience helper for cached .npy staff images.
        """
        staff_image = np.load(path).astype(np.float32)
        return self.predict_prepared_staff(staff_image)


# Common aliases so older scripts do not break.
HomrWrapper = HOMRBlackBoxWrapper
HOMRWrapper = HOMRBlackBoxWrapper


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
def find_first_cached_staff(
    cache_root: str | os.PathLike[str] = "dataset/cached_prepared_staffs",
    staff_index: int = 0,
) -> Path:
    """
    Find the first cached prepared staff .npy file.

    Default:
        dataset/cached_prepared_staffs/<first_score_folder>/staff_000.npy

    Falls back to the first staff_*.npy if staff_000.npy does not exist.
    """
    cache_root = Path(cache_root)

    if not cache_root.exists():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")

    preferred = sorted(cache_root.glob(f"*/staff_{staff_index:03d}.npy"))

    if preferred:
        return preferred[0]

    fallback = sorted(cache_root.glob("*/staff_*.npy"))

    if fallback:
        return fallback[0]

    raise FileNotFoundError(
        f"No cached staff .npy files found under {cache_root}. "
        "Run dataset/cache_prepared_staffs.py first."
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test ONNX-only TrOMR wrapper on one cached prepared staff."
    )

    parser.add_argument(
        "staff_npy",
        type=Path,
        nargs="?",
        help="Path to cached prepared staff .npy file.",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/onnx"),
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPUExecutionProvider.",
    )

    parser.add_argument(
        "--max-decode-len",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print ONNX metadata and vocabulary sizes.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print prediction as JSON.",
    )
    
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("dataset/cached_prepared_staffs"),
        help="Root directory for cached prepared staffs.",
    )

    parser.add_argument(
        "--staff-index",
        type=int,
        default=0,
        help="Preferred staff index for auto-discovery, e.g. 0 means staff_000.npy.",
    )

    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="Print ONNX metadata and exit without running prediction.",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[init] Loading HOMRBlackBoxWrapper...")

    wrapper = HOMRBlackBoxWrapper(
        model_dir=args.model_dir,
        use_cuda=not args.cpu,
        max_decode_len=args.max_decode_len,
    )

    print("[init] Wrapper loaded.")

    if args.describe or args.describe_only:
        wrapper.describe()

    if args.describe_only:
        return

    if args.staff_npy is None:
        staff_npy = find_first_cached_staff(
            cache_root=args.cache_root,
            staff_index=args.staff_index,
        )
        print(f"[auto] Using cached staff: {staff_npy}")
    else:
        staff_npy = args.staff_npy

    symbols = wrapper.predict_file(staff_npy)

    if args.json:
        payload = [
            {
                "rhythm": symbol.rhythm,
                "pitch": symbol.pitch,
                "lift": symbol.lift,
                "articulation": symbol.articulation,
                "position": symbol.position,
                "coordinates": symbol.coordinates,
                "string": str(symbol),
            }
            for symbol in symbols
        ]

        print(json.dumps(payload, indent=2))
    else:
        print(f"Predicted {len(symbols)} symbols:")
        for symbol in symbols:
            print(symbol)


if __name__ == "__main__":
    main()