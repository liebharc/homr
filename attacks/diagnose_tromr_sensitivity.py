"""
Diagnose whether the ONNX TrOMR wrapper is sensitive to image perturbations.

This is NOT a benchmark sweep. It is a debugging script for Track B.

Run from repository root:

    python attacks/diagnose_tromr_sensitivity.py

or:

    python attacks/diagnose_tromr_sensitivity.py --staff dataset/cached_prepared_staffs/.../staff_000.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from attacks.src.homr_wrapper import HomrWrapper, symbols_to_strings
from attacks.src.statistics_engine import symbol_error_rate, character_error_rate


def find_first_cached_staff(cache_root: Path) -> Path:
    preferred = sorted(cache_root.glob("*/staff_000.npy"))
    if preferred:
        return preferred[0]

    fallback = sorted(cache_root.glob("*/staff_*.npy"))
    if fallback:
        return fallback[0]

    raise FileNotFoundError(
        f"No cached staff_*.npy files found under {cache_root}. "
        "Run dataset/cache_prepared_staffs.py first."
    )


def load_staff(path: Path) -> np.ndarray:
    x = np.load(path).astype(np.float32)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[:, :, 0]

    if x.ndim != 2:
        raise ValueError(f"Expected [H, W] staff image, got {x.shape}")

    if x.max(initial=0.0) > 1.5:
        x = x / 255.0

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def center_square_variant(x: np.ndarray, value: float, fraction: float) -> np.ndarray:
    y = x.copy()
    h, w = y.shape
    side = max(1, int(round(min(h, w) * fraction)))
    r0 = (h - side) // 2
    c0 = (w - side) // 2
    y[r0:r0 + side, c0:c0 + side] = value
    return y


def wide_band_variant(x: np.ndarray, value: float, height_fraction: float) -> np.ndarray:
    y = x.copy()
    h, _ = y.shape
    band_h = max(1, int(round(h * height_fraction)))
    r0 = (h - band_h) // 2
    y[r0:r0 + band_h, :] = value
    return y


def random_linf_variant(x: np.ndarray, epsilon: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=x.shape)
    return np.clip(x + epsilon * signs, 0.0, 1.0).astype(np.float32)


def gaussian_variant(x: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return np.clip(x + noise, 0.0, 1.0).astype(np.float32)


def token_preview(tokens: list[str], n: int = 10) -> str:
    if not tokens:
        return "<EMPTY>"
    return " | ".join(tokens[:n])


def evaluate_variant(
    *,
    name: str,
    image: np.ndarray,
    wrapper: Any,
    clean_tokens: list[str],
) -> dict[str, Any]:
    del name
    symbols = wrapper.predict_prepared_staff(image)
    tokens = symbols_to_strings(symbols)

    ser = symbol_error_rate(tokens, clean_tokens)
    cer = character_error_rate(" ".join(tokens), " ".join(clean_tokens))

    return {
        "n_symbols": len(symbols),
        "ser_vs_clean": ser,
        "cer_vs_clean": cer,
        "tokens": tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose TrOMR sensitivity under obvious image corruptions."
    )

    parser.add_argument(
        "--staff",
        type=Path,
        default=None,
        help="Path to cached staff .npy. If omitted, first cached staff is used.",
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("dataset/cached_prepared_staffs"),
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
        "--seed",
        type=int,
        default=123,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    staff_path = args.staff if args.staff is not None else find_first_cached_staff(args.cache_root)

    print(f"[staff] {staff_path}")
    x = load_staff(staff_path)
    print(
        f"[image] shape={x.shape} dtype={x.dtype} "
        f"min={float(x.min()):.6f} max={float(x.max()):.6f} mean={float(x.mean()):.6f}"
    )

    print("[init] Loading ONNX wrapper...")
    wrapper = HomrWrapper(
        model_dir=args.model_dir,
        use_cuda=not args.cpu,
        max_decode_len=args.max_decode_len,
    )
    print("[init] Wrapper loaded.")

    clean_symbols = wrapper.predict_prepared_staff(x)
    clean_tokens = symbols_to_strings(clean_symbols)

    print("\n[clean]")
    print(f"  n_symbols={len(clean_symbols)}")
    print(f"  first_tokens={token_preview(clean_tokens)}")

    variants: list[tuple[str, np.ndarray]] = [
        ("all_black", np.zeros_like(x, dtype=np.float32)),
        ("all_white", np.ones_like(x, dtype=np.float32)),
        ("inverted", 1.0 - x),
        ("random_linf_eps_0.10", random_linf_variant(x, epsilon=0.10, seed=args.seed)),
        ("random_linf_eps_0.30", random_linf_variant(x, epsilon=0.30, seed=args.seed + 1)),
        ("gaussian_sigma_0.10", gaussian_variant(x, sigma=0.10, seed=args.seed + 2)),
        ("center_square_black_50pct", center_square_variant(x, value=0.0, fraction=0.50)),
        ("center_square_white_50pct", center_square_variant(x, value=1.0, fraction=0.50)),
        ("wide_black_band_50pct", wide_band_variant(x, value=0.0, height_fraction=0.50)),
        ("wide_white_band_50pct", wide_band_variant(x, value=1.0, height_fraction=0.50)),
    ]

    print("\n[variants]")
    any_changed = False

    for name, image in variants:
        result = evaluate_variant(
            name=name,
            image=image,
            wrapper=wrapper,
            clean_tokens=clean_tokens,
        )

        changed = result["ser_vs_clean"] > 0.0
        any_changed = any_changed or changed

        print(f"\n{name}")
        print(f"  changed={changed}")
        print(f"  n_symbols={result['n_symbols']}")
        print(f"  SER_vs_clean={result['ser_vs_clean']:.6f}")
        print(f"  CER_vs_clean={result['cer_vs_clean']:.6f}")
        print(f"  first_tokens={token_preview(result['tokens'])}")

    print("\n[diagnosis]")
    if any_changed:
        print(
            "At least one obvious perturbation changed the decoded sequence. "
            "The wrapper is image-sensitive; Square Attack likely needs stronger budgets, "
            "more queries, or a denser loss."
        )
    else:
        print(
            "No obvious perturbation changed the decoded sequence. "
            "Do not scale Square Attack yet. Investigate ONNX decoder generation, "
            "image preprocessing, max_decode_len, or token comparison."
        )


if __name__ == "__main__":
    main()
