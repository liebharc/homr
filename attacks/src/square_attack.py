"""
Black-box Square Attack for cached HOMR-prepared staff images.

Purpose
-------
Implements the Track B adversarial optimization loop for the adversarial HOMR
benchmark. The attack perturbs cached prepared staff images, not full-page sheet
music images and not raw staff-band crops.

Inputs
------
- staff_image:
    HOMR-prepared TrOMR staff image, normally produced by
    homr.staff_parsing.prepare_staff_image(...), shape [256, 1280] or
    [256, 1280, 1], float32 in [0, 1].
- target_tokens:
    Ground-truth token/symbol sequence for the staff.
- wrapper:
    ONNX-only recognition wrapper exposing score_query(staff_image, target_tokens).
- epsilon:
    L-infinity perturbation budget in [0, 1] image space.
- n_max:
    Maximum total score-query budget, including the initial query.
- p_init, p_final:
    Initial and final square side fractions relative to image width.

Outputs
-------
run_square_attack(...) returns:

{
    "x_adv": np.ndarray,
    "L_best": float,
    "n_queries": int,
    "loss_trajectory": list[float]
}

Benchmark boundary
------------------
This module performs no neural inference directly. It delegates recognition only
through wrapper.score_query(...). In benchmark mode, that wrapper must use ONNX
Runtime for TrOMR encoder/decoder execution. This file must not import or call
Staff2Score.predict, Encoder, get_decoder, parse_staff_tromr, PyTorch, or
TensorFlow.

Important restriction
---------------------
Track B attacks already-prepared staff images. Do not pass x_adv back through
the full HOMR layout pipeline. x_adv should be evaluated with
wrapper.predict_prepared_staff(...) or wrapper.score_query(...).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SquareAttackConfig:
    """Configuration for one Square Attack run."""

    epsilon: float
    n_max: int
    p_init: float
    p_final: float
    seed: int | None = None


def _validate_config(config: SquareAttackConfig) -> None:
    """Validate attack hyperparameters before querying the wrapper."""
    if not np.isfinite(config.epsilon):
        raise ValueError("epsilon must be finite")

    if config.epsilon < 0.0 or config.epsilon > 1.0:
        raise ValueError("epsilon must be in [0, 1] float image space")

    if config.n_max < 1:
        raise ValueError("n_max must be at least 1")

    if not (0.0 < config.p_final <= config.p_init <= 1.0):
        raise ValueError("Require 0 < p_final <= p_init <= 1")


def _coerce_staff_image(staff_image: np.ndarray) -> np.ndarray:
    """
    Convert an input staff image to float32 [H, W] in [0, 1].

    Accepts [H, W] or [H, W, 1]. Rejects RGB-like inputs because Track B should
    operate on grayscale prepared TrOMR staff images.
    """
    arr = np.asarray(staff_image)

    if arr.ndim == 3:
        if arr.shape[-1] != 1:
            raise ValueError(
                f"Expected grayscale prepared staff image with shape [H, W] "
                f"or [H, W, 1], got {arr.shape}"
            )
        arr = arr[:, :, 0]

    if arr.ndim != 2:
        raise ValueError(
            f"Expected prepared staff image with shape [H, W] or [H, W, 1], got {arr.shape}"
        )

    arr = arr.astype(np.float32, copy=True)

    if arr.size == 0:
        raise ValueError("staff_image is empty")

    if not np.all(np.isfinite(arr)):
        raise ValueError("staff_image contains NaN or infinite values")

    if arr.max() > 1.0:
        arr /= 255.0

    arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    return arr


def _restore_shape(x_2d: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Return adversarial image with the same rank convention as the input."""
    if len(original_shape) == 3 and original_shape[-1] == 1:
        return x_2d[:, :, np.newaxis].astype(np.float32, copy=False)
    return x_2d.astype(np.float32, copy=False)


def _get_square_size(
    step: int,
    n_max: int,
    p_init: float,
    p_final: float,
    width: int,
) -> int:
    """
    Geometrically decay square size over the query budget.

    step is 1-indexed for candidate queries. The side length is measured as a
    fraction of image width, following the benchmark blueprint.
    """
    if n_max <= 1:
        return max(1, round(width * p_final))

    t = min(max(step / n_max, 0.0), 1.0)
    p = p_init * (p_final / p_init) ** t
    return max(1, min(width, round(width * p)))


def _linf_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return L-infinity distance between two arrays."""
    return float(np.max(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def _project_to_linf_ball(
    candidate: np.ndarray,
    clean: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Project candidate into [clean - epsilon, clean + epsilon] and image range [0, 1]."""
    lower = np.maximum(clean - epsilon, 0.0)
    upper = np.minimum(clean + epsilon, 1.0)
    return np.clip(candidate, lower, upper).astype(np.float32, copy=False)


def _score(
    wrapper: Any,
    staff_image_2d: np.ndarray,
    target_tokens: list[Any],
    original_shape: tuple[int, ...],
) -> float:
    """
    Evaluate attack objective through the wrapper.

    Higher returned value means greater divergence from the target sequence.
    """
    query_image = _restore_shape(staff_image_2d, original_shape)
    return float(wrapper.score_query(query_image, target_tokens))


def run_square_attack(
    staff_image: np.ndarray,
    target_tokens: list[Any],
    wrapper: Any,
    epsilon: float,
    n_max: int,
    p_init: float,
    p_final: float,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Run black-box Square Attack on one prepared staff image.

    Parameters
    ----------
    staff_image:
        Prepared staff image in [0, 1], shape [256, 1280] or [256, 1280, 1].
    target_tokens:
        Ground-truth token sequence.
    wrapper:
        Object exposing score_query(staff_image, target_tokens) -> float.
    epsilon:
        L-infinity perturbation budget in [0, 1].
    n_max:
        Maximum total query count, including the initial query.
    p_init:
        Initial square side fraction of image width.
    p_final:
        Final square side fraction of image width.
    seed:
        Optional RNG seed for reproducible attack trajectories.

    Returns
    -------
    dict
        {
            "x_adv": np.ndarray,
            "L_best": float,
            "n_queries": int,
            "loss_trajectory": list[float]
        }
    """
    config = SquareAttackConfig(
        epsilon=float(epsilon),
        n_max=int(n_max),
        p_init=float(p_init),
        p_final=float(p_final),
        seed=seed,
    )
    _validate_config(config)

    if not hasattr(wrapper, "score_query"):
        raise TypeError("wrapper must expose score_query(staff_image, target_tokens)")

    original_shape = tuple(np.asarray(staff_image).shape)
    x_clean = _coerce_staff_image(staff_image)
    height, width = x_clean.shape

    rng = np.random.default_rng(config.seed)

    if config.epsilon == 0.0:
        l_clean = _score(wrapper, x_clean, target_tokens, original_shape)
        return {
            "x_adv": _restore_shape(x_clean, original_shape),
            "L_best": float(l_clean),
            "n_queries": 1,
            "loss_trajectory": [float(l_clean)],
        }

    # Initialize on the boundary of the L-infinity ball.
    init_signs = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=x_clean.shape,
    )
    x_best = _project_to_linf_ball(
        x_clean + config.epsilon * init_signs,
        x_clean,
        config.epsilon,
    )

    l_best = _score(wrapper, x_best, target_tokens, original_shape)
    n_queries = 1
    loss_trajectory: list[float] = [float(l_best)]

    # Candidate queries. n_max is total query budget, so loop stops at n_max.
    while n_queries < config.n_max:
        step = n_queries
        square_size = _get_square_size(
            step=step,
            n_max=config.n_max,
            p_init=config.p_init,
            p_final=config.p_final,
            width=width,
        )

        square_size = min(square_size, height, width)

        max_r = height - square_size
        max_c = width - square_size

        r = int(rng.integers(0, max_r + 1))
        c = int(rng.integers(0, max_c + 1))

        patch_sign = float(rng.choice(np.array([-1.0, 1.0], dtype=np.float32)))

        x_candidate = x_best.copy()

        # Set selected square to a boundary value relative to the clean image.
        # This keeps the candidate inside the L-infinity attack domain while
        # making a meaningful zero-order coordinate update.
        x_candidate[
            r : r + square_size,
            c : c + square_size,
        ] = x_clean[
            r : r + square_size,
            c : c + square_size,
        ] + patch_sign * config.epsilon

        x_candidate = _project_to_linf_ball(
            x_candidate,
            x_clean,
            config.epsilon,
        )

        l_new = _score(wrapper, x_candidate, target_tokens, original_shape)
        n_queries += 1

        # Greedy accept if the black-box loss increases.
        if l_new > l_best:
            x_best = x_candidate
            l_best = float(l_new)
            loss_trajectory.append(float(l_best))

    x_adv = _restore_shape(x_best, original_shape)

    # Final defensive invariant: never return an invalid perturbation.
    linf = _linf_distance(_coerce_staff_image(x_adv), x_clean)
    if linf > config.epsilon + 1e-6:
        raise RuntimeError(
            f"Internal error: returned adversarial image violates L_inf budget. "
            f"distance={linf}, epsilon={config.epsilon}"
        )

    return {
        "x_adv": x_adv,
        "L_best": float(l_best),
        "n_queries": int(n_queries),
        "loss_trajectory": loss_trajectory,
    }


class _DummyWrapper:
    """
    Deterministic wrapper used only for local self-test.

    This does not simulate HOMR. It simply makes score_query increase when the
    image mean moves away from the clean baseline, allowing the attack mechanics
    to be tested without ONNX models.
    """

    def __init__(self, clean: np.ndarray) -> None:
        self.clean_mean = float(np.mean(_coerce_staff_image(clean)))

    def score_query(self, staff_image: np.ndarray, target_tokens: list[Any]) -> float:
        del target_tokens
        x = _coerce_staff_image(staff_image)
        return float(abs(float(np.mean(x)) - self.clean_mean))


def _self_test() -> None:
    """Run lightweight tests for direct CLI execution."""
    clean = np.full((256, 1280), 0.75, dtype=np.float32)
    wrapper = _DummyWrapper(clean)

    result = run_square_attack(
        staff_image=clean,
        target_tokens=["clef_G", "note_C4_quarter"],
        wrapper=wrapper,
        epsilon=0.02,
        n_max=50,
        p_init=0.8,
        p_final=0.05,
        seed=123,
    )

    assert set(result) == {"x_adv", "L_best", "n_queries", "loss_trajectory"}
    assert result["x_adv"].shape == clean.shape
    assert result["x_adv"].dtype == np.float32
    assert 1 <= result["n_queries"] <= 50
    assert result["L_best"] >= result["loss_trajectory"][0]
    assert all(
        a <= b
        for a, b in zip(
            result["loss_trajectory"],
            result["loss_trajectory"][1:],
            strict=False,
        )
    )
    assert _linf_distance(result["x_adv"], clean) <= 0.020001

    zero_eps = run_square_attack(
        staff_image=clean,
        target_tokens=[],
        wrapper=wrapper,
        epsilon=0.0,
        n_max=10,
        p_init=0.8,
        p_final=0.05,
        seed=123,
    )

    assert zero_eps["n_queries"] == 1
    assert np.array_equal(zero_eps["x_adv"], clean)

    print("square_attack.py self-test passed")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a lightweight Square Attack mechanics self-test."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic local self-test. This is the default when no args are given.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    parser.parse_args()
    _self_test()


if __name__ == "__main__":
    main()