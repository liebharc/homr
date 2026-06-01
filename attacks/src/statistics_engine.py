"""
Metric utilities for the adversarial HOMR benchmark.

This module is benchmark-valid. It contains no neural inference and does not
call HOMR prediction code, PyTorch, TensorFlow, or ONNX Runtime.

Purpose
-------
Provide centralized Symbol Error Rate (SER), Character Error Rate (CER), raw
Levenshtein distance, and batch aggregation utilities used by:

- homr_wrapper.py for score_query(...)
- square_attack.py for black-box loss evaluation
- run_square_sweep.py for Track B reporting
- run_spectral_sweep.py for Track A reporting

Inputs
------
SER accepts predicted and ground-truth musical token sequences.
CER accepts predicted and ground-truth strings.
batch_metrics accepts a list of dictionaries containing metric values.

Outputs
-------
All public metric functions return plain Python floats or ints.

Benchmark boundary
------------------
This file is purely deterministic metric code. It is safe for final benchmark
use and does not violate the ONNX-only neural inference boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import mean, pstdev
from typing import Any


try:
    import editdistance as _editdistance
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    _editdistance = None


TokenLike = str | int | float | object


def _as_sequence(value: Any) -> list[Any]:
    """Convert a token/string-like object into a list suitable for edit distance."""
    if value is None:
        return []

    if isinstance(value, str):
        # For SER, strings are interpreted as whitespace-tokenized sequences.
        return value.split()

    if isinstance(value, Sequence):
        return list(value)

    if isinstance(value, Iterable):
        return list(value)

    return [value]


def _normalize_token(token: Any) -> str:
    """
    Normalize a token-like object into a stable string representation.

    EncodedSymbol-like HOMR objects may have useful __str__ output. For dict-like
    objects, sort keys for deterministic comparison.
    """
    if token is None:
        return ""

    if isinstance(token, str):
        return token

    if isinstance(token, dict):
        return repr(sorted(token.items()))

    return str(token)


def _fallback_levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    """
    Compute Levenshtein distance with O(min(len(a), len(b))) memory.

    This fallback avoids making smoke tests fail solely because the optional
    editdistance package is missing.
    """
    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (item_a != item_b)
            current.append(min(insertion, deletion, substitution))
        previous = current

    return previous[-1]


def levenshtein_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """
    Return Levenshtein edit distance between two sequences.

    Parameters
    ----------
    a:
        Predicted sequence.
    b:
        Ground-truth sequence.

    Returns
    -------
    int
        Minimum number of insertions, deletions, and substitutions.
    """
    a_list = list(a)
    b_list = list(b)

    if _editdistance is not None:
        return int(_editdistance.eval(a_list, b_list))

    return _fallback_levenshtein(a_list, b_list)


def symbol_error_rate(pred_tokens: Any, gt_tokens: Any) -> float:
    """
    Compute normalized Symbol Error Rate.

    SER = Levenshtein(pred_tokens, gt_tokens) / max(len(gt_tokens), 1)

    Empty prediction against non-empty ground truth returns 1.0 when the edit
    distance equals the ground-truth length. Empty prediction against empty
    ground truth returns 0.0.
    """
    pred_seq = [_normalize_token(token) for token in _as_sequence(pred_tokens)]
    gt_seq = [_normalize_token(token) for token in _as_sequence(gt_tokens)]

    if not pred_seq and not gt_seq:
        return 0.0

    distance = levenshtein_distance(pred_seq, gt_seq)
    return float(distance / max(len(gt_seq), 1))


def character_error_rate(pred_str: Any, gt_str: Any) -> float:
    """
    Compute normalized Character Error Rate.

    CER = Levenshtein(list(pred_str), list(gt_str)) / max(len(gt_str), 1)
    """
    pred = "" if pred_str is None else str(pred_str)
    gt = "" if gt_str is None else str(gt_str)

    if pred == "" and gt == "":
        return 0.0

    distance = levenshtein_distance(list(pred), list(gt))
    return float(distance / max(len(gt), 1))


def _extract_metric(result: dict[str, Any], *names: str) -> float | None:
    """Return the first available metric value from a result dictionary."""
    for name in names:
        if name in result and result[name] is not None:
            return float(result[name])
    return None


def batch_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    """
    Aggregate SER/CER metrics across benchmark result rows.

    Accepted per-row keys:
    - ser or symbol_error_rate
    - cer or character_error_rate

    Returns
    -------
    dict
        {
            "mean_ser": float,
            "std_ser": float,
            "mean_cer": float,
            "std_cer": float,
            "n": float
        }
    """
    ser_values: list[float] = []
    cer_values: list[float] = []

    for result in results:
        ser = _extract_metric(result, "ser", "symbol_error_rate", "SER")
        cer = _extract_metric(result, "cer", "character_error_rate", "CER")

        if ser is not None:
            ser_values.append(ser)
        if cer is not None:
            cer_values.append(cer)

    return {
        "mean_ser": float(mean(ser_values)) if ser_values else 0.0,
        "std_ser": float(pstdev(ser_values)) if len(ser_values) > 1 else 0.0,
        "mean_cer": float(mean(cer_values)) if cer_values else 0.0,
        "std_cer": float(pstdev(cer_values)) if len(cer_values) > 1 else 0.0,
        "n": float(len(results)),
    }


# Backward-compatible aliases for older local code.
calculate_ser = symbol_error_rate
calculate_cer = character_error_rate


def _self_test() -> None:
    """Run small validation checks for direct CLI execution."""
    assert symbol_error_rate(["a", "b"], ["a", "b"]) == 0.0
    assert symbol_error_rate([], ["a", "b"]) == 1.0
    assert symbol_error_rate(["a", "x"], ["a", "b"]) == 0.5
    assert symbol_error_rate(["a", "b", "c"], ["a", "b"]) == 0.5

    assert character_error_rate("abc", "abc") == 0.0
    assert character_error_rate("", "abc") == 1.0
    assert character_error_rate("axc", "abc") == 1 / 3

    metrics = batch_metrics(
        [
            {"ser": 0.0, "cer": 0.1},
            {"ser": 0.5, "cer": 0.3},
            {"symbol_error_rate": 1.0, "character_error_rate": 0.5},
        ]
    )
    assert metrics["mean_ser"] == 0.5
    assert metrics["mean_cer"] == 0.3
    assert metrics["std_ser"] > 0.0
    assert metrics["std_cer"] > 0.0
    assert metrics["n"] == 3.0

    print("statistics_engine.py self-test passed")


if __name__ == "__main__":
    _self_test()