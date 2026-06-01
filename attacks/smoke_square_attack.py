"""
Smoke test for Track B Square Attack on one cached prepared staff image.

This version uses clean ONNX prediction as an in-memory pseudo-target.
It does not require metadata.json to contain gt_tokens.

This is a self-consistency adversarial smoke test, not final ground-truth
benchmark scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from attacks.src.homr_wrapper import HomrWrapper
from attacks.src.square_attack import run_square_attack


PREPARED_STAFFS_DIR = Path("dataset/cached_prepared_staffs")


def find_first_cached_staff() -> tuple[Path, Path]:
    """
    Return:
        staff_npy_path, metadata_path
    """
    if not PREPARED_STAFFS_DIR.exists():
        raise FileNotFoundError(
            f"Missing {PREPARED_STAFFS_DIR}. "
            "Run dataset/cache_prepared_staffs.py first."
        )

    score_dirs = sorted(p for p in PREPARED_STAFFS_DIR.iterdir() if p.is_dir())

    if not score_dirs:
        raise FileNotFoundError(
            f"No score subdirectories found in {PREPARED_STAFFS_DIR}. "
            "Run dataset/cache_prepared_staffs.py first."
        )

    for score_dir in score_dirs:
        metadata_path = score_dir / "metadata.json"

        if not metadata_path.exists():
            continue

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        staffs = metadata.get("staffs", [])

        for staff_info in staffs:
            filename_npy = staff_info.get("filename_npy")

            if not filename_npy:
                continue

            staff_npy_path = score_dir / filename_npy

            if staff_npy_path.exists():
                return staff_npy_path, metadata_path

    raise FileNotFoundError(
        "Could not find any cached staff_*.npy referenced by metadata.json."
    )


def load_staff_image(staff_npy_path: Path) -> np.ndarray:
    staff_image = np.load(staff_npy_path).astype(np.float32)

    if staff_image.max() > 1.0:
        staff_image = staff_image / 255.0

    return np.clip(staff_image, 0.0, 1.0).astype(np.float32)


def linf_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def main() -> None:
    staff_npy_path, metadata_path = find_first_cached_staff()

    print(f"Using staff image: {staff_npy_path}")
    print(f"Using metadata:    {metadata_path}")

    staff_image = load_staff_image(staff_npy_path)

    print(f"Staff shape:       {staff_image.shape}")
    print(f"Staff dtype:       {staff_image.dtype}")
    print(f"Staff min/max:     {staff_image.min():.6f} / {staff_image.max():.6f}")

    wrapper = HomrWrapper()

    # Self-consistency target:
    # clean model prediction becomes the target sequence for this smoke test.
    target_tokens: list[Any] = wrapper.predict_prepared_staff(staff_image)

    print(f"Target source:     clean_onnx_prediction")
    print(f"Target token count:{len(target_tokens)}")

    if len(target_tokens) == 0:
        raise RuntimeError(
            "Clean ONNX prediction returned zero tokens. "
            "The attack can run mechanically, but this staff is not useful "
            "for a self-consistency smoke test."
        )

    clean_score = wrapper.score_query(staff_image, target_tokens)
    print(f"Clean score_query: {clean_score:.6f}")

    epsilon = 0.02
    n_max = 50

    result = run_square_attack(
        staff_image=staff_image,
        target_tokens=target_tokens,
        wrapper=wrapper,
        epsilon=epsilon,
        n_max=n_max,
        p_init=0.8,
        p_final=0.05,
        seed=123,
    )

    x_adv = result["x_adv"]
    linf = linf_distance(x_adv, staff_image)

    print()
    print("Square Attack smoke result")
    print("--------------------------")
    print(f"L_best:                  {result['L_best']:.6f}")
    print(f"n_queries:               {result['n_queries']}")
    print(f"loss_trajectory length:  {len(result['loss_trajectory'])}")
    print(f"L_inf distance:          {linf:.8f}")

    assert 1 <= result["n_queries"] <= n_max
    assert result["L_best"] >= result["loss_trajectory"][0]
    assert all(
        a <= b
        for a, b in zip(
            result["loss_trajectory"],
            result["loss_trajectory"][1:],
            strict=False,
        )
    )
    assert linf <= epsilon + 1e-6

    print()
    print("smoke_square_attack.py passed")


if __name__ == "__main__":
    main()