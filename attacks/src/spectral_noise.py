"""
Spectral noise utilities for Track A of the adversarial HOMR benchmark.

Track A evaluates natural out-of-distribution degradation on full-page sheet
music images by injecting frequency-shaped noise.

Noise colors:
    alpha = 0.0  -> white noise
    alpha = 1.0  -> pink noise
    alpha = 2.0  -> brown noise

The perturbation is applied in float image space [0, 1]:

    X_phys = clip(X + epsilon_phys * C, 0, 1)

where C is a zero-mean, unit-variance colored noise field.

This file contains no neural inference and is benchmark-safe.

CLI examples
------------
Self-test:

    python attacks/src/spectral_noise.py --self-test

Auto-preview first image in dataset/images:

    python attacks/src/spectral_noise.py --preview

Preview first 5 images:

    python attacks/src/spectral_noise.py --preview --max-images 5

Preview specific image:

    python attacks/src/spectral_noise.py dataset/images/some_image.png --preview
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def _validate_image_array(image: np.ndarray) -> np.ndarray:
    """
    Convert image to float32 [0, 1], preserving shape.

    Accepts:
        [H, W]
        [H, W, C]

    Returns:
        float32 array in [0, 1].
    """
    arr = np.asarray(image)

    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected image with shape [H, W] or [H, W, C], got {arr.shape}")

    if arr.size == 0:
        raise ValueError("Image is empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Image contains NaN or infinite values.")

    arr = arr.astype(np.float32, copy=False)

    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


COMPONENT_ALPHAS = (0.0, 1.0, 2.0)
COMPONENT_NAMES = ("white", "pink", "brown")
DEFAULT_WEIGHT_TEMPERATURE = 2.0
DEFAULT_WEIGHT_CONCENTRATION = 16.0


def generate_colored_noise(
    height: int,
    width: int,
    alpha: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate one 2D colored-noise component with approximate 1/f^alpha shaping.

    This lower-level function remains useful for diagnostics. Track A's default
    injection path uses generate_multiscale_colored_noise(...), which mixes
    white/pink/brown components with alpha-conditioned random weights.
    """
    height = int(height)
    width = int(width)
    alpha = float(alpha)

    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid noise shape: height={height}, width={width}")

    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite.")

    rng = np.random.default_rng(seed)

    # Use float64 internally. For alpha > 0, low-frequency amplification can be
    # large enough that float32 normalization leaves a non-trivial residual mean.
    white = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float64)

    spectrum = np.fft.fft2(white)

    fy = np.fft.fftfreq(height).reshape(height, 1)
    fx = np.fft.fftfreq(width).reshape(1, width)
    radial_freq = np.sqrt(fx * fx + fy * fy)

    if alpha == 0.0:
        spectral_filter = np.ones((height, width), dtype=np.float64)
    else:
        stabilized_freq = np.maximum(radial_freq, 1e-6)
        spectral_filter = np.power(stabilized_freq, -alpha)

    # Remove DC so the generated field is genuinely zero-centered.
    spectral_filter[0, 0] = 0.0

    shaped_spectrum = spectrum * spectral_filter
    colored_raw = np.real(np.fft.ifft2(shaped_spectrum)).astype(np.float64)

    mean = float(np.mean(colored_raw))
    std = float(np.std(colored_raw))

    if std < 1e-12:
        raise RuntimeError(
            "Generated degenerate colored noise with near-zero standard deviation."
        )

    colored = (colored_raw - mean) / std
    colored = colored - float(np.mean(colored))
    colored = colored / (float(np.std(colored)) + 1e-12)

    return colored.astype(np.float32, copy=False)


def alpha_conditioned_expected_weights(
    alpha: float,
    *,
    component_alphas: tuple[float, ...] = COMPONENT_ALPHAS,
    temperature: float = DEFAULT_WEIGHT_TEMPERATURE,
) -> np.ndarray:
    """
    Expected white/pink/brown mixture weights for a requested alpha.

    alpha is now interpreted as a continuous bias over spectral components:
        alpha near 0 -> expected mixture leans white/grainy
        alpha near 1 -> expected mixture leans pink/medium-scale
        alpha near 2 -> expected mixture leans brown/large-scale

    The returned weights sum to 1.0.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite.")

    centers = np.asarray(component_alphas, dtype=np.float64)
    logits = -float(temperature) * np.square(centers - alpha)
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError("Could not construct alpha-conditioned mixture weights.")
    return (weights / total).astype(np.float64)


def sample_alpha_conditioned_weights(
    alpha: float,
    rng: np.random.Generator,
    *,
    concentration: float = DEFAULT_WEIGHT_CONCENTRATION,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample random white/pink/brown weights from an alpha-conditioned Dirichlet.

    The expected value is controlled by alpha, while individual images can still
    be more grainy, more blobby, or more balanced. The weights always sum to 1.
    """
    expected = alpha_conditioned_expected_weights(alpha)
    concentration = float(concentration)
    if concentration <= 0.0 or not np.isfinite(concentration):
        raise ValueError("concentration must be positive and finite.")

    # Keep every Dirichlet parameter safely positive even for extreme alpha.
    dirichlet_params = np.maximum(expected * concentration, 1e-3)
    sampled = rng.dirichlet(dirichlet_params).astype(np.float64)
    sampled = sampled / float(np.sum(sampled))
    return sampled, expected


def _normalize_to_signed_unit_interval(field: np.ndarray) -> np.ndarray:
    """
    Normalize a real field to [-1, 1].

    This makes epsilon_phys the single global strength parameter. The mixture
    weights determine texture composition, not the overall amplitude.
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D noise field, got {arr.shape}")

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        raise RuntimeError("Degenerate noise field; cannot normalize to [-1, 1].")

    signed = 2.0 * ((arr - lo) / (hi - lo)) - 1.0
    return np.clip(signed, -1.0, 1.0).astype(np.float32, copy=False)


def generate_multiscale_colored_noise(
    height: int,
    width: int,
    alpha: float,
    seed: int | None = None,
    *,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """
    Generate one alpha-conditioned multi-scale Fourier noise field.

    The field is a weighted mixture of alpha components 0, 1, and 2. The requested
    alpha controls the distribution from which the component weights are sampled;
    it is not a categorical switch. The output field is normalized to [-1, 1].
    """
    height = int(height)
    width = int(width)
    rng = np.random.default_rng(seed)

    weights, expected = sample_alpha_conditioned_weights(alpha, rng)
    component_seeds = [int(rng.integers(0, np.iinfo(np.int32).max)) for _ in COMPONENT_ALPHAS]

    components: list[np.ndarray] = []
    for component_alpha, component_seed in zip(COMPONENT_ALPHAS, component_seeds, strict=True):
        components.append(
            generate_colored_noise(
                height=height,
                width=width,
                alpha=float(component_alpha),
                seed=component_seed,
            ).astype(np.float64)
        )

    mixed = np.zeros((height, width), dtype=np.float64)
    for weight, component in zip(weights, components, strict=True):
        mixed += float(weight) * component

    signed = _normalize_to_signed_unit_interval(mixed)

    metadata: dict[str, object] = {
        "model": "alpha_conditioned_multiscale_fourier_endpoint",
        "alpha": float(alpha),
        "component_names": list(COMPONENT_NAMES),
        "component_alphas": [float(x) for x in COMPONENT_ALPHAS],
        "expected_weights": {name: float(value) for name, value in zip(COMPONENT_NAMES, expected, strict=True)},
        "sampled_weights": {name: float(value) for name, value in zip(COMPONENT_NAMES, weights, strict=True)},
        "component_seeds": {name: int(value) for name, value in zip(COMPONENT_NAMES, component_seeds, strict=True)},
        "normalization": "minmax_to_signed_unit_interval",
    }

    if return_metadata:
        return signed, metadata
    return signed


def apply_endpoint_interpolation(
    image: np.ndarray,
    signed_noise: np.ndarray,
    epsilon_phys: float,
) -> np.ndarray:
    """
    Apply one signed continuous degradation field to an image.

    Negative noise moves pixels continuously toward pure black.
    Positive noise moves pixels continuously toward pure white.
    epsilon_phys is the fraction of allowed movement toward those endpoints.
    """
    arr = _validate_image_array(image)
    epsilon_phys = float(epsilon_phys)

    if not np.isfinite(epsilon_phys):
        raise ValueError("epsilon_phys must be finite.")
    if epsilon_phys < 0.0 or epsilon_phys > 1.0:
        raise ValueError("epsilon_phys must be in [0, 1] for endpoint interpolation.")

    z = np.asarray(signed_noise, dtype=np.float32)
    if z.shape != arr.shape[:2]:
        raise ValueError(f"Noise field shape {z.shape} does not match image shape {arr.shape[:2]}.")
    z = np.clip(z, -1.0, 1.0)

    if arr.ndim == 3:
        z = z[:, :, np.newaxis]

    positive = np.maximum(z, 0.0)
    negative = np.maximum(-z, 0.0)

    perturbed = arr * (1.0 - epsilon_phys * negative) + (1.0 - arr) * epsilon_phys * positive
    return np.clip(perturbed, 0.0, 1.0).astype(np.float32, copy=False)


def inject_spectral_noise(
    image: np.ndarray,
    epsilon_phys: float,
    alpha: float,
    seed: int | None = None,
    *,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """
    Inject alpha-conditioned multi-scale spectral degradation into an image.

    This is a single continuous corruption model:
        1. sample a white/pink/brown Fourier mixture biased by alpha
        2. normalize it to a signed field in [-1, 1]
        3. move pixels toward black for negative values and toward white for positive values

    epsilon_phys is the only strength/severity parameter.
    """
    epsilon_phys = float(epsilon_phys)

    if not np.isfinite(epsilon_phys):
        raise ValueError("epsilon_phys must be finite.")
    if epsilon_phys < 0.0 or epsilon_phys > 1.0:
        raise ValueError("epsilon_phys must be in [0, 1].")

    arr = _validate_image_array(image)
    height, width = arr.shape[:2]

    noise_2d, metadata = generate_multiscale_colored_noise(
        height=height,
        width=width,
        alpha=alpha,
        seed=seed,
        return_metadata=True,
    )
    metadata = dict(metadata)
    metadata["epsilon_phys"] = float(epsilon_phys)

    if epsilon_phys == 0.0:
        out = arr.copy()
    else:
        out = apply_endpoint_interpolation(arr, noise_2d, epsilon_phys)

    if return_metadata:
        return out, metadata
    return out


def save_float_image(path: Path, image: np.ndarray) -> None:
    """
    Save float [0, 1] image to disk as uint8 PNG/JPEG.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _validate_image_array(image)
    out = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(path), out)


def find_image_paths(images_dir: Path, recursive: bool = False) -> list[Path]:
    """
    Find full-page input images in a dataset directory.

    Important on Windows:
        Path.glob("*.png") may also match files ending in ".PNG", so iterating
        over both lower- and upper-case extensions can return the same file
        twice. Deduplicate by resolved, case-normalized path before sorting.

    Filters common debug/derived names so previews do not accidentally use
    already-generated artifacts.
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        if recursive:
            paths.extend(images_dir.rglob(f"*{ext}"))
        else:
            paths.extend(images_dir.glob(f"*{ext}"))

    unique: dict[str, Path] = {}
    for path in paths:
        if (
            "_teaser" in path.name
            or "_debug" in path.name
            or "_staff" in path.name
            or "_tesseract" in path.name
            or "spectral_noise_preview" in path.name
        ):
            continue

        try:
            key = str(path.resolve()).casefold()
        except Exception:
            key = str(path.absolute()).casefold()

        unique[key] = path

    return sorted(unique.values(), key=lambda p: str(p).casefold())


def write_preview_for_image(
    *,
    image_path: Path,
    output_dir: Path,
    epsilon_phys: float,
    alpha: float,
    seed: int,
) -> Path:
    """
    Generate and save one noisy preview image.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    perturbed = inject_spectral_noise(
        image=image,
        epsilon_phys=epsilon_phys,
        alpha=alpha,
        seed=seed,
    )

    output_path = output_dir / f"{image_path.stem}_spectral_eps_{epsilon_phys:g}_alpha_{alpha:g}.png"
    save_float_image(output_path, perturbed)
    return output_path


def _self_test() -> None:
    """
    Run lightweight deterministic validation.
    """
    zero = np.zeros((32, 64), dtype=np.float32)
    white = np.ones((32, 64), dtype=np.float32)
    gray_rgb = np.full((32, 64, 3), 0.5, dtype=np.float32)

    for alpha in (0.0, 1.0, 2.0):
        noise = generate_colored_noise(32, 64, alpha=alpha, seed=123)
        assert noise.shape == (32, 64)
        assert noise.dtype == np.float32
        assert np.isfinite(noise).all()
        assert abs(float(noise.mean())) < 1e-4, float(noise.mean())
        assert 0.99 < float(noise.std()) < 1.01, float(noise.std())

    unchanged = inject_spectral_noise(gray_rgb, epsilon_phys=0.0, alpha=1.0, seed=123)
    assert np.array_equal(unchanged, gray_rgb)

    for image in (zero, white, gray_rgb):
        perturbed = inject_spectral_noise(image, epsilon_phys=0.1, alpha=1.0, seed=123)
        assert perturbed.shape == image.shape
        assert perturbed.dtype == np.float32
        assert np.isfinite(perturbed).all()
        assert 0.0 <= float(perturbed.min()) <= 1.0
        assert 0.0 <= float(perturbed.max()) <= 1.0

    print("spectral_noise.py self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spectral noise utilities and preview generator for Track A."
    )

    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Optional specific input image. If omitted with --preview, auto-discovers from --images-dir.",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic module self-test.",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate debug preview image(s), not benchmark outputs.",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("dataset/images"),
        help="Directory used for auto-discovery when no image path is provided.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/debug/spectral_noise_previews"),
        help="Directory for debug preview outputs. These are not benchmark result plots.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=1,
        help="Maximum auto-discovered images to preview.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search --images-dir recursively.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.preview:
        print(
            "Nothing to do. Use --self-test or --preview.\n"
            "Example: python attacks/src/spectral_noise.py --preview --max-images 3"
        )
        return

    if args.image is not None:
        image_paths = [args.image]
    else:
        image_paths = find_image_paths(args.images_dir, recursive=args.recursive)
        if args.max_images is not None:
            image_paths = image_paths[: int(args.max_images)]

    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {args.images_dir}. "
            "Check dataset/images or pass a specific image path."
        )

    print(f"[preview] n_images={len(image_paths)} epsilon={args.epsilon} alpha={args.alpha}")
    print(f"[preview] output_dir={args.output_dir}")

    for i, image_path in enumerate(image_paths, start=1):
        output_path = write_preview_for_image(
            image_path=image_path,
            output_dir=args.output_dir,
            epsilon_phys=args.epsilon,
            alpha=args.alpha,
            seed=args.seed + i - 1,
        )
        print(f"  [{i}/{len(image_paths)}] {image_path} -> {output_path}")


if __name__ == "__main__":
    main()
