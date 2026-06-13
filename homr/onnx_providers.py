"""GPU execution-provider selection for ONNX Runtime.

homr was written for NVIDIA/CUDA only. This helper adds support for the
Apple Silicon GPU / Neural Engine via the CoreML execution provider.

On Macs only segnet (and, opt-in, the encoder) runs on CoreML: the CoreML EP
needs ``ModelFormat=MLProgram`` to accept any nodes of homr's fp16 models,
and under MLProgram the decoder crashes on its dynamic KV-cache dimension
(onnxruntime 1.26). The decoder therefore always stays on the CPU EP with the
fp32 model, which is faster than the fp16 model on the CPU EP.

CUDA exposes device memory that the rest of homr binds to as ``"cuda"``; the
CoreML provider does not expose such device memory to user code, so its
inputs/outputs are bound on the CPU ("cpu") while the heavy compute still
runs on the GPU/ANE.
"""

import os
from typing import Any

import onnxruntime as ort

# Let CoreML use the whole Apple Neural Engine + GPU + CPU and pick the
# fastest unit per op. Use "CPUAndGPU" to force the GPU only.
_COREML_DEFAULT_OPTIONS: dict[str, str] = {"MLComputeUnits": "ALL"}

# Environment variables that override/extend the CoreML provider options,
# so users can tune them without code changes (see the onnxruntime CoreML
# execution-provider documentation for the accepted values).
_COREML_ENV_OPTIONS: dict[str, str] = {
    "HOMR_COREML_COMPUTE_UNITS": "MLComputeUnits",
    "HOMR_COREML_SPECIALIZATION_STRATEGY": "SpecializationStrategy",
    "HOMR_COREML_MODEL_FORMAT": "ModelFormat",
    "HOMR_COREML_MODEL_CACHE_DIR": "ModelCacheDirectory",
}


def _coreml_options() -> dict[str, str]:
    options = dict(_COREML_DEFAULT_OPTIONS)
    for env_var, option in _COREML_ENV_OPTIONS.items():
        value = os.environ.get(env_var)
        if value:
            options[option] = value
    return options


def coreml_mlprogram_providers(model_path: str, compute_units: str = "ALL") -> list[Any]:
    """CoreML provider options that actually place nodes on the GPU/ANE.

    With ORT's default "NeuralNetwork" model format the CoreML EP accepts
    almost no nodes of homr's fp16 models and everything silently runs on the
    CPU EP; "MLProgram" is required for real GPU execution. Compiling an
    MLProgram is expensive, so the compiled model is cached in a directory
    derived from the (version-hashed) model filename — a model update
    automatically invalidates the cache.
    """
    options = {
        "MLComputeUnits": compute_units,
        "ModelFormat": "MLProgram",
        "ModelCacheDirectory": os.path.splitext(model_path)[0] + ".coreml_cache",
    }
    for env_var, option in _COREML_ENV_OPTIONS.items():
        value = os.environ.get(env_var)
        if value:
            options[option] = value
    os.makedirs(options["ModelCacheDirectory"], exist_ok=True)
    return [("CoreMLExecutionProvider", options)]


def cuda_available() -> bool:
    return "CUDAExecutionProvider" in ort.get_available_providers()


def coreml_available() -> bool:
    return "CoreMLExecutionProvider" in ort.get_available_providers()


def gpu_available() -> bool:
    """True if any GPU execution provider (CUDA or Apple CoreML) is present."""
    return cuda_available() or coreml_available()


def gpu_providers(cuda_options: dict[str, Any] | None = None) -> tuple[list[Any], str]:
    """Return ``(providers, io_binding_device)`` for the best available GPU.

    Raises ``RuntimeError`` if no GPU provider is available so callers can fall
    back to a plain CPU session exactly as they did for CUDA failures before.
    """
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        if cuda_options:
            return [("CUDAExecutionProvider", cuda_options)], "cuda"
        return ["CUDAExecutionProvider"], "cuda"
    if "CoreMLExecutionProvider" in available:
        return [("CoreMLExecutionProvider", _coreml_options())], "cpu"
    raise RuntimeError("No GPU execution provider (CUDA or CoreML) is available")
