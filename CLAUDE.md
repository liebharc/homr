# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## GUARDRAILS - READ FIRST, ALWAYS APPLY

These rules apply to every file, every edit, every response, without exception.

**No comments in code.**
Do not write any comments in Python files. No inline comments, no block comments, no docstrings. If the code needs explaining, the name of the function or variable should explain it. The only exception is a single short line when a non-obvious external constraint forces a specific value or workaround - and even then it must be one line maximum.

**No emojis, no AI symbols, no decorative characters.**
Do not use emojis anywhere - not in code, not in commit messages, not in output text, not in markdown. Do not use checkmarks, arrows made from special unicode characters, stars, sparkles, or any character that a normal programmer would not type on a standard keyboard. Use plain ASCII only in code and commit messages.

**Use plain dashes, not AI punctuation.**
Use a plain hyphen-minus (-) for list items and dashes. Do not use em dashes (--), en dashes, bullet points (- vs *), or any typographic substitution. In markdown, use - for unordered lists.

**Write human-readable code.**
Name things clearly so the code reads like plain English. No one-letter variables outside of loop counters. No abbreviations that are not universally obvious. No clever tricks that compress logic at the cost of readability.

**Use established libraries instead of reimplementing standard algorithms.**
When a well-known library already solves the problem correctly - loss functions, optimizers, learning rate schedulers, metric computation, data augmentation, image transforms - use it. Do not write a custom implementation of something that PyTorch, torchvision, albumentations, editdistance, or scipy already provide. This applies when the library's interface fits cleanly and does not require bending the surrounding code into unusual shapes to accommodate it. If fitting the library in would make the code harder to read or would pull in a large dependency for a trivial use, write the straightforward implementation instead.

---

## Project Overview

This project audits the robustness of HOMR (Hierarchical Optical Music Recognition) against adversarial attacks and natural image degradation. It runs spectral-noise and black-box square-attack tracks, and builds a differentiable surrogate model that is hardened with PGD adversarial training (defense) and attacked with AutoAttack (evaluation) to study gradient-based robustness and transfer.

Pipeline:

```
Sheet Music Image -> Perturbation/Attack -> HOMR ONNX Pipeline -> MusicXML -> SER/CER Metrics
```

---

## Commands

```bash
make init
make test
make lint
make format
make typecheck
make ci

poetry run pytest -s tests/test_model.py
```

- `make init` - install deps and set up pre-commit hooks (Poetry required)
- `make test` - run pytest against tests/
- `make lint` - black + isort + ruff check on CI directories
- `make format` - auto-format code in place
- `make typecheck` - mypy
- `make ci` - lint + typecheck + test

CI directories: `homr training tests validation`. The `attacks/`, `distillation/`, `dataset/` directories are not type-checked by default.

---

## Architecture

### ONNX Boundary Policy - Hard Rule

All neural inference must go through ONNX Runtime. Never call `Staff2Score.predict(...)`, `Encoder(...)`, `get_decoder(...)`, or `parse_staff_tromr(...)` in benchmark or training code. HOMR Python code is only allowed for deterministic preprocessing and postprocessing.

ONNX models live at `models/onnx/`:

- `segnet.onnx` - 6-class semantic segmentation, input `[B, 3, 320, 320]`, output `[B, 6, 320, 320]`
- `tromr_encoder.onnx` - input `[1, 1, 256, 1280]`, output `[1, 1280, 512]`
- `tromr_decoder.onnx` - multi-stream autoregressive over rhythm/pitch/lift/articulation/position with KV-cache

Always use `CUDAExecutionProvider` (not TensorRT) with:

```python
{"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kNextPowerOfTwo"}
```

### HOMR Pipeline Decomposition

```
Full-page image X
  -> SegNet ONNX (sliding-window 320x320 patches -> 6-class argmax masks)
  -> HOMR deterministic layout pipeline (Python/OpenCV geometry -> Staff/MultiStaff objects)
  -> prepare_staff_image() -> [256, 1280] grayscale float32 [0, 1]
  -> TrOMR normalization: (x - 0.7931) / 0.1738 -> [1, 1, 256, 1280]
  -> tromr_encoder.onnx -> context [1, 1280, 512]
  -> tromr_decoder.onnx (autoregressive, 4 streams) -> token sequences
  -> HOMR vocabulary decoding + MusicXML assembly
```

SegNet class indices: 0=background, 1=stems/rests, 2=noteheads, 3=clefs/keys, 4=staff lines, 5=symbols.

### Attack Tracks

Track A - Spectral Noise (Natural OOD):
Frequency-domain 1/f-alpha noise injected into full-page images before the full pipeline. Config: `attacks/config/sweep_parameters.yaml`. Script: `attacks/run_spectral_sweep.py`. Implementation: `attacks/src/spectral_noise.py`.

Track B - Square Attack (Adversarial):
Zero-order black-box attack on cached prepared staff images (output of `prepare_staff_image()`). Bypasses the expensive layout pipeline (~6s to ~0.18s per query). Script: `attacks/run_square_sweep.py`. Implementation: `attacks/src/square_attack.py`.

The staff image cache lives at `dataset/cached_prepared_staffs/<score_id>/` as `staff_NNN.npy` (float32 [0,1], shape [256, 1280]) plus `metadata.json`. Generate via `dataset/cache_prepared_staffs.py` and stop before `parse_staff_tromr()`.

Track C - Surrogate Adversarial Robustness (Defense: PGD, Attack: AutoAttack):
AutoAttack (the standard APGD-CE / APGD-T / FAB / Square ensemble) is the designated white-box attack against the differentiable full-page surrogate. The defense is PGD adversarial training of that surrogate. The study trains a clean surrogate and a PGD-trained surrogate on identical data, then compares them across an L-inf epsilon grid using branch accuracy and SER. Defense implementation: `distillation/pgd_attack.py` plus `distillation/train_student.py --adv-train`. Attack and comparison: `distillation/evaluate_surrogate.py`. Adversarial examples that defeat the surrogate are intended to be replayed against the HOMR ONNX pipeline to measure transfer.

### attacks/src/ Modules

| Module | Role |
|---|---|
| `homr_wrapper.py` | `HOMRBlackBoxWrapper` - loads encoder and decoder ONNX sessions, exposes `predict_prepared_staff()`, `score_query()`, `encoder_forward()`, `decoder_generate()`. Has a CLI smoke test. |
| `square_attack.py` | `run_square_attack()` - L-inf square attack loop |
| `spectral_noise.py` | `inject_spectral_noise()`, `generate_colored_noise()` |
| `statistics_engine.py` | `symbol_error_rate()`, `character_error_rate()`, `batch_metrics()` using editdistance |
| `segmentation_onnx.py` | `SegNetONNX` - sliding-window SegNet ONNX inference |

### Surrogate Model (Track C / Distillation)

A differentiable full-page student (`FullPageHOMRSurrogate` / `PageStaffARStudent`) trained on HOMR teacher outputs via ONNX. Purpose: enable gradient-based adversarial attacks that transfer to HOMR.

Target architecture (full spec in `docs/STUDENT_ARCHITECTURE.md`):

```
CNN/ConvNeXt page encoder
  -> Transformer page memory
  -> DETR-style ordered staff-slot decoder
  -> shared autoregressive per-staff decoder
  -> HOMR factorized heads (rhythm/pitch/lift/articulation/position)
```

Current status: Phase 0 infrastructure baseline is complete with a temporary smoke model (`FullPageStudent` inline in `distillation/train_student.py`). The real `PageStaffARStudent` still needs to be implemented at `distillation/models/page_staff_ar_student.py`. The `FullPageStudent` encoder ends with `AdaptiveAvgPool2d((4, 4))` and a `page_projection` linear so spatial layout survives into gradient-based attacks.

Distillation pipeline, run in this order:

```
distillation/build_source_pool.py
distillation/select_batch.py
distillation/render_batch.py
distillation/flatten_renders.py
distillation/augment_pages.py
distillation/run_onnx_teacher_batch.py
distillation/build_training_manifest.py
distillation/make_splits.py
distillation/vocab.py
distillation/train_student.py
distillation/evaluate_surrogate.py
```

`distillation/flatten_renders.py` composites transparent MuseScore 3 (`mscore3`) PNG renders onto a white background. SegNet expects black-on-white, so this step is required whenever rendering uses MuseScore 3 instead of MuseScore 4.

Defense and attack:

- `distillation/pgd_attack.py` - `pgd_attack()` runs L-inf PGD using `compute_loss` from `train_student`. It clamps to the model input domain `[-1, 1]` (the page loader normalizes to `[-1, 1]`), not `[0, 1]`.
- `distillation/train_student.py --adv-train` - replaces each training batch with PGD adversarial examples before the forward pass. Flags: `--pgd-epsilon`, `--pgd-steps`, `--pgd-alpha`. Validation stays clean.
- `distillation/evaluate_surrogate.py` - loads a clean and a PGD-trained checkpoint, sweeps an `--epsilon-grid`, and writes `results/surrogate_comparison/comparison.json` (schema `homr_surrogate_pgd_comparison_v1`). AutoAttack is the designated attack here; a PGD epsilon grid is the current built-in baseline.

Manifest schema: `homr_factorized_page_training_manifest_v1`. Each row has `image_path`, `n_staffs`, and `homr_target_staffs` (list of staffs with `rhythm_ids`, `pitch_ids`, `lift_ids`, `articulation_ids`, `position_ids`, `mask`).

No `<STAFF_BREAK>` token. No newline token in the staff vocabulary. `EncodedSymbol("newline")` is inserted only during postprocessing page reconstruction.

### Metrics

- SER (Symbol Error Rate) - Levenshtein distance on token sequences divided by ground-truth length. Primary loss proxy for `score_query()`.
- CER (Character Error Rate) - character-level Levenshtein divided by ground-truth char length.
- Results written to `results/logs/sweep_metrics.json`. Every result entry must include `"stage"` (`"A1"` or `"A2"`) and `"onnx_mode"` fields.

---

## Runtime Environment

This project runs on Google Colab with a GPU runtime. The repository is cloned into `/content/adversarial-homr` and the working directory for all scripts is `/content/adversarial-homr`.

The PDMX dataset lives on Google Drive and is mounted at `/content/drive`. The relevant files are:

- MXL archive: `/content/drive/MyDrive/College/Projects/AI-AOL/mxl.tar.gz`
- Metadata CSV: `/content/drive/MyDrive/College/Projects/AI-AOL/PDMX.csv`

After mounting Drive, the dataset is extracted once per Colab session:

```bash
from google.colab import drive
drive.mount("/content/drive")

cp /content/drive/MyDrive/College/Projects/AI-AOL/PDMX.csv dataset/PDMX.csv
tar -xzf /content/drive/MyDrive/College/Projects/AI-AOL/mxl.tar.gz -C dataset/
```

Any script or pipeline step that reads `dataset/PDMX.csv` or `dataset/mxl/` assumes the above extraction has already been run. Do not hardcode the Drive path inside scripts - copy to `dataset/` first and reference local paths only.

ONNX models are stored as zip files in `models/onnx/` and must be extracted before first use:

```bash
cd models/onnx
unzip encoder_pytorch_model_331-*.zip && mv encoder_pytorch_model_331-*.onnx tromr_encoder.onnx
unzip decoder_pytorch_model_331-*.zip && mv decoder_pytorch_model_331-*.onnx tromr_decoder.onnx
```

The segnet model must also be present as `models/onnx/segnet.onnx`. Verify all three with:

```bash
python attacks/src/homr_wrapper.py --describe-only
```

---

## Key Configuration

`attacks/config/sweep_parameters.yaml` is the single source of truth for all sweep hyperparameters. Both Track A and Track B sweep scripts load from this file.

Environment (Python 3.11, Poetry):

```bash
poetry install
export PYTHONPATH=$(pwd)
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The decoder has `decoder_depth * 4 = 32` KV-cache tensors, each shaped `[1, 8, cache_len, 64]`. On step 0 the full context `[1, 1280, 512]` is passed. On all subsequent steps only `context[:, :1, :]` is used.

