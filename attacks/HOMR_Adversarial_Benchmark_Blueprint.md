# HOMR Adversarial Robustness Benchmark
## Execution Blueprint — Full Technical Specification
### Version 3.0 — Fully Integrated Revision

---

## Preface

This document constitutes the complete, end-to-end execution specification for auditing the
robustness of the **Hierarchical Optical Music Recognition (HOMR)** system against both
natural out-of-distribution image degradation and targeted adversarial perturbation.

The system under test is a **heterogeneous, read-only pipeline** composed of three distinct
ONNX inference graphs and a surrounding deterministic layout-analysis engine:

- **SegNet/HOMR Segmentation** — a semantic pixel-classification network, compiled to
  `segnet.onnx`. It maps full-page image patches to a six-class semantic segmentation map.
  HOMR converts the six-channel softmax output to an integer class map using
  `np.argmax(out, axis=0)`, then derives binary masks for each semantic class. These masks
  feed a classical OpenCV-based layout pipeline; the network is not a bounding-box detector.
- **TrOMR Encoder** — compiled to `tromr_encoder.onnx`. Encodes a prepared single-staff
  image of shape `[1, 1, 256, 1280]` into a context tensor of shape `[1, 1280, 512]`.
- **TrOMR Decoder** — compiled to `tromr_decoder.onnx`. A multi-stream autoregressive
  decoder that consumes encoder context plus per-step inputs across four independent
  vocabularies: rhythm, pitch, lift, and articulation. The decoder is not a generic
  single-stream text decoder and cannot be treated as one.
- **HOMR Deterministic Middle Pipeline** — the classical OpenCV/NumPy geometry engine
  that converts SegNet masks into `Staff` and `MultiStaff` objects, constructs bounding
  boxes, groups staves, computes dewarping geometry, and produces the exact image that is
  fed into TrOMR. This stage is implemented in Python and is entirely deterministic; it
  contains no neural network inference.

Because model weights are not exposed as differentiable checkpoints, gradient-based
adversarial training is **structurally impossible**. The entire evaluation therefore operates
under a strict **Black-Box Threat Model**, and the objective is reformulated as a
**Comprehensive Vulnerability Benchmark**.

---

## ONNX Boundary Policy

This section defines the hard architectural boundary governing all benchmark implementations.
It applies to every script, notebook, wrapper, and sweep in the project without exception.

**All neural network inference must go through ONNX Runtime:**

| Model | File | ONNX Runtime Session |
|---|---|---|
| SegNet segmentation | `models/onnx/segnet.onnx` | Required |
| TrOMR encoder | `models/onnx/tromr_encoder.onnx` | Required |
| TrOMR decoder | `models/onnx/tromr_decoder.onnx` | Required |

**HOMR Python code may be reused exclusively for deterministic preprocessing and
postprocessing:**

- Image autocrop, resize, CLAHE
- Segmentation mask postprocessing (`argmax`, binary mask extraction)
- Bounding box construction from contours
- Staff detection and geometry
- Notehead, stem, and barline geometry analysis
- Brace and bracket grouping
- Staff dewarping transformations
- TrOMR canvas preparation (`prepare_staff_image`)
- Vocabulary and token decoding utilities
- MusicXML assembly from decoded symbol sequences

**Prohibited in benchmark mode:**

The benchmark must not silently invoke PyTorch or TensorFlow neural model inference.
Specifically, `Staff2Score.predict(...)` instantiates `Encoder` and `get_decoder` via
PyTorch — this call is strictly prohibited in final benchmark measurements. If `Staff2Score`
is used for temporary diagnostics or smoke-testing during development, that run must be
clearly labeled as diagnostic and excluded from all reported results.

**The canonical benchmark pipeline is:**

```
SegNet ONNX
  → HOMR deterministic layout pipeline (Python geometry only, no neural inference)
  → cached prepared staff images (output of prepare_staff_image)
  → TrOMR encoder ONNX
  → TrOMR decoder ONNX
  → HOMR vocabulary decoding + MusicXML assembly (Python, no neural inference)
```

---

## Validated Facts from Hardware and Model Inspection

The following facts have been confirmed on the target machine and in the HOMR source code.
They are the authoritative ground truth for all implementation decisions and supersede any
contradictory assumptions.

**1. ONNX Runtime available execution providers (as detected on target machine):**
```
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

**2. Forced provider configuration used for all benchmark runs:**
```
['CUDAExecutionProvider', 'CPUExecutionProvider']
```
TensorRT is present but intentionally bypassed. See §3.2 for rationale.

**3. SegNet ONNX tensor shapes:**
```
input:  [batch_size, 3, 320, 320]    dtype: float32
output: [batch_size, 6, 320, 320]    dtype: float32
```

**4. SegNet semantic class index mapping (from HOMR source):**
```
Class 0 = background
Class 1 = stems / rests
Class 2 = noteheads
Class 3 = clefs / key signatures
Class 4 = staff lines
Class 5 = symbols
```
HOMR extracts the class map via `np.argmax(out, axis=0)` applied to the `[6, 320, 320]`
output slice.

**5. TrOMR encoder expected input shape:**
```
[1, 1, 256, 1280]    dtype: float32
```

**6. TrOMR encoder output shape:**
```
[1, 1280, 512]    dtype: float32
```

**7. TrOMR image normalization transform (sourced from `Staff2Score.ConvertToArray` and
`Config`, where `channels=1`, `max_height=256`, `max_width=1280`):**
```python
arr = np.array(image, dtype=np.float32) / 255.0
arr = arr[np.newaxis, np.newaxis, :, :]   # → [1, 1, H, W]
arr = (arr - 0.7931) / 0.1738
# dtype: float32, shape: [1, 1, 256, 1280]
```

**8. TrOMR decoder structure:**
The decoder is multi-stream, operating over four independent token vocabularies
simultaneously: rhythm, pitch, lift, and articulation. It also consumes `context`,
`cache_len`, and `cache` tensors at each autoregressive step. It is not a generic
single-stream logit decoder and must not be treated as one when implementing teacher-forced
cross-entropy or decoder generation.

---

## Part I — System Model and Threat Formulation

### 1.1 The End-to-End Pipeline as a Black-Box Function

Let $\mathbf{X} \in [0,1]^{H \times W \times C}$ denote a raw, full-page sheet music image.
The complete HOMR pipeline is treated as a single, monolithic, non-differentiable function:

$$F(\mathbf{X}) = \mathbf{Y}_{pred} \in \Sigma^*$$

where $\Sigma^*$ is the set of all finite sequences over the MusicXML/Symbol vocabulary
$\Sigma$.

Internally, the pipeline decomposes as:

$$\mathbf{Y}_{pred}
=
\mathcal{A}
\left(
  \left\{
    f_{rec}
    \left(
      P_k
      \left(
        G
        \left(
          f_{seg}(\mathbf{X}), \mathbf{X}
        \right)
      \right)
    \right)
  \right\}_{k=1}^{K}
\right)$$

| Symbol | Module | Role |
|---|---|---|
| $f_{seg}(\mathbf{X})$ | SegNet ONNX | Maps $\mathbf{X}$ via sliding-window 320×320 patches to a six-class semantic pixel map; output is integer class indices via `argmax`, not bounding boxes |
| $G(f_{seg}(\mathbf{X}), \mathbf{X})$ | HOMR deterministic layout pipeline | Consumes SegNet class masks and the original image to construct `Staff`/`MultiStaff` geometry objects, bounding boxes, notehead/stem/barline geometry, and dewarping parameters |
| $P_k(\cdot)$ | `prepare_staff_image(...)` | Crops, dewarps, resizes, cleans edge artifacts, converts to grayscale, and centers the $k$-th staff onto the TrOMR canvas (256×1280); produces the exact image input to TrOMR |
| $f_{rec}(\cdot)$ | TrOMR encoder + decoder ONNX | Encodes the prepared staff image via `tromr_encoder.onnx` and autoregressively decodes symbol sequences via `tromr_decoder.onnx` over four token streams: rhythm, pitch, lift, articulation |
| $\mathcal{A}(\cdot)$ | HOMR MusicXML assembly | Joins per-staff decoded symbol lists into the full-page prediction |

**Critical architectural note:** The decomposition is not simply
`segmentation → crop → TrOMR`. The middle stage $G$ is a substantial deterministic
HOMR/OpenCV layout-analysis pipeline. It consumes SegNet semantic masks and constructs
`Staff`/`MultiStaff` geometry before producing the image actually fed into TrOMR. This is
grounded in `main.py`, where HOMR calls SegNet extraction, wraps masks into
`InputPredictions`, then runs filtering, staff detection, note/stem/barline processing,
staff grouping, and `parse_staffs`, all before `parse_staff_tromr` is reached.

SegNet maps page image patches to six semantic pixel classes:
```
background, stems/rests, noteheads, clefs/key signatures, staff lines, symbols
```
HOMR's classical postprocessing then converts those masks into `Staff` and `MultiStaff`
objects, crop regions, and prepared TrOMR inputs.

---

### 1.2 Threat Model Classification

| Property | Specification |
|---|---|
| **Access type** | Black-box (query-only; no gradients, no logits by default) |
| **Attack surface** | Pixel space of full-page images $\mathbf{X}$ (Track A) or HOMR-prepared staff images $P_k(\cdot)$ (Track B) |
| **Perturbation norm** | $L_\infty$ (bounded pixel deviation) |
| **Differentiability** | End-to-end non-differentiable for this benchmark. Although ONNX neural subgraphs internally compute continuous tensors, the deployed HOMR pipeline contains `argmax` segmentation, contour extraction, bounding-box construction, staff grouping, dewarping decisions, token sampling/greedy decoding, and MusicXML assembly. The evaluation therefore treats the complete system as a query-only black box. SegNet itself outputs continuous scores, but HOMR immediately applies `argmax` and extensive discrete geometry that eliminates any gradient path. |
| **Defense training** | Impossible (read-only ONNX weights) |

---

### 1.3 Dual-Track Evaluation Strategy

The benchmark runs two independent evaluation tracks:

```
Track A: Natural OOD Degradation
         └─ Spectral Noise Injection
            └─ Simulates physical scanner / paper / sensor artifacts
            └─ Noise generated in frequency domain via 1/f^alpha shaping,
               inverse-transformed to pixel space
            └─ Three spectral colors: white (alpha=0), pink (alpha=1), brown (alpha=2)
            └─ Sweep: epsilon_phys in {0.00, 0.05, 0.10, 0.20, 0.30, 0.50}
            └─ Metric: SER / CER on full-pipeline ONNX output

Track B: Malicious Targeted Exploitation
         └─ Square Attack (zero-order random coordinate search)
            └─ Operates on HOMR-prepared staff images: the exact images produced by
               prepare_staff_image(...), immediately before TrOMR encoder input
            └─ Primary loss proxy: normalized SER / Levenshtein distance
               (black-box; no logit access required)
            └─ Optional advanced mode: multi-head decoder loss over rhythm, pitch,
               lift, and articulation streams via teacher-forced ONNX decoder execution
            └─ Sweep: epsilon_math in {0.01, 0.02, 0.05, 0.10, 0.20}
```

**Track B input clarification:** Track B does not attack full-page images. It attacks
the prepared staff images cached by `dataset/cache_prepared_staffs.py`. These images have
already passed through the full HOMR deterministic layout pipeline including SegNet
segmentation, mask postprocessing, staff detection, grouping, dewarping, resizing, and
centering. This is correct and deliberate: it decouples the adversarial query loop from
the expensive, non-differentiable layout pipeline and focuses the perturbation budget on
the recognition module $f_{rec}$.

---

## Part II — Mathematical Derivations

### 2.1 Track A — Spectral Noise Injection

**Purpose:** Simulate physically plausible image degradation (scanner noise, paper aging,
CCD sensor artifacts) by injecting spectrally shaped noise in the frequency domain.

#### Step 1 — White Noise Sampling

Sample a 2D spatial noise array from an i.i.d. standard normal distribution:

$$W(x, y) \sim \mathcal{N}(0, 1), \quad W \in \mathbb{R}^{H \times W}$$

The Power Spectral Density (PSD) of $W$ is flat across all frequencies.

#### Step 2 — Transform to Frequency Domain

Apply the 2D Discrete Fourier Transform:

$$\hat{W}(u, v) = \mathcal{F}\{W(x, y)\}$$

#### Step 3 — Spectral Shaping via 1/fᵅ Filter

To produce colored noise, attenuate high-frequency amplitudes by the radial spatial
frequency $f(u,v) = \sqrt{u^2 + v^2}$. To prevent division-by-zero at the DC component
$(u=0,\,v=0)$, apply a stabilization floor $\eta = 10^{-6}$:

$$f_{stab}(u, v) = \max\!\left(\sqrt{u^2 + v^2},\; \eta\right)$$

The filtered spectrum is:

$$\hat{C}(u, v) = \hat{W}(u, v) \cdot \left[f_{stab}(u, v)\right]^{-\alpha}$$

| $\alpha$ | Noise Color | Physical Analogy |
|---|---|---|
| 0.0 | White | Thermal / electronic sensor noise |
| 1.0 | Pink | Photocopier degradation |
| 2.0 | Brown | Heavy paper aging / low-frequency fog |

#### Step 4 — Inverse Transform and Normalization

Recover the spatial-domain noise, discarding floating-point imaginary residuals:

$$C_{raw}(x, y) = \Re\!\left(\mathcal{F}^{-1}\!\left\{\hat{C}(u, v)\right\}\right)$$

Standardize to zero mean and unit variance, protected by numerical floor
$\epsilon_{num} = 10^{-12}$:

$$C(x, y) = \frac{C_{raw}(x, y) - \mu(C_{raw})}{\sigma(C_{raw}) + \epsilon_{num}}$$

#### Step 5 — Perturbation Integration

Blend with the original image and clamp to the valid image manifold:

$$\mathbf{X}_{phys} = \text{clip}\!\left(\mathbf{X} + \epsilon_{phys} \cdot C,\; 0,\; 1\right)$$

The scalar $\epsilon_{phys}$ is the sole intensity control swept during Track A evaluation.

---

### 2.2 Track B — Square Attack (Zero-Order Adversarial Optimization)

**Purpose:** Execute a directed, gradient-free adversarial attack against the TrOMR
recognition module $f_{rec}$, operating strictly on HOMR-prepared staff images to bypass
the discrete decision boundaries of the layout pipeline $G$.

#### Formal Objective

Given a clean prepared staff image $\mathbf{x}$ (the output of `prepare_staff_image(...)`),
an $L_\infty$ perturbation budget $\epsilon_{math}$, and ground-truth token sequence
$\mathbf{Y}_{gt}$, find the perturbation $\boldsymbol{\delta}^*$ that maximizes the
sequence divergence metric $D$:

$$\boldsymbol{\delta}^*
=
\arg\max_{\|\boldsymbol{\delta}\|_\infty \le \epsilon_{math}}
D
\left(
  \operatorname{Decode}_{ONNX}(\mathbf{x}+\boldsymbol{\delta}),\;
  \mathbf{Y}_{gt}
\right)$$

**Primary loss proxy (initial / default implementation):**

$D$ is the normalized edit distance (Symbol Error Rate or Character Error Rate) between the
predicted symbol sequence and the ground-truth sequence. This is a black-box objective that
requires no logit access and no gradient information:

$$D_{SER}(\hat{\mathbf{Y}}, \mathbf{Y}_{gt})
= \frac{S_{sym} + D_{sym} + I_{sym}}{\left|\mathbf{Y}_{gt}\right|}$$

**Practical note:** Cross-entropy is not immediately available from HOMR's normal inference
path. `Staff2Score.predict` uses `Staff2Score.generate`, which calls the decoder
autoregressively without exposing per-step logits externally. The initial `score_query`
implementation must therefore use SER/CER as the loss proxy. Do not block Track B
implementation on teacher-forced cross-entropy.

**Optional advanced loss (teacher-forced multi-stream cross-entropy):**

If teacher-forced ONNX decoder execution is implemented — feeding ground-truth tokens as
decoder inputs at each step — the loss can be decomposed across the four token streams:

$$\mathcal{L}_{multi}
=
\lambda_r \cdot \text{CE}_r
+
\lambda_p \cdot \text{CE}_p
+
\lambda_l \cdot \text{CE}_l
+
\lambda_a \cdot \text{CE}_a$$

where subscripts $r, p, l, a$ denote rhythm, pitch, lift, and articulation streams
respectively. This requires correct feeding of `rhythms`, `pitchs`, `lifts`,
`articulations`, `context`, `cache_len`, and `cache` tensors into `tromr_decoder.onnx` at
each step. It is the advanced mode for future extension, not the initial baseline.

#### Initialization

Initialize $\boldsymbol{\delta}^{(0)}$ on the boundary of the $L_\infty$ hypercube:

$$\boldsymbol{\delta}^{(0)} \in \{-\epsilon_{math},\; +\epsilon_{math}\}^{h \times w \times c}$$

$$\mathbf{x}^{(0)} = \text{clip}\!\left(\mathbf{x} + \boldsymbol{\delta}^{(0)},\; 0,\; 1\right),
\quad
L_{best} = D\!\left(\operatorname{Decode}_{ONNX}(\mathbf{x}^{(0)}),\; \mathbf{Y}_{gt}\right)$$

#### Iteration Loop (steps $i = 1 \ldots N_{max}$)

**1. Decay schedule** — Compute the square side length $h^{(i)}$ via a monotonically
decreasing proportion $p^{(i)}$ of the staff image width $w$:

$$h^{(i)} = \text{round}\!\left(w \cdot p^{(i)}\right),
\quad
p^{(i)} \text{ decreasing geometrically over } [1, N_{max}]$$

**2. Sample patch origin** — Draw uniformly:

$$(r, c) \sim \mathcal{U}\!\left(\{0,\ldots,h - h^{(i)}\} \times \{0,\ldots,w - h^{(i)}\}\right)$$

**3. Construct candidate perturbation** — Fill the selected square with a constant value
drawn from $\{-\epsilon_{math}, +\epsilon_{math}\}$; retain all other positions from the
previous state.

**4. Project and clip:**

$$\mathbf{x}_{cand} = \text{clip}\!\left(\Pi_{[\mathbf{x}-\epsilon_{math},\;\mathbf{x}+\epsilon_{math}]}\!\left(\mathbf{x}^{(i-1)} + \boldsymbol{\delta}_{new}\right),\; 0,\; 1\right)$$

**5. Query the ONNX engine:**

$$L_{new} = D\!\left(\operatorname{Decode}_{ONNX}(\mathbf{x}_{cand}),\; \mathbf{Y}_{gt}\right)$$

Each query executes the following ONNX call chain:

```
x_cand (prepared staff image, float32 [0,1])
  → _apply_tromr_transform(...)     → [1, 1, 256, 1280] float32
  → tromr_encoder.onnx              → context [1, 1280, 512]
  → tromr_decoder.onnx (loop)       → rhythm / pitch / lift / articulation tokens
  → HOMR vocabulary decoding        → list[EncodedSymbol]
  → statistics_engine.symbol_error_rate(...)  → scalar SER
```

**6. Greedy acceptance:**

$$\mathbf{x}^{(i)} =
\begin{cases}
\mathbf{x}_{cand} & \text{if } L_{new} > L_{best} \\
\mathbf{x}^{(i-1)} & \text{otherwise}
\end{cases}
\qquad
L_{best} \leftarrow \max(L_{best},\; L_{new})$$

**Perturbation domain convention:** The attack perturbs the prepared staff image (float32
in $[0,1]$) before the TrOMR normalization transform is applied inside
`wrapper.score_query(...)`. The TrOMR transform is applied freshly on each query candidate
inside the wrapper. The perturbation operates in image space $[0,1]$ float32, matching
the output of `prepare_staff_image(...)`. This convention must be consistent throughout;
do not mix $[0,255]$ uint8 and $[0,1]$ float32 perturbation targets without explicit
conversion.

---

## Part III — Performance Architecture

Two structural optimizations are implemented to make zero-order iteration tractable on an
8 GB VRAM laptop GPU.

### 3.1 Offline Layout Cache — Decoupling the Layout Pipeline from the Attack Loop

A naïve end-to-end forward pass of $F(\mathbf{X})$ takes approximately **6.0 seconds** per
query, dominated by SegNet's sliding-window inference over the full page followed by all
subsequent HOMR deterministic geometry.

**Solution:** Run the complete HOMR layout pipeline — SegNet ONNX inference, mask
postprocessing, staff detection, grouping, dewarping, and `prepare_staff_image(...)` —
exactly once per image on the clean dataset. Persist the resulting prepared staff images
to `dataset/cached_prepared_staffs/`. The Square Attack loop then operates exclusively on
these cached images, querying only the TrOMR encoder and decoder.

| Execution mode | Time per query |
|---|---|
| Full pipeline $F(\mathbf{X})$ | ~6.0 s |
| Cached prepared staff → TrOMR encoder + decoder only | ~0.18 s |
| **Speedup factor** | **~33×** |

**The cache target is the output of `prepare_staff_image(...)`, not a raw vertical band
crop.** `prepare_staff_image(...)` performs cropping, rescaling, dewarping, edge artifact
removal, grayscale conversion, and centering onto the TrOMR canvas (256×1280). This is
architecturally significant: the image has been processed by the entire deterministic
geometry pipeline and is in exactly the format expected by `tromr_encoder.onnx`.

Naive staff-band vertical crops are acceptable only for early smoke tests. All final Track B
measurements must use outputs of `prepare_staff_image(...)`.

The cache script stops immediately after `prepare_staff_image(...)` returns, before
`parse_staff_tromr(...)` or `Staff2Score.predict(...)` is called. TrOMR inference must
not execute inside the cache generation step.

---

### 3.2 CUDA Execution Provider Locking — Eliminating cuDNN Fallback

ONNX Runtime falls back from optimized cuDNN routines to generic CUDA loops when input
widths vary across staff images of different lengths, producing severe performance
degradation and log spam.

**Solution:** Inject a provider configuration that locks the engine to the `HEURISTIC`
algorithm search strategy, preventing continuous re-benchmarking:

```python
cuda_options = {
    "device_id": 0,
    "cudnn_conv_algo_search": "HEURISTIC",
    "arena_extend_strategy": "kNextPowerOfTwo",
}
session = ort.InferenceSession(
    model_path,
    providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
)
```

**TensorRT policy:** Although `TensorrtExecutionProvider` is detected and available on the
target machine, all benchmark runs must default to `CUDAExecutionProvider` +
`CPUExecutionProvider`. TensorRT is intentionally excluded for the following reasons:

- First-run graph compilation introduces latency of approximately 30 seconds per model.
- Compiled engine behavior may differ subtly across runs due to kernel selection.
- Benchmarking requires run-to-run consistency and reproducibility.

TensorRT compilation behavior may be measured in a separately labeled experiment if desired.
It must not be mixed with CUDAExecutionProvider results in the same benchmark table.

---

## Part IV — Complete File Map and Module Specifications

### Repository Structure

```
adversarial-homr/
├── attacks/
│   ├── config/
│   │   └── sweep_parameters.yaml
│   ├── notebooks/
│   │   ├── 01_check_onnx.ipynb
│   │   ├── 02_segmentation_smoke_test.ipynb
│   │   ├── 03_spectral_noise_injection_v2.ipynb
│   │   └── 04_tromr_encoder_smoke_test.ipynb
│   ├── src/
│   │   ├── __init__.py
│   │   ├── homr_wrapper.py
│   │   ├── segmentation_onnx.py
│   │   ├── spectral_noise.py
│   │   ├── square_attack.py
│   │   └── statistics_engine.py
│   ├── run_spectral_sweep.py
│   └── run_square_sweep.py
├── dataset/
│   ├── render_to_images.py
│   ├── cache_prepared_staffs.py
│   ├── images/
│   ├── cached_prepared_staffs/
│   └── reference_mxl/
├── models/
│   └── onnx/
│       ├── segnet.onnx
│       ├── tromr_encoder.onnx
│       └── tromr_decoder.onnx
└── results/
    ├── logs/
    └── plots/
```

---

### 4.1 `attacks/config/sweep_parameters.yaml`

**Role:** Single source of truth for all hyperparameters. Both sweep scripts load this
file at startup. No magic numbers appear elsewhere in the codebase.

```yaml
spectral_noise:
  alpha: 1.0                              # Spectral exponent (1.0 = Pink noise)
  epsilon_phys_grid: [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

square_attack:
  epsilon_math_grid: [0.01, 0.02, 0.05, 0.10, 0.20]
  n_max: 1000                             # Maximum queries per prepared staff image
  p_init: 0.8                             # Initial square size as fraction of staff image width
  p_final: 0.05                           # Minimum square size fraction

dataset:
  images_dir: "dataset/images"
  prepared_staffs_dir: "dataset/cached_prepared_staffs"
  reference_dir: "dataset/reference_mxl"
  n_images: 100

output:
  logs_dir: "results/logs"
  plots_dir: "results/plots"
  log_filename: "sweep_metrics.json"
```

---

### 4.2 `attacks/src/homr_wrapper.py`

**Role:** The single inference abstraction layer. Loads all three ONNX Runtime sessions,
applies the CUDA optimization patch, and exposes the prepared-staff prediction and
score-query interfaces consumed by the attack loop. Does not call PyTorch or TensorFlow
in any benchmark code path.

**Responsibilities:**

1. Load and configure `ort.InferenceSession` for all three ONNX models:
   - `models/onnx/segnet.onnx`
   - `models/onnx/tromr_encoder.onnx`
   - `models/onnx/tromr_decoder.onnx`

   Each session configured with `HEURISTIC` cuDNN strategy and CUDAExecutionProvider forced.

2. Use HOMR Python only for deterministic preprocessing and postprocessing:
   image resizing, CLAHE, autocrop, staff detection geometry, dewarping, TrOMR canvas
   preparation, vocabulary/token decoding utilities, and MusicXML assembly.

3. Expose the following public interface:

```python
def predict_prepared_staff(
    self,
    staff_image: np.ndarray,
    # HOMR-prepared staff image: output of prepare_staff_image(...)
    # Shape: [256, 1280] (grayscale) or [256, 1280, 1]
    # Convention: float32 in [0, 1]
) -> list:
    # Returns list[EncodedSymbol] via ONNX encoder + decoder execution
    # Does NOT call Staff2Score.predict or any PyTorch/TF inference

def score_query(
    self,
    staff_image: np.ndarray,
    target_symbols: list,
) -> float:
    # Primary (default) implementation: normalized SER or Levenshtein distance
    # between predict_prepared_staff(staff_image) and target_symbols
    # Higher value = greater divergence from ground truth
    # Optional advanced mode: multi-stream teacher-forced cross-entropy

def encoder_forward(
    self,
    staff_image: np.ndarray,
) -> np.ndarray:
    # Applies TrOMR normalization transform, runs tromr_encoder.onnx
    # Returns context tensor [1, 1280, 512]

def decoder_generate(
    self,
    context: np.ndarray,
) -> list:
    # Autoregressively decodes using tromr_decoder.onnx
    # Manages rhythm, pitch, lift, articulation token streams
    # Manages context, cache_len, cache tensors across steps
    # Returns list[EncodedSymbol]
```

4. Must not call `Staff2Score.predict(...)` in any code path used by sweeps, attacks, or
   benchmark measurement.

**ONNX boundary rule:** Neural model calls must go through ONNX Runtime. HOMR Python
functions may be used only for deterministic preprocessing, postprocessing, geometry, token
dictionaries, and MusicXML assembly.

#### 4.2.1 TrOMR Input Transform

The exact transform applied to every prepared staff image before encoding. Constants are
sourced from `Staff2Score.ConvertToArray` and `Config` and must not be changed without
verifying against HOMR source.

```python
def _apply_tromr_transform(staff_image: np.ndarray) -> np.ndarray:
    # staff_image: [H, W] or [H, W, 1], uint8 or float32 [0,1]
    arr = np.array(staff_image, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    # Ensure 2D [H, W]
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    # Add batch and channel dims → [1, 1, H, W]
    arr = arr[np.newaxis, np.newaxis, :, :]
    # Normalize with dataset statistics from Config
    arr = (arr - 0.7931) / 0.1738
    return arr.astype(np.float32)
    # Shape: [1, 1, 256, 1280], dtype: float32
```

Expected encoder input shape: `[1, 1, 256, 1280]`
Expected encoder output shape: `[1, 1280, 512]`

---

### 4.3 `attacks/src/square_attack.py`

**Role:** Self-contained implementation of the Square Attack optimization loop (§2.2).
Accepts a HOMR-prepared staff image and returns the adversarially perturbed image alongside
the full attack trajectory log.

**Public interface:**

```python
def run_square_attack(
    staff_image: np.ndarray,
    # HOMR-prepared staff image: output of prepare_staff_image(...)
    # Shape: [256, 1280] or [256, 1280, 1], float32 [0, 1]
    # TrOMR canvas size before normalization: [256, 1280]
    # ONNX encoder input after transform: [1, 1, 256, 1280]
    target_tokens: list,         # Ground-truth EncodedSymbol sequence
    wrapper: HomrWrapper,        # Inference handle (§4.2)
    epsilon: float,              # L_inf budget in [0, 1] float space
    n_max: int,                  # Query budget
    p_init: float,               # Initial square size fraction of staff image width
    p_final: float,              # Minimum square size fraction
) -> dict:
    # Returns:
    # {
    #   "x_adv": np.ndarray,            # Final adversarial staff image, same shape as input
    #   "L_best": float,                # Final achieved loss (SER or Levenshtein)
    #   "n_queries": int,               # Total ONNX queries consumed
    #   "loss_trajectory": list[float]  # L_best at each accepted improvement step
    # }
```

**Perturbation domain:** Attack operates in float32 $[0,1]$ space, matching the output of
`prepare_staff_image(...)`. The TrOMR normalization transform is applied inside
`wrapper.score_query(...)` on each candidate, not by the attack itself.

**Decay schedule implementation:**

```python
def _get_square_size(step: int, n_max: int, p_init: float,
                     p_final: float, width: int) -> int:
    # Geometrically interpolate p across the query budget
    t = step / n_max
    p = p_init * (p_final / p_init) ** t
    return max(1, round(width * p))
```

---

### 4.4 `attacks/src/statistics_engine.py`

**Role:** Centralized metric computation. Exposes SER and CER calculators backed by a
Levenshtein distance computation. All results are normalized by ground-truth length. SER
is also the primary loss proxy for `score_query` in the initial black-box Square Attack
implementation.

**Metric definitions:**

$$\text{SER} = \frac{S_{sym} + D_{sym} + I_{sym}}{N_{gt\_sym}}
\qquad
\text{CER} = \frac{S_{char} + D_{char} + I_{char}}{N_{gt\_char}}$$

where $S$, $D$, $I$ denote substitution, deletion, and insertion counts respectively from
the Levenshtein alignment of predicted vs. ground-truth sequences.

**Public interface:**

```python
def symbol_error_rate(pred_tokens: list[str], gt_tokens: list[str]) -> float: ...
    # SER from Levenshtein alignment on whitespace-tokenized sequences
    # Returns 1.0 for empty pred_tokens against non-empty gt_tokens

def character_error_rate(pred_str: str, gt_str: str) -> float: ...
    # CER from Levenshtein alignment on raw character arrays

def batch_metrics(results: list[dict]) -> dict: ...
    # Returns {"mean_ser": float, "std_ser": float,
    #          "mean_cer": float, "std_cer": float}
```

**Implementation note:** Use the `editdistance` library for the core Levenshtein
computation. For SER, tokenize both sequences on whitespace before alignment. For CER,
flatten to raw character arrays. Both functions must handle empty predicted sequences
without raising exceptions.

---

### 4.5 `attacks/src/segmentation_onnx.py`

**Role:** Standalone module encapsulating all interaction with `segnet.onnx`. Implements
HOMR-style sliding-window inference and class mask extraction so that SegNet can be tested
and run independently of HOMR Python internals.

**Key responsibilities:**

1. Load `segnet.onnx` with CUDAExecutionProvider.
2. Implement the sliding-window patch extraction that feeds `[batch_size, 3, 320, 320]`
   patches from a full-page image into SegNet.
3. Reassemble patch outputs into a full-page `[6, H, W]` score map, then apply
   `np.argmax(out, axis=0)` to obtain a `[H, W]` integer class map.
4. Extract per-class binary masks:

```python
masks = {
    "background":   class_map == 0,
    "stems_rests":  class_map == 1,
    "noteheads":    class_map == 2,
    "clefs_keys":   class_map == 3,
    "staff_lines":  class_map == 4,
    "symbols":      class_map == 5,
}
```

5. Return masks in a format compatible with HOMR's `InputPredictions` wrapper so the
   existing HOMR deterministic layout pipeline can consume them without modification.

This module is used by `dataset/cache_prepared_staffs.py` and provides a clean ONNX-only
alternative to `homr.segmentation.inference_segnet.extract(...)`.

---

### 4.6 `attacks/src/spectral_noise.py`

**Role:** Reusable library implementing the full frequency-domain noise generation pipeline
(§2.1). Extracted from the development notebook into a module so both the sweep script and
the notebook share a single tested implementation with no duplication.

**Public interface:**

```python
def generate_colored_noise(
    height: int,
    width: int,
    alpha: float,           # Spectral exponent: 0=white, 1=pink, 2=brown
    seed: int | None = None,
) -> np.ndarray:
    # Returns float32 noise array of shape [height, width], zero mean, unit variance

def inject_spectral_noise(
    image: np.ndarray,      # float32, [0, 1], shape [H, W] or [H, W, C]
    epsilon_phys: float,    # Perturbation scale
    alpha: float,
    seed: int | None = None,
) -> np.ndarray:
    # Returns float32 array same shape as image, clamped to [0, 1]
    # At epsilon_phys=0.0, output is identical to input (round-trip identity)
```

---

### 4.7 Track A — `attacks/run_spectral_sweep.py`

**Role:** Main orchestration script for Track A (Natural OOD Degradation). Iterates over
all $\epsilon_{phys}$ values in the grid, injects spectrally shaped noise into each
full-page image, runs the ONNX pipeline, computes SER/CER, and writes results to the log.

#### Staged Implementation

Track A is implemented in two stages to account for ONNX wrapper maturity:

**Stage A1 — Compatibility / Diagnostic Mode:**
Use HOMR's existing full Python pipeline (`Staff2Score.predict`) for baseline measurement
when the ONNX TrOMR wrapper is not yet complete. Label all Stage A1 results explicitly
with `"onnx_mode": false` and `"stage": "A1"` in the output JSON. Stage A1 results must
not appear in the final benchmark table without those labels.

**Stage A2 — Final ONNX Mode:**
Replace all TrOMR inference with the explicit ONNX encoder/decoder wrapper from
`homr_wrapper.py`. Keep HOMR Python for SegNet mask postprocessing, geometry, and MusicXML
assembly. Stage A2 results constitute the benchmark.

**Warning:** Do not mix Stage A1 and Stage A2 results in the same benchmark table row.

#### Final Track A Pipeline

```
Full-page image X
  → spectral_noise.inject_spectral_noise(X, epsilon_phys, alpha)
  → segmentation_onnx.SegNetONNX.infer(X_phys)
  → HOMR deterministic layout pipeline (Python geometry, no neural inference)
  → prepare_staff_image (Python)
  → tromr_encoder.onnx  → context [1, 1280, 512]
  → tromr_decoder.onnx  → token streams
  → HOMR vocabulary decoding + MusicXML assembly (Python)
  → statistics_engine.symbol_error_rate(Y_pred, Y_gt)
```

#### Execution Flow

```
1. Load sweep_parameters.yaml
2. For each epsilon_phys in epsilon_phys_grid:
   a. For each image X in dataset/images/ (n=100):
      i.   Generate colored noise: X_phys = inject_spectral_noise(X, epsilon_phys, alpha)
      ii.  Run full ONNX pipeline → Y_pred    [Stage A2]
      iii. Load Y_gt from dataset/reference_mxl/
      iv.  Compute SER, CER via statistics_engine
   b. Aggregate: mean_ser, std_ser, mean_cer, std_cer across all images
   c. Append entry {"epsilon_phys", "stage", "mean_ser", ...} to results buffer
3. Write results/logs/sweep_metrics.json (spectral_noise_results block)
4. Call plot_ser_cer_curves() → results/plots/spectral_ser_cer.html
```

---

### 4.8 Track B — `attacks/run_square_sweep.py`

**Role:** Main orchestration script for Track B (Adversarial Exploitation). Iterates over
all $\epsilon_{math}$ values, runs the Square Attack on each cached prepared staff image,
computes SER/CER on the adversarial output, and writes results to the log.

#### Execution Flow

```
1. Load sweep_parameters.yaml
2. Check dataset/cached_prepared_staffs/
   If empty or missing, run dataset/cache_prepared_staffs.py on all images first.
3. For each epsilon_math in epsilon_math_grid:
   a. For each prepared staff image x_k in cached_prepared_staffs/:
      i.   Load staff_XXX.npy (preferred over PNG to avoid encode/decode ambiguity)
      ii.  Run wrapper.predict_prepared_staff(x_k) → Y_pred_baseline
           [ONNX encoder + decoder; no PyTorch/TF]
      iii. Load aligned ground-truth token sequence from metadata.json
      iv.  Run square_attack.run_square_attack(x_k, target_tokens=Y_gt, ...) → result
           result = {x_adv, L_best, n_queries, loss_trajectory}
      v.   Run wrapper.predict_prepared_staff(x_adv) → Y_pred_adv
           [x_adv is already a prepared staff image; do NOT re-run layout pipeline]
      vi.  Compute SER, CER: statistics_engine.symbol_error_rate(Y_pred_adv, Y_gt)
      vii. Record n_queries
   b. Aggregate: mean_ser, std_ser, mean_cer, std_cer, avg_queries_to_success
   c. Append entry {"epsilon_math", "loss_proxy", "mean_ser", ...} to results buffer
4. Write results/logs/sweep_metrics.json (square_attack_results block)
5. Call plot_ser_cer_curves() → results/plots/square_ser_cer.html
```

**Key correction:** The final ONNX call on the adversarial image uses
`wrapper.predict_prepared_staff(x_adv)` because `x_adv` is already a HOMR-prepared staff
image. `wrapper.full_predict(x_adv)` must not be called here, as that would re-run the
entire layout pipeline on a staff-sized input rather than passing it directly to TrOMR.

---

### 4.9 Notebooks — `attacks/notebooks/`

Four development notebooks cover distinct validation stages and must be executed in order
before committing to full sweeps.

#### `01_check_onnx.ipynb`

**Role:** ONNX model inventory and execution provider verification.

**Mandatory cells:**

1. Verify all three ONNX files exist at `models/onnx/`.
2. Load each session with explicit CUDAExecutionProvider:

```python
cuda_options = {
    "device_id": 0,
    "cudnn_conv_algo_search": "HEURISTIC",
    "arena_extend_strategy": "kNextPowerOfTwo",
}
for name in ["segnet.onnx", "tromr_encoder.onnx", "tromr_decoder.onnx"]:
    s = ort.InferenceSession(
        f"models/onnx/{name}",
        providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
    )
    print(name)
    print("inputs: ", [(i.name, i.shape, i.type) for i in s.get_inputs()])
    print("outputs:", [(o.name, o.shape, o.type) for o in s.get_outputs()])
    print("providers:", s.get_providers())
```

**PowerShell note:** Use a Python script file or notebook cell for this check. The bash
here-string syntax `python - <<'PY'` is invalid PowerShell redirection syntax and will fail.

3. Confirm `CUDAExecutionProvider` is listed in `s.get_providers()` for each session.
4. Confirm SegNet input is `[batch_size, 3, 320, 320]` and output `[batch_size, 6, 320, 320]`.
5. Confirm TrOMR encoder input is `[1, 1, 256, 1280]` and output `[1, 1280, 512]`.
6. Confirm TrOMR decoder input names include at least: `context`, `rhythms`, `pitchs`,
   `lifts`, `articulations`, `cache_len`, `cache`.

---

#### `02_segmentation_smoke_test.ipynb`

**Role:** Visual confirmation that HOMR-style sliding-window SegNet inference produces
correct and spatially coherent class masks on a real sheet music page.

**Mandatory cells:**

1. Load one image from `dataset/images/`.
2. Run `segmentation_onnx.SegNetONNX.infer(...)` sliding-window inference.
3. Apply `np.argmax(out, axis=0)` to get the `[H, W]` integer class map.
4. Visualize each class mask side-by-side with a distinct color per class:
   - Class 0 (background) — gray
   - Class 1 (stems/rests) — blue
   - Class 2 (noteheads) — red
   - Class 3 (clefs/keys) — green
   - Class 4 (staff lines) — yellow
   - Class 5 (symbols) — magenta
5. Validate that staff-line masks (class 4) form spatially coherent horizontal bands.
6. Validate that notehead masks (class 2) cluster at expected vertical positions.
7. Confirm the class map contains no values outside `{0, 1, 2, 3, 4, 5}`.

---

#### `03_spectral_noise_injection_v2.ipynb`

**Role:** Development and visual verification of the frequency-domain noise generation
pipeline. This notebook is the reference implementation from which `spectral_noise.py`
is extracted. It supersedes any earlier version of this notebook.

**Mandatory cells:**

1. Single-image noise generation at $\alpha \in \{0, 1, 2\}$ — display original and three
   noisy versions side-by-side.
2. PSD verification plot — log-log frequency vs. amplitude for each $\alpha$, confirming
   the $1/f^\alpha$ slope visually.
3. Perturbed image quality audit — confirm musical notation remains visually legible at
   $\epsilon_{phys} \in \{0.05, 0.10\}$.
4. Edge case validation: zero-image, all-white image, and narrow-crop inputs.
5. Round-trip identity test: confirm that at $\epsilon_{phys} = 0.0$, the output is
   numerically identical to the input (within floating-point precision).

---

#### `04_tromr_encoder_smoke_test.ipynb`

**Role:** End-to-end validation of the TrOMR encoder ONNX session on a real cached
prepared staff image. Optionally exercises a single decoder step.

**Mandatory cells:**

1. Load `dataset/cached_prepared_staffs/<score>/staff_000.npy`.
2. Apply `_apply_tromr_transform(...)` → assert shape `[1, 1, 256, 1280]`, dtype `float32`.
3. Run `tromr_encoder.onnx` session with CUDAExecutionProvider.
4. Assert output shape is `[1, 1280, 512]` and dtype `float32`.
5. Print `min`, `max`, `mean` of the context tensor. Non-degenerate values (not all zeros
   or NaN) confirm the encoder is working correctly.
6. Optional: Pass context into `tromr_decoder.onnx` for one autoregressive step with
   start tokens. Print the first predicted token for each stream (rhythm, pitch, lift,
   articulation). Confirm the output tokens are valid vocabulary indices.

---

### 4.10 `attacks/src/__init__.py`

**Role:** Package marker. Re-exports the public interfaces for clean imports.

```python
from .homr_wrapper import HomrWrapper
from .square_attack import run_square_attack
from .statistics_engine import symbol_error_rate, character_error_rate, batch_metrics
from .segmentation_onnx import SegNetONNX
from .spectral_noise import generate_colored_noise, inject_spectral_noise
```

---

### 4.11 `dataset/render_to_images.py`

**Role:** Converts MusicXML/MXL files into PNG page images using MuseScore.

**Purpose:** Produces `dataset/images/*.png`, which become the full-page inputs for Track A
and the source images for the prepared-staff cache used in Track B.

**Behavior:**
- Recursively scans `dataset/mxl/` for source MusicXML/MXL files.
- Uses `PDMX.csv` to resolve canonical source file paths.
- Uses checkpoint file `render_checkpoint.txt` to avoid re-rendering already completed
  scores, enabling crash recovery and incremental dataset growth.
- Supports parallel MuseScore rendering across CPU cores.
- Output images are saved to `dataset/images/` named by score stem.

---

### 4.12 `dataset/cache_prepared_staffs.py`

**Role:** Runs HOMR's real pre-TrOMR pipeline on every image in `dataset/images/` and
caches the exact prepared staff images that would normally be passed into TrOMR.

**Execution path:**

```
image_path
  → homr.main.detect_staffs_in_image(...)
  → SegNet ONNX via segmentation_onnx.SegNetONNX (or homr.segmentation.inference_segnet)
  → HOMR mask filtering, staff detection, staff grouping (Python, deterministic)
  → homr.staff_parsing.prepare_staff_image(...) for each staff k
  → save staff_k.png      (visual inspection copy)
  → save staff_k.npy      (exact numerical float32 array for attacks)
  → save metadata.json    (staff index, geometry info, ground-truth token list)
  → STOP — do not call parse_staff_tromr(...) or Staff2Score.predict(...)
```

**Critical constraint:** This script must stop immediately after `prepare_staff_image(...)`
returns. `parse_staff_tromr(...)` and `Staff2Score.predict(...)` must not execute. The
purpose of this script is to produce the TrOMR input, not the TrOMR output.

---

### 4.13 `dataset/cached_prepared_staffs/` — Directory Specification

**Format:** One subdirectory per source image, named by image stem.

```
cached_prepared_staffs/
└── score_001/
    ├── metadata.json
    ├── staff_000.png      # visual inspection copy — grayscale uint8
    ├── staff_000.npy      # exact numerical array — float32 [0, 1], shape [256, 1280]
    ├── staff_001.png
    ├── staff_001.npy
    └── ...
```

**metadata.json schema:**

```json
{
  "image_stem": "score_001",
  "source_image": "dataset/images/score_001.png",
  "n_staffs": 4,
  "staffs": [
    {
      "index": 0,
      "filename_npy": "staff_000.npy",
      "filename_png": "staff_000.png",
      "shape": [256, 1280],
      "gt_tokens": ["clef_G", "time_4/4", "note_C4_quarter", "..."]
    }
  ]
}
```

Each staff image is the output of HOMR's `prepare_staff_image(...)`, not a raw
segmentation crop. The `.npy` file is the **preferred attack input** because it avoids
repeated PNG encode/decode ambiguity that would alter pixel values. The `.png` file is
retained only for visual inspection and debugging.

---

### 4.14 `results/logs/sweep_metrics.json` — Output Schema

Complete schema covering both tracks:

```json
{
  "experiment_metadata": {
    "target_hardware": "NVIDIA GeForce RTX 3070 Laptop GPU",
    "onnxruntime_version": "<filled at runtime>",
    "execution_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "onnx_mode": true,
    "total_images_evaluated": 100,
    "spectral_alpha": 1.0,
    "square_n_max": 1000
  },
  "spectral_noise_results": [
    {
      "epsilon_phys": 0.0,
      "stage": "A2",
      "onnx_mode": true,
      "mean_ser": 0.042,
      "std_ser": 0.011,
      "mean_cer": 0.015,
      "std_cer": 0.006,
      "total_queries": 100
    }
  ],
  "square_attack_results": [
    {
      "epsilon_math": 0.02,
      "loss_proxy": "SER",
      "mean_ser": 0.312,
      "std_ser": 0.047,
      "mean_cer": 0.145,
      "std_cer": 0.029,
      "avg_queries_to_success": 241.6
    }
  ]
}
```

The `"stage"` field in spectral results (`"A1"` or `"A2"`) and the `"loss_proxy"` field
in square attack results (`"SER"`, `"CER"`, or `"multi_stream_CE"`) must be populated
truthfully and must not be omitted from any result entry. Results with different stages
or loss proxies must never be aggregated into the same table row.

---

## Part V — Environment Setup

### 5.1 Verified Runtime State

The following provider configuration has been confirmed on the target machine:

```
Available providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
Benchmark forced to: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

TensorRT is present and would be used preferentially by ONNX Runtime if not explicitly
overridden. All benchmark `InferenceSession` calls must override this by passing:

```python
providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
```

This avoids first-run TensorRT graph compilation latency (~30 s per model) and ensures
reproducible performance across runs.

---

### 5.2 Full Environment Specification

```powershell
# 1. Create and activate isolated environment (Python 3.11 required)
conda create -n adv-homr python=3.11 -y
conda activate adv-homr

# 2. Install benchmark dependencies with pinned critical versions
pip install `
    onnxruntime-gpu `
    "numpy==1.26.4" `
    opencv-python `
    pillow `
    editdistance `
    plotly `
    pandas `
    pyyaml `
    tqdm `
    ipykernel

# 3. Install HOMR in editable mode (required for deterministic preprocessing/postprocessing)
# HOMR's geometry, vocabulary, and MusicXML code is reused directly in benchmark mode.
pip install -e /path/to/homr

# 4. Register project root on PYTHONPATH
$env:PYTHONPATH = "D:\Users\theda\Documents\dev\Projects\Uni\Semester 4\Artificial Intelligence\adversarial-homr"

# 5. Persist PYTHONPATH across sessions (optional)
[System.Environment]::SetEnvironmentVariable(
    "PYTHONPATH",
    $env:PYTHONPATH,
    "User"
)

# 6. Verify GPU execution provider
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expected: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# 7. Verify numpy version (must be 1.x; NumPy 2.x breaks ONNX Runtime array layout)
python -c "import numpy as np; print(np.__version__)"
# Expected: 1.26.4
```

**HOMR dependency caveat:** Because the benchmark reuses HOMR's deterministic
preprocessing and postprocessing code (geometry, vocabulary, MusicXML assembly), the
environment must satisfy HOMR's own package requirements in addition to the packages above.
These include OpenCV, NumPy, Pillow, and ONNX Runtime at minimum. If title detection or
OCR is not part of the benchmark workflow, disable or bypass HOMR's title detection path
where possible to avoid loading unrelated OCR dependencies.

---

### 5.3 Dependency Rationale

| Package | Version | Reason |
|---|---|---|
| `onnxruntime-gpu` | latest | ONNX graph execution on RTX 3070 for all three models |
| `numpy` | ==1.26.4 | Pin to 1.x; NumPy 2.x breaks ONNX Runtime internal array layout |
| `opencv-python` | latest | Image I/O, resizing, CLAHE, contour extraction in geometry pipeline |
| `pillow` | latest | Image loading and PNG encode/decode for cache file management |
| `editdistance` | latest | Fast C-backed Levenshtein for SER/CER computation |
| `plotly` | latest | Interactive SER/CER vs. ε plots in `results/plots/` |
| `pandas` | latest | Result aggregation and CSV export |
| `pyyaml` | latest | Config file parsing for `sweep_parameters.yaml` |
| `tqdm` | latest | Progress bars for long sweeps |
| `ipykernel` | latest | Notebook execution in conda environment |
| HOMR (editable) | source | Deterministic preprocessing, geometry, vocabulary, MusicXML assembly |

---

## Part VI — Evaluation Metrics

### 6.1 Symbol Error Rate (SER)

Tracks structural accuracy at the musical token level. Tokens are musical primitives:
clefs, note names with octave and duration, rests, time signatures, barlines. SER is also
the primary loss proxy for `score_query` in the initial Square Attack implementation.

$$\text{SER} = \frac{S_{sym} + D_{sym} + I_{sym}}{N_{gt\_sym}}$$

where $S$, $D$, $I$ denote substitutions, deletions, and insertions from the Levenshtein
alignment of the predicted token sequence against the ground-truth token sequence.

---

### 6.2 Character Error Rate (CER)

Tracks precision at the raw character level by flattening all token strings into a single
character sequence before alignment.

$$\text{CER} = \frac{S_{char} + D_{char} + I_{char}}{N_{gt\_char}}$$

CER is more sensitive to partial token corruption (e.g., correct note name but wrong octave
or duration). Both SER and CER are reported for all experiments.

---

### 6.3 Query Efficiency (Track B only)

Average number of ONNX engine calls consumed before the attack reaches a loss value
$L \ge L_{threshold}$ (defined as any improvement over the clean baseline):

$$\bar{Q} = \frac{1}{|\mathcal{D}|} \sum_{i \in \mathcal{D}} q_i$$

A lower $\bar{Q}$ at a given $\epsilon_{math}$ indicates higher exploitability of the
recognition module at that perturbation budget.

---

### 6.4 Interpretation Framework

| Metric | $\epsilon = 0$ (baseline) | Moderate $\epsilon$ | High $\epsilon$ |
|---|---|---|---|
| SER | System floor (~0.04) | Gradual structural breakdown | Full token sequence collapse |
| CER | System floor (~0.015) | Character-level drift | Unintelligible output |
| $\bar{Q}$ | — | 200–400 queries | <100 queries |

---

### 6.5 Multi-Stream Token Error (Advanced — Track B only)

When teacher-forced ONNX decoder cross-entropy is implemented, per-stream error rates
can be reported independently:

$$\text{SER}_r,\quad \text{SER}_p,\quad \text{SER}_l,\quad \text{SER}_a$$

for rhythm, pitch, lift, and articulation streams respectively. This decomposition reveals
which musical dimension (temporal structure vs. pitch vs. ornament) is most vulnerable to
adversarial perturbation. This metric is not part of the initial benchmark but should be
included in the final report if teacher-forced mode is implemented.

---

## Part VII — Immediate Next Steps

Execute in strict order. Do not advance to a later step until the earlier step passes its
stated validation criterion.

**Step 1 — Verify ONNX model files and tensor shapes**

Confirm all three ONNX files are present in `models/onnx/`:
`segnet.onnx`, `tromr_encoder.onnx`, `tromr_decoder.onnx`.

Run `01_check_onnx.ipynb`. For each model, print input/output tensor names, shapes, and
dtypes. Confirm `CUDAExecutionProvider` is active via `s.get_providers()`. Confirm SegNet
input is `[batch_size, 3, 320, 320]`, TrOMR encoder input is `[1, 1, 256, 1280]`, and
TrOMR encoder output is `[1, 1280, 512]`. Confirm decoder input names include `context`,
`rhythms`, `pitchs`, `lifts`, `articulations`, `cache_len`, `cache`.

*PowerShell note:* Use a notebook cell or `.py` script file for this verification, not
a bash here-string. `python - <<'PY'` is invalid PowerShell syntax.

**Step 2 — Run segmentation smoke test**

Run `02_segmentation_smoke_test.ipynb`. Confirm sliding-window SegNet inference produces
integer class maps in `{0,1,2,3,4,5}`. Confirm staff-line masks (class 4) are spatially
coherent horizontal bands. This validates `segmentation_onnx.py`.

**Step 3 — Implement and run `dataset/cache_prepared_staffs.py`**

Implement using `homr.main.detect_staffs_in_image(...)` and
`homr.staff_parsing.prepare_staff_image(...)`. Stop before `parse_staff_tromr(...)`.
Save `staff_000.png`, `staff_000.npy`, and `metadata.json`.

Run on one image. Inspect `staff_000.png` — it should be a dewarped, clean,
single-staff grayscale image approximately 256 pixels tall. Inspect `metadata.json` —
it should show correct ground-truth token list and staff shape `[256, 1280]`.

**Step 4 — Run TrOMR encoder smoke test**

Run `04_tromr_encoder_smoke_test.ipynb`. Load `staff_000.npy`. Apply
`_apply_tromr_transform(...)`. Feed into `tromr_encoder.onnx`. Assert output shape is
`[1, 1280, 512]`. Print `min`/`max`/`mean` of the context tensor. Non-degenerate values
(not all zeros, not NaN) confirm the encoder is functioning correctly.

**Step 5 — Implement ONNX TrOMR decoder generation**

Implement `wrapper.decoder_generate(context)` using `tromr_decoder.onnx`. Reproduce the
autoregressive generation behavior of `Staff2Score.generate(...)`, managing all four token
streams: rhythm, pitch, lift, articulation, and the `context`, `cache_len`, `cache` tensors
across decoding steps.

Validate by comparing decoder output against a HOMR Python baseline prediction on the same
staff image. Token sequences should match modulo vocabulary alignment.

**Step 6 — Implement `wrapper.predict_prepared_staff(...)`**

Wire encoder and decoder together in a single method:

```
prepared staff image
  → _apply_tromr_transform
  → tromr_encoder.onnx
  → tromr_decoder.onnx (autoregressive loop)
  → HOMR vocabulary decoding
  → list[EncodedSymbol]
```

Validate on several cached staff images from different scores. Compare against Python
baseline. SER between ONNX prediction and Python prediction should be near 0.

**Step 7 — Implement `wrapper.score_query(...)`**

Initial version: call `predict_prepared_staff(staff_image)` and compute normalized
Levenshtein/SER against `target_symbols` using `statistics_engine.symbol_error_rate(...)`.
Return the scalar SER.

Smoke test: `score_query(clean_staff, gt_tokens)` should return approximately the system
floor (~0.04). `score_query(all_zeros_staff, gt_tokens)` should return a substantially
higher value, confirming the score function is sensitive to image content.

**Step 8 — Run Square Attack smoke test**

Run `run_square_attack(...)` on one prepared staff image with `n_max=50` and
`epsilon=0.02`. Confirm:
- `n_queries` is between 1 and 50.
- `L_best` is greater than or equal to `L` at step 0.
- `loss_trajectory` is monotonically non-decreasing.
- `x_adv` differs from input by at most `epsilon` in $L_\infty$ norm.

**Step 9 — Verify spectral noise pipeline**

Run `03_spectral_noise_injection_v2.ipynb`. Confirm PSD plots show correct $1/f^\alpha$
slopes for $\alpha \in \{0, 1, 2\}$. Confirm perturbed images at $\epsilon_{phys} = 0.10$
remain visually legible. Extract `spectral_noise.py` from the notebook.

Run `run_spectral_sweep.py` in Stage A1 mode on 5 images as a JSON schema validation pass.
Confirm all expected fields appear in `sweep_metrics.json`.

**Step 10 — Run small benchmark sweeps**

Run Track A (Stage A2, ONNX mode) on 10 images across all $\epsilon_{phys}$ values.
Confirm JSON schema is correct and all fields are populated with `"stage": "A2"` and
`"onnx_mode": true`. Run Track B on 5 prepared staff images at one $\epsilon_{math}$ value.
Confirm `loss_proxy` field is `"SER"` in output JSON.

**Step 11 — Full benchmark sweeps**

Restore `n_images: 100` in `sweep_parameters.yaml`. Run Track A and Track B full sweeps.
Expected runtime: approximately 2–4 hours depending on staff count per image and query
budget.

**Step 12 — Advanced mode (optional)**

Implement teacher-forced multi-stream cross-entropy in `score_query`. Re-run Track B with
the advanced loss proxy. Report per-stream error rates $\text{SER}_r$, $\text{SER}_p$,
$\text{SER}_l$, $\text{SER}_a$. Label all results with `"loss_proxy": "multi_stream_CE"`.

---

*Blueprint version 3.0 — Fully integrated revision. All sections reflect the validated
HOMR ONNX architecture: SegNet six-class semantic segmentation, deterministic HOMR layout
pipeline (G), prepared staff image cache (output of `prepare_staff_image`), and multi-stream
TrOMR encoder/decoder. All prior references to Oemer, a single `tromr.onnx`, naive staff
band crops, and cross-entropy as the default loss proxy have been corrected and replaced
throughout. This document supersedes all prior versions.*
