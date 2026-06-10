# Student Architecture: FullPageHOMRSurrogate / PageStaffARStudent

Status: design specification  
Target implementation: `distillation/models/page_staff_ar_student.py`  
Trainer integration: `distillation/train_student.py`  
Primary role: differentiable surrogate for HOMR, trained by HOMR teacher outputs

---

## 1. Purpose

The student is not intended to be an independent OMR model trained directly against ideal ground truth. Its purpose is to be a differentiable behavioral surrogate for HOMR.

The surrogate target is:

```text
student(image) ≈ HOMR(image)
```

not:

```text
student(image) ≈ ideal human-correct transcription
```

The student exists so that later adversarial attacks can be optimized with gradients through the student and then evaluated against HOMR. Therefore, the student must approximate HOMR's input-output behavior and must preserve HOMR-compatible output structure.

The student should remain full-page. HOMR itself is a full-page system at the system level:

```text
full page image
  -> segmentation / layout / staff reconstruction
  -> staff dewarping / extraction
  -> TrOMR-like recognition per staff
  -> page-level symbolic assembly
  -> MusicXML
```

The student should not be staff-only. It should accept a full rendered/photographed/scanned page image and output ordered staff-level HOMR-compatible token sequences.

---

## 2. Research framing

The project should describe this model as a query-based behavioral distillation surrogate:

```text
Inputs:
  images selected by us: clean renders, degraded renders, or external full-page images

Teacher labels:
  HOMR outputs on those exact images

Student objective:
  imitate HOMR's output behavior

Attack objective:
  generate perturbations using student gradients and measure transfer to HOMR
```

This is different from retraining HOMR or reproducing HOMR's original training. It is also different from training a new OMR model on dataset ground truth.

The experimental claim should be tied to the query/data regime:

```text
S0: clean-render HOMR-labeled surrogate
S1: clean + camera/degraded HOMR-labeled surrogate
S2: public/external full-page HOMR-labeled surrogate
S3: adversarially queried / adversarially trained surrogate
```

For the immediate baseline, S0 and then S1 are enough.

---

## 3. Literature grounding

The architecture is a hybrid because no single referenced model matches all constraints:

```text
required constraints:
  full-page input
  HOMR-compatible ordered staff output
  TrOMR/HOMR factorized symbol prediction
  differentiable surrogate for adversarial attacks
  no invented <STAFF_BREAK> vocabulary token
  no learned newline token inside staff vocabulary
```

Relevant architectural sources:

1. **TrOMR** motivates Transformer-based OMR and factorized/autoregressive music-symbol prediction. The student should use TrOMR/HOMR-like factorized branches for rhythm, pitch, lift, articulation, and position.
2. **Sheet Music Transformer / SMT** motivates image-to-sequence Transformer modeling for polyphonic OMR.
3. **SMT++ / full-page OMR work** motivates full-page autoregressive modeling, CNN visual front ends, and curriculum-like staged training.
4. **DETR** motivates learned query slots over image memory. The student adapts this idea as ordered staff slots rather than object slots.
5. **HOMR** motivates preserving full-page input, staff-structured processing, and `EncodedSymbol("newline")` only during page reconstruction.

References are listed at the end of this document.

---

## 4. Non-goals

The student should not do the following:

```text
Do not train from dataset ground truth as the main target.
Do not use <STAFF_BREAK>.
Do not add a custom staff-break vocabulary token.
Do not treat HOMR newline as a normal TrOMR staff-level token.
Do not flatten all staff targets into one giant learned page sequence for v1.
Do not claim the current smoke trainer is the final architecture.
Do not differentiate through MusicXML generation in v1.
```

The current `train_student.py` smoke model was useful for proving the data path, CUDA training, logging, and checkpoint writing. It is not the final architecture.

---

## 5. Input/output contract

### 5.1 Input

The model receives a full-page grayscale score image:

```text
image: [B, 1, H, W]
```

Example:

```text
[B, 1, 768, 512]
```

where:

```text
B = batch size
1 = grayscale channel
H, W = resized full-page image dimensions
```

The input image may be:

```text
clean render
camera/degraded render
external full-page OMR image
later: adversarially perturbed image
```

### 5.2 Output

The model outputs an ordered list of staff sequences.

```text
page:
  staff 0: HOMR factorized symbol sequence
  staff 1: HOMR factorized symbol sequence
  staff 2: HOMR factorized symbol sequence
  ...
```

The neural output is not a flat page stream.

The predicted tensor shapes are:

```text
rhythm_logits       [B, S_max, T_max, V_rhythm]
pitch_logits        [B, S_max, T_max, V_pitch]
lift_logits         [B, S_max, T_max, V_lift]
articulation_logits [B, S_max, T_max, V_articulation]
position_logits     [B, S_max, T_max, V_position]

staff_exists_logits [B, S_max]
staff_count_logits  [B, S_max + 1]
```

where:

```text
S_max = maximum number of staff sequences on a page
T_max = maximum decoded token length per staff
V_*   = vocabulary size for each HOMR factor branch
```

---

## 6. Target representation

The encoded manifest should keep the current structural multi-staff form:

```json
{
  "schema": "homr_factorized_page_training_manifest_v1",
  "image_path": "...",
  "n_staffs": 4,
  "homr_target_staffs": [
    {
      "staff_index": 0,
      "rhythm_ids": [],
      "pitch_ids": [],
      "lift_ids": [],
      "articulation_ids": [],
      "position_ids": [],
      "mask": []
    }
  ],
  "homr_page_structure": {
    "postprocess_separator": {
      "rhythm": "newline",
      "encoded_in_transformer_vocabulary": false
    }
  }
}
```

Each staff sequence already includes BOS/EOS in the factorized HOMR branch IDs.

The model learns staff sequences only. It does not learn a page-level newline token.

---

## 7. Newline / staff separator policy

There is no `<STAFF_BREAK>` token.

HOMR page assembly uses `EncodedSymbol("newline")` as a page-level separator. However, this separator is not part of the TrOMR staff-level vocabulary. Therefore:

```text
Training target:
  ordered list of staff sequences

Model output:
  ordered list of staff sequences

Postprocessing:
  insert HOMR EncodedSymbol("newline") between or after staff sequences
  according to HOMR-compatible reconstruction behavior
```

During reconstruction:

```python
page_symbols = []

for staff_symbols in predicted_staffs:
    page_symbols.extend(staff_symbols)
    page_symbols.append(EncodedSymbol("newline"))
```

Whether the final staff should have a trailing newline should be made compatible with the HOMR assembly path used by evaluation/export.

---

## 8. Architecture overview

The final v1 architecture is:

```text
FullPageHOMRSurrogate / PageStaffARStudent

full-page image
  -> visual page encoder
  -> 2D positional page memory
  -> layout-aware Transformer encoder
  -> ordered staff-slot decoder
  -> autoregressive per-staff symbol decoder
  -> HOMR/TrOMR factorized output heads
  -> ordered staff token sequences
  -> HOMR newline postprocessing for reconstruction
```

Short form:

```text
CNN/ConvNeXt page encoder
+ Transformer page memory
+ DETR-inspired ordered staff queries
+ shared autoregressive per-staff decoder
+ HOMR factorized heads
```

---

## 9. Component details

### 9.1 Visual page encoder

Input:

```text
image [B, 1, H, W]
```

Output:

```text
page_features [B, C, H', W']
```

Example:

```text
[B, 1, 768, 512]
  -> [B, 256, 48, 32]
```

The visual encoder should preserve spatial layout. Recommended v1 choices:

```text
small ResNet-like CNN
or lightweight ConvNeXt-style CNN
```

Do not begin with an oversized ViT. Sheet music has fine local structures, and CNN inductive bias is useful for staff lines, stems, noteheads, beams, and accidentals.

### 9.2 Page memory

Transformers attend over sequences. The 2D feature map is flattened:

```text
page_features [B, C, H', W']
  -> memory [B, H'W', D]
```

Example:

```text
[B, 256, 48, 32]
  -> [B, 1536, 256]
```

Each memory token corresponds to a page-region visual descriptor.

2D positional encoding is required because flattening removes explicit row/column position. Use either:

```text
fixed 2D sine/cosine positional encoding
or learned 2D row/column embeddings
```

### 9.3 Layout-aware Transformer encoder

The flattened page memory is passed through a Transformer encoder:

```text
memory [B, H'W', D]
  -> layout_memory [B, H'W', D]
```

Purpose:

```text
allow distant page regions to communicate
model staff/system relationships
model horizontal and vertical score layout
support global page context for staff selection
```

### 9.4 Ordered staff-slot decoder

Define a maximum number of staff sequences per page:

```text
S_max
```

Example:

```text
S_max = 16
```

Create learned staff query embeddings:

```text
staff_queries [S_max, D]
```

Each staff query cross-attends to the full layout memory:

```text
staff_queries + layout_memory
  -> staff_slots [B, S_max, D]
```

The staff slots should be interpreted as ordered staff candidates:

```text
slot 0 -> first staff sequence
slot 1 -> second staff sequence
slot 2 -> third staff sequence
...
```

This is inspired by DETR-style learned object queries, but adapted to ordered staff slots. Unlike DETR object detection, we do not treat staff slots as an unordered set for v1; ordering is part of the target contract.

Auxiliary heads:

```text
staff_exists_logits [B, S_max]
staff_count_logits  [B, S_max + 1]
```

Optional future head if reliable labels exist:

```text
staff_box_pred [B, S_max, 4]
```

### 9.5 Per-staff autoregressive decoder

Each staff slot conditions a shared autoregressive decoder.

For a staff sequence:

```text
BOS, symbol_1, symbol_2, ..., symbol_n, EOS
```

Training uses teacher forcing:

```text
decoder input:
  BOS, symbol_1, symbol_2, ..., symbol_{n-1}

decoder target:
  symbol_1, symbol_2, ..., symbol_n, EOS
```

Each previous symbol is factorized into:

```text
rhythm_id
pitch_id
lift_id
articulation_id
position_id
```

The previous-symbol embedding is:

```text
symbol_embedding =
  rhythm_embedding[rhythm_id]
+ pitch_embedding[pitch_id]
+ lift_embedding[lift_id]
+ articulation_embedding[articulation_id]
+ position_embedding[position_id]
```

The decoder conditions on:

```text
previous symbol embeddings
+ staff slot state
+ layout page memory
+ token positional encoding
```

Output:

```text
token_states [B, S_max, T_max, D]
```

### 9.6 Factorized output heads

For each token state, apply five heads:

```text
rhythm_head       -> rhythm_logits
pitch_head        -> pitch_logits
lift_head         -> lift_logits
articulation_head -> articulation_logits
position_head     -> position_logits
```

This keeps the output aligned with HOMR/TrOMR factorization.

The model does not use a single flat class for full symbol strings.

---

## 10. Differentiability

The neural training path is differentiable:

```text
image
  -> encoder
  -> page memory
  -> layout encoder
  -> staff slots
  -> autoregressive decoder under teacher forcing
  -> logits
  -> loss
```

Gradients flow through:

```text
visual encoder
positional projections
Transformer encoder
staff-slot decoder
symbol embeddings
autoregressive decoder
output heads
staff existence/count heads
```

Not differentiable, and not part of training loss in v1:

```text
argmax / beam decoding
symbolic newline insertion
MusicXML generation
edit distance metrics
```

That is acceptable. Those are inference/evaluation/postprocessing steps.

---

## 11. Losses

### 11.1 Main token loss

The main loss is TrOMR-like shifted masked factorized cross-entropy.

For each valid non-padding target position:

```text
L_token =
  CE(rhythm_logits, rhythm_target)
+ CE(pitch_logits, pitch_target)
+ CE(lift_logits, lift_target)
+ CE(articulation_logits, articulation_target)
+ CE(position_logits, position_target)
```

Only positions with `mask == true` count.

Padding positions do not contribute.

### 11.2 Staff existence loss

For each staff slot:

```text
staff_exists_target = 1 for real staff slots
staff_exists_target = 0 for padded staff slots
```

Loss:

```text
L_exists = binary_cross_entropy_with_logits(
  staff_exists_logits,
  staff_exists_target
)
```

### 11.3 Staff count loss

For each page:

```text
n_staffs = number of target staff sequences
```

Loss:

```text
L_count = CE(staff_count_logits, n_staffs)
```

### 11.4 Optional staff box loss

If reliable staff boxes are exposed from preprocessing metadata:

```text
L_box = L1(pred_box, target_box) + generalized IoU loss
```

This should be optional and not required for v1.

### 11.5 Total loss

```text
L_total =
  L_token
+ λ_exists * L_exists
+ λ_count  * L_count
+ optional λ_box * L_box
```

Recommended initial weights:

```text
λ_exists = 0.1
λ_count  = 0.1
λ_box    = disabled
```

Weights should be logged in `metrics.json`.

---

## 12. Metrics

Training loss alone is not enough. The trainer/evaluator should eventually report:

### 12.1 Teacher-forced metrics

```text
loss
rhythm_loss
pitch_loss
lift_loss
articulation_loss
position_loss
staff_exists_loss
staff_count_loss
```

### 12.2 Branch accuracies

```text
rhythm_accuracy
pitch_accuracy
lift_accuracy
articulation_accuracy
position_accuracy
```

Computed over valid masked positions only.

### 12.3 Symbol exact-match

A symbol is correct only if all five branches are correct:

```text
rhythm correct
and pitch correct
and lift correct
and articulation correct
and position correct
```

### 12.4 Sequence metrics

After decoding:

```text
staff-level edit distance
normalized staff-level edit distance
page-level edit distance after HOMR newline reconstruction
EOS accuracy
sequence length error
```

### 12.5 HOMR surrogate metrics

Since this is a HOMR surrogate:

```text
student-HOMR agreement on clean images
student-HOMR agreement on degraded images
student-HOMR agreement under spectral-noise attacks
student-HOMR agreement under square attack samples
student-to-HOMR attack transfer success
```

The project should not only report ground-truth OMR quality.

---

## 13. Inference procedure

Inference:

```text
1. Load full page image.
2. Encode page into layout memory.
3. Predict staff slots and staff existence.
4. Keep ordered slots where staff_exists is above threshold,
   or use predicted staff_count.
5. For each kept staff slot:
     autoregressively generate factorized symbols until EOS or T_max.
6. Convert predicted IDs to HOMR EncodedSymbol-like symbols.
7. Insert EncodedSymbol("newline") during page reconstruction.
8. Export/evaluate via HOMR-compatible MusicXML path.
```

Staff slot selection policy for v1:

```text
use staff_count prediction if reliable;
otherwise threshold staff_exists logits;
preserve slot order.
```

For the first implementation, teacher-forced validation is enough. Full autoregressive decoding can be added immediately after the model trains.

---

## 14. Data regimes

### 14.1 Current clean baseline

The current clean-render pipeline remains the baseline:

```text
rendered clean pages
  -> HOMR teacher
  -> teacher JSON
  -> manifest
  -> split
  -> vocab encoding
  -> student training
```

This baseline has already passed infrastructure smoke with the temporary trainer:

```text
rows: 59
staff_count: 345
token_count: 12466
loss: 9.110053418046338
outputs:
  latest.pt
  training_log.jsonl
  metrics.json
```

This proves the pipeline works, not the final architecture.

### 14.2 Sequential degradation data

Clean renders are too clean to represent HOMR's full-page capability. HOMR can handle imperfect images, staff geometry, perspective, and dewarping. Therefore the pipeline should add a sequential degradation stage.

The degradation pipeline should be:

```text
clean render
  -> geometry transform
  -> paper/ink degradation
  -> lighting/shadow field
  -> spectral physical texture/noise
  -> blur
  -> compression
  -> final image
  -> HOMR teacher
```

The exact final transformed image must be labeled by HOMR.

Do not label the clean image and then reuse that label after transformation.

### 14.3 Sequential degradation families

Recommended v1 families:

```text
geometry:
  rotation
  translation
  scale
  skew
  perspective

paper/ink:
  tint
  contrast reduction
  mild ink fading
  mild ink thickening/thinning
  low-frequency paper texture

lighting:
  brightness shift
  contrast shift
  illumination gradient
  vignette
  soft shadow band

spectral physical texture/noise:
  alpha-conditioned white/pink/brown spectral field
  epsilon-controlled strength

optics:
  Gaussian blur
  optional mild motion blur later

compression:
  JPEG quality degradation
```

The spectral-noise method is usable as one component of physical document corruption. It should not be the only degradation family because it does not test perspective/dewarping.

### 14.4 Parameter-space mixture, not image-space blending

Do not form:

```text
final = w1 * warped + w2 * blurred + w3 * noisy
```

Instead use a sequential pipeline and sample per-family strengths:

```text
image_0 = clean
image_1 = T_geometry(image_0; θ_geometry)
image_2 = T_paper_ink(image_1; θ_paper_ink)
image_3 = T_lighting(image_2; θ_lighting)
image_4 = T_spectral(image_3; θ_spectral)
image_5 = T_blur(image_4; θ_blur)
image_6 = T_compression(image_5; θ_compression)
```

Use a global severity plus per-family sampled weights:

```text
global_severity ~ Uniform(profile_min, profile_max)
family_weights  ~ Dirichlet(...)
effective_strength_i = global_severity * (floor_i + residual_weight_i)
```

All families are present, but with varying intensities.

### 14.5 Recommended degradation profiles

`camera_light_v1`:

```text
global severity: 0.15-0.35
rotation: up to ±1.5 degrees
perspective: up to 2%
spectral epsilon: up to 0.03
blur sigma: up to 0.5
JPEG quality: 80-95
shadow: mild
```

`camera_medium_v1`:

```text
global severity: 0.35-0.65
rotation: up to ±3 degrees
perspective: up to 4%
spectral epsilon: up to 0.07
blur sigma: up to 1.0
JPEG quality: 65-90
shadow: moderate
```

`camera_hard_v1`:

```text
global severity: 0.65-1.0
rotation: up to ±5 degrees
perspective: up to 7%
spectral epsilon: up to 0.12
blur sigma: up to 1.5
JPEG quality: 45-80
shadow: strong
```

Use `camera_light_v1` first.

---

## 15. Pipeline changes required before large-scale training

### 15.1 Add `distillation/augment_pages.py`

Input:

```text
distillation/batches/<batch>/logs/render_log.jsonl
```

Output:

```text
distillation/batches/<batch>_camera_light/augmented_pages/*.png
distillation/batches/<batch>_camera_light/logs/augment_log.jsonl
distillation/batches/<batch>_camera_light/augmentation_summary.json
```

Example:

```powershell
python distillation/augment_pages.py `
  --render-log distillation/batches/batch_000000_smoke/logs/render_log.jsonl `
  --out-dir distillation/batches/batch_000000_smoke_camera_light `
  --profile camera_light_v1 `
  --variants-per-page 1 `
  --seed 1337 `
  --progress-every 1
```

### 15.2 Modify `distillation/run_onnx_teacher_batch.py`

The teacher stage should accept a generalized page log:

```text
--page-log
```

or continue supporting:

```text
--render-log
```

where the page log may be either clean render records or augmented page records.

The common required fields should be:

```json
{
  "status": "ok",
  "score_id": "...",
  "page_id": "...",
  "page_number": 1,
  "image_path": "..."
}
```

Then the existing downstream stages can remain unchanged:

```text
build_training_manifest.py
make_splits.py
vocab.py
train_student.py
```

because teacher outputs should keep the same schema.

---

## 16. Implementation organization

Recommended files:

```text
distillation/docs/STUDENT_ARCHITECTURE.md
distillation/augment_pages.py
distillation/models/__init__.py
distillation/models/page_staff_ar_student.py
distillation/train_student.py
distillation/decode_student.py       # later
distillation/evaluate_student.py     # later
```

`train_student.py` should be refactored so that it does not contain the model architecture inline.

Instead:

```python
from distillation.models.page_staff_ar_student import PageStaffARStudent
```

`train_student.py` should own:

```text
argument parsing
dataset loading
collation
loss computation
optimizer
scheduler
logging
checkpointing
validation
```

`page_staff_ar_student.py` should own:

```text
visual encoder
page memory
layout encoder
staff slot decoder
autoregressive staff decoder
output heads
forward pass
generation helper later
```

---

## 17. Implementation phases

### Phase 0: infrastructure baseline

Status: passed.

```text
current temporary student
clean smoke split
CUDA training
finite losses
checkpoints/logs produced
```

### Phase 1: architecture specification

Create:

```text
distillation/docs/STUDENT_ARCHITECTURE.md
```

This document is Phase 1.

### Phase 2: sequential degradation stage

Implement:

```text
distillation/augment_pages.py
```

Modify:

```text
distillation/run_onnx_teacher_batch.py
```

Run small smoke:

```text
clean render log
  -> camera_light augment log
  -> HOMR teacher
  -> manifests/splits/vocab
  -> temporary trainer smoke
```

Purpose:

```text
prove degraded page ingestion and HOMR labeling work
```

### Phase 3: real hybrid model

Implement:

```text
distillation/models/page_staff_ar_student.py
```

Update:

```text
distillation/train_student.py
```

Purpose:

```text
replace temporary smoke architecture with real FullPageHOMRSurrogate
```

### Phase 4: overfit checks

Use small clean and degraded smoke datasets.

Success criteria:

```text
finite losses
loss decreases
branch accuracies improve
can overfit tiny subset
checkpoints written
```

### Phase 5: decoding/evaluation

Add:

```text
autoregressive generation
branch accuracy
symbol exact match
staff edit distance
page edit distance after newline reconstruction
HOMR-compatible reconstruction
```

### Phase 6: adversarial work

After surrogate agreement is adequate:

```text
spectral physical-noise evaluation
square attack on HOMR
gradient attacks on surrogate
transfer evaluation to HOMR
optional adversarial training
```

---

## 18. Open decisions

1. `S_max` default:
   - propose 16 for initial full-page rendered data.
   - should be measured from manifests.

2. `T_max` default:
   - use maximum staff length from encoded manifests with optional clipping.
   - should be logged.

3. Visual backbone:
   - start with small CNN/ResNet-like encoder.
   - later compare ConvNeXt/Swin if needed.

4. Staff-slot ordering:
   - v1 uses ordered slots directly.
   - future version may use matching if ordering becomes unstable.

5. Staff boxes:
   - optional, only if reliable page/staff metadata is available.

6. Newline reconstruction:
   - verify whether HOMR expects newline after each staff or between staffs only.
   - mirror HOMR behavior exactly.

7. Degradation intensity:
   - start with `camera_light_v1`.
   - use HOMR success rate to calibrate intensity.

---

## 19. Acceptance criteria for first real model

The first real architecture is acceptable when:

```text
1. model code lives in distillation/models/page_staff_ar_student.py
2. train_student.py imports the model instead of defining it inline
3. training runs on CUDA
4. clean smoke split trains with finite losses
5. degraded camera_light smoke split trains with finite losses
6. branch losses and branch accuracies are logged
7. latest.pt, training_log.jsonl, metrics.json are written
8. no <STAFF_BREAK> or learned newline token is introduced
9. targets remain HOMR-factorized ordered staff sequences
10. architecture and loss are documented here
```

---

## 20. References

- TrOMR: Transformer-Based Polyphonic Optical Music Recognition. arXiv:2308.09370.  
  https://arxiv.org/abs/2308.09370

- Sheet Music Transformer: End-To-End Optical Music Recognition Beyond Monophonic Transcription. arXiv:2402.07596.  
  https://arxiv.org/abs/2402.07596

- End-to-End Full-Page Optical Music Recognition for Pianoform Sheet Music / SMT++. arXiv:2405.12105.  
  https://arxiv.org/html/2405.12105v4

- DETR: End-to-End Object Detection with Transformers. arXiv:2005.12872.  
  https://arxiv.org/abs/2005.12872

- HOMR repository by liebharc.  
  https://github.com/liebharc/homr

- Existing Track A spectral noise utility in this project:
  `attacks/src/spectral_noise.py`
