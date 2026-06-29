# Colab Setup and One-Shot Pipeline Prompt

## Before running claude on Colab

Run these in order in the VS Code terminal connected to Colab via SSH:

```bash
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cp "/content/drive/MyDrive/College/Projects/AI-AOL/datasets/PDMX.csv" /content/adversarial-homr/dataset/PDMX.csv
cp -r "/content/drive/MyDrive/College/Projects/AI-AOL/datasets/mxl" /content/adversarial-homr/dataset/mxl
```

```bash
cd /content/adversarial-homr
cd models/onnx
unzip encoder_pytorch_model_331-*.zip && mv encoder_pytorch_model_331-*.onnx tromr_encoder.onnx
unzip decoder_pytorch_model_331-*.zip && mv decoder_pytorch_model_331-*.onnx tromr_decoder.onnx
cd ../..
```

```bash
pip install poetry
poetry config virtualenvs.create false
poetry install
export PYTHONPATH=/content/adversarial-homr
```

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code
```

```bash
cd /content/adversarial-homr
claude
```

---

## One-shot prompt to paste into claude on Colab

Read CLAUDE.md and COLAB_SETUP.md before doing anything. All guardrails in CLAUDE.md apply without exception.

Do the following in order. Do not stop between steps. Do not ask for confirmation. If a step fails, fix it and continue.

Dataset is already extracted at:
- dataset/PDMX.csv
- dataset/mxl/

ONNX models are already extracted at:
- models/onnx/segnet.onnx
- models/onnx/tromr_encoder.onnx
- models/onnx/tromr_decoder.onnx

Step 1 - Verify ONNX models load:
Run: python attacks/src/homr_wrapper.py --describe-only
If it errors, fix the issue before continuing.

Step 2 - Generate 300 labeled training pages:
Run the distillation pipeline in order:
  python distillation/build_source_pool.py
  python distillation/select_batch.py --size 300 --out-dir distillation/batches/batch_main_300
  python distillation/render_batch.py --batch-dir distillation/batches/batch_main_300
  python distillation/run_onnx_teacher_batch.py --render-log distillation/batches/batch_main_300/logs/render_log.jsonl --out-dir distillation/batches/batch_main_300
  python distillation/build_training_manifest.py --batch-dir distillation/batches/batch_main_300
  python distillation/make_splits.py --batch-dir distillation/batches/batch_main_300
  python distillation/vocab.py --batch-dir distillation/batches/batch_main_300

Step 3 - Fix the student architecture:
In distillation/train_student.py, find the FullPageStudent encoder and replace AdaptiveAvgPool2d((1, 1)) with AdaptiveAvgPool2d((4, 4)). Update the page feature flatten to produce [B, embed_dim * 16] and add a nn.Linear(embed_dim * 16, embed_dim) projection named page_projection. Update the forward method to use page_projection after flattening. This preserves spatial layout for PGD gradients.

Step 4 - Implement PGD attack:
Create distillation/pgd_attack.py with a single function:
  pgd_attack(model, images, targets, sequence_mask, staff_exists, n_staffs, epsilon, steps, alpha, staff_loss_weight)
The function runs PGD using compute_loss imported from train_student. It initializes x_adv with uniform noise in [-epsilon, epsilon], then iterates: forward pass, compute_loss, backward, gradient sign step, project back into [x-epsilon, x+epsilon], clamp to [0,1]. Returns detached x_adv. Import compute_loss directly from distillation.train_student.

Step 5 - Add --adv-train to train_student.py:
Add these args to parse_args: --adv-train (store_true), --pgd-epsilon (float, default 0.02), --pgd-steps (int, default 10), --pgd-alpha (float, default 0.005).
At the top of run_epoch, if training and args.adv_train, call pgd_attack on each batch before the forward pass and replace batch images with x_adv.

Step 6 - Train clean surrogate:
python distillation/train_student.py \
  --train-manifest distillation/batches/batch_main_300/training_manifest.train.encoded.jsonl \
  --val-manifest distillation/batches/batch_main_300/training_manifest.val.encoded.jsonl \
  --out-dir distillation/runs/clean_surrogate \
  --epochs 30 \
  --device cuda

Step 7 - Train PGD surrogate:
python distillation/train_student.py \
  --train-manifest distillation/batches/batch_main_300/training_manifest.train.encoded.jsonl \
  --val-manifest distillation/batches/batch_main_300/training_manifest.val.encoded.jsonl \
  --out-dir distillation/runs/pgd_surrogate \
  --epochs 30 \
  --device cuda \
  --adv-train \
  --pgd-epsilon 0.02 \
  --pgd-steps 10 \
  --pgd-alpha 0.005

Step 8 - Implement evaluation script:
Create distillation/evaluate_surrogate.py that:
- Takes --clean-checkpoint, --pgd-checkpoint, --val-manifest, --epsilon-grid, --out-dir as args
- Loads both checkpoints and reconstructs both models
- For each epsilon in epsilon-grid, runs PGD on the val set images and measures branch accuracy and SER for both models
- Prints a comparison table to stdout
- Writes results/surrogate_comparison/comparison.json with all numbers

Step 9 - Run evaluation:
python distillation/evaluate_surrogate.py \
  --clean-checkpoint distillation/runs/clean_surrogate/best_clean.pt \
  --pgd-checkpoint distillation/runs/pgd_surrogate/best_clean.pt \
  --val-manifest distillation/batches/batch_main_300/training_manifest.val.encoded.jsonl \
  --epsilon-grid 0.0 0.01 0.02 0.05 0.10 \
  --out-dir results/surrogate_comparison

Step 10 - Save results to Drive:
cp results/surrogate_comparison/comparison.json "/content/drive/MyDrive/College/Projects/AI-AOL/results/comparison.json"
cp distillation/runs/clean_surrogate/best_clean.pt "/content/drive/MyDrive/College/Projects/AI-AOL/checkpoints/clean_surrogate.pt"
cp distillation/runs/pgd_surrogate/best_clean.pt "/content/drive/MyDrive/College/Projects/AI-AOL/checkpoints/pgd_surrogate.pt"

