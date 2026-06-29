set -e

REPO=/content/adversarial-homr
cd "$REPO"
export PYTHONPATH="$REPO"
export QT_QPA_PLATFORM=offscreen

N_SCORES="${1:-900}"
EPOCHS="${2:-40}"
BATCH_ID="batch_large_${N_SCORES}"
B="distillation/batches/$BATCH_ID"
MXL_SRC="/content/drive/MyDrive/College/Projects/AI-AOL/datasets/mxl/0"
LOCAL_MXL="dataset/mxl_big"

echo "=== [1] Gather $N_SCORES mxl scores from Drive shard 0 ==="
mkdir -p "$LOCAL_MXL"
find "$MXL_SRC" -name "*.mxl" 2>/dev/null | head -"$N_SCORES" | xargs -I{} cp {} "$LOCAL_MXL/" 2>/dev/null
echo "gathered: $(ls "$LOCAL_MXL" | wc -l)"

echo "=== [2] build_source_pool ==="
python distillation/build_source_pool.py --mxl-root "$LOCAL_MXL" \
  --output-jsonl distillation/data/source_pool_big.jsonl --overwrite

echo "=== [3] select_batch ==="
python distillation/select_batch.py --source-pool distillation/data/source_pool_big.jsonl \
  --batch-id "$BATCH_ID" --batch-size "$N_SCORES" --overwrite

echo "=== [4] render (MuseScore 3 via xvfb) + flatten ==="
xvfb-run -a python distillation/render_batch.py \
  --batch-manifest "$B/batch_manifest.jsonl" --render-log "$B/logs/render_log.jsonl"
python distillation/flatten_renders.py --images-dir "$B/rendered_images"

echo "=== [5] ONNX teacher on GPU ==="
python distillation/run_onnx_teacher_batch.py \
  --render-log "$B/logs/render_log.jsonl" \
  --teacher-dir "$B/teacher_outputs" --teacher-log "$B/logs/teacher_log.jsonl" \
  --segnet-onnx models/onnx/segnet.onnx --model-dir models/onnx --continue-on-error

echo "=== [6] manifest + splits + vocab ==="
python distillation/build_training_manifest.py --teacher-dir "$B/teacher_outputs" \
  --out "$B/training_manifest.jsonl" --allow-empty
python distillation/make_splits.py --manifest "$B/training_manifest.jsonl" \
  --out-dir "$B/splits" --allow-missing-score-id
python distillation/vocab.py \
  --train-manifest "$B/splits/training_manifest.train.jsonl" \
  --validate-manifest "$B/splits/training_manifest.val.jsonl" \
  --out "$B/vocab.json" --encoded-out-dir "$B"

echo "=== [7] Train ORIGINAL surrogate (no defense) ==="
python distillation/train_student.py \
  --train-manifest "$B/training_manifest.train.encoded.jsonl" \
  --val-manifest "$B/training_manifest.val.encoded.jsonl" \
  --out-dir distillation/runs/clean_surrogate_large \
  --epochs "$EPOCHS" --batch-size 4 --device cuda --quiet

echo "=== [8] Train PGD-DEFENDED surrogate ==="
python distillation/train_student.py \
  --train-manifest "$B/training_manifest.train.encoded.jsonl" \
  --val-manifest "$B/training_manifest.val.encoded.jsonl" \
  --out-dir distillation/runs/pgd_surrogate_large \
  --epochs "$EPOCHS" --batch-size 4 --device cuda --quiet \
  --adv-train --pgd-epsilon 0.02 --pgd-steps 10 --pgd-alpha 0.005

echo "=== [9] PGD epsilon-grid comparison ==="
python distillation/evaluate_surrogate.py \
  --clean-checkpoint distillation/runs/clean_surrogate_large/best_clean.pt \
  --pgd-checkpoint distillation/runs/pgd_surrogate_large/best_clean.pt \
  --val-manifest "$B/training_manifest.val.encoded.jsonl" \
  --epsilon-grid 0.0 0.01 0.02 0.05 0.10 \
  --out-dir results/surrogate_comparison_large --device cuda

echo "=== [10] AutoAttack before vs after defense ==="
python distillation/autoattack_eval.py \
  --clean-checkpoint distillation/runs/clean_surrogate_large/best_clean.pt \
  --pgd-checkpoint distillation/runs/pgd_surrogate_large/best_clean.pt \
  --val-manifest "$B/training_manifest.val.encoded.jsonl" \
  --epsilon-grid 0.0 0.01 0.02 0.05 0.10 \
  --out-dir results/surrogate_comparison_large --device cuda --version standard

echo "=== [11] Back up results to Drive ==="
SESS="/content/drive/MyDrive/College/Projects/AI-AOL/adversarial_homr_session"
mkdir -p "$SESS/runs" "$SESS/results_large"
cp -rf distillation/runs/clean_surrogate_large distillation/runs/pgd_surrogate_large "$SESS/runs/" 2>/dev/null || true
cp -f results/surrogate_comparison_large/*.json "$SESS/results_large/" 2>/dev/null || true
echo "DONE. Comparisons in results/surrogate_comparison_large/ and on Drive."
