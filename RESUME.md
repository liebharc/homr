# Resume adversarial-homr in a new Colab runtime

Continue the PGD-defense vs AutoAttack surrogate study without redoing setup.

## Methodology

- Original surrogate: differentiable full-page student trained on HOMR teacher outputs (no defense).
- Defense: PGD adversarial training of the surrogate (distillation/pgd_attack.py + train_student.py --adv-train).
- Attack: AutoAttack (APGD-CE / APGD-T / FAB / Square), run BEFORE defense (on the clean surrogate)
  and AFTER defense (on the PGD-trained surrogate) for a like-for-like robustness comparison.
- Reports: PGD epsilon-grid comparison (evaluate_surrogate.py) and AutoAttack before/after
  defense (autoattack_eval.py). AutoAttack targets the surrogate staff-count classifier head,
  operates in the [0,1] image domain, and also reports downstream token accuracy and SER under attack.

## Quick start in a fresh runtime

1. Mount Drive:

       from google.colab import drive
       drive.mount('/content/drive')

2. Ensure the repo is at /content/adversarial-homr (git clone if needed).

3. Set up the environment (GPU driver, musescore3, xvfb, onnxruntime-gpu, autoattack, models, code):

       bash "/content/drive/MyDrive/College/Projects/AI-AOL/adversarial_homr_session/setup_session.sh"

4. Run the full large-scale pipeline (gather -> render -> GPU teacher -> train clean + PGD -> compare):

       export PYTHONPATH=/content/adversarial-homr
       bash "/content/drive/MyDrive/College/Projects/AI-AOL/adversarial_homr_session/run_large_pipeline.sh" 900 40

   Arguments: <N_SCORES> <EPOCHS>. Outputs go to results/surrogate_comparison_large/ and are
   copied to Drive under adversarial_homr_session/results_large/ and runs/.

## GPU note (L4)

The L4 kernel module is 580.82.07 but Colab ships only 570 userspace libs, causing CUDA error 803.
gpu_config_backup/restore_gpu.sh installs the matching 580.82.07 libcuda/libnvidia-ml into
/opt/nvidia-userspace and runs ldconfig. nvidia-smi may warn but torch.cuda works.

## onnxruntime on GPU

setup_session.sh installs onnxruntime-gpu==1.20.1 (CUDA 12) and registers cuDNN/cublas/cufft/curand
(from the pip nvidia-* packages) plus torch's bundled libs via /etc/ld.so.conf.d/001-ort-cuda.conf,
so the ONNX teacher runs on CUDAExecutionProvider. If CUDA EP fails, add --cpu to run_onnx_teacher_batch.

## Rendering note (MuseScore 3)

mscore3 exports transparent PNGs; SegNet needs black-on-white. The pipeline runs
distillation/flatten_renders.py after render_batch to composite onto white. Required only when
rendering with MuseScore 3 instead of MuseScore 4.

## Dataset note

The Drive mxl tree is two-level sharded (mxl/<shard>/<subshard>/*.mxl) and the FUSE mount is lazy:
a recursive find over an un-cached shard can return empty. Shard 0 alone holds ~14,872 scores and
is reliable; run_large_pipeline.sh gathers N scores from shard 0. Using the full ~250k-score dataset
is not feasible in one session (render + teacher would take days).

## Manual pipeline (real arg names)

    python distillation/build_source_pool.py --mxl-root dataset/mxl_big --output-jsonl distillation/data/source_pool_big.jsonl --overwrite
    python distillation/select_batch.py --source-pool distillation/data/source_pool_big.jsonl --batch-id <BATCH_ID> --batch-size <N>
    xvfb-run -a python distillation/render_batch.py --batch-manifest distillation/batches/<BATCH_ID>/batch_manifest.jsonl --render-log distillation/batches/<BATCH_ID>/logs/render_log.jsonl
    python distillation/flatten_renders.py --images-dir distillation/batches/<BATCH_ID>/rendered_images
    python distillation/run_onnx_teacher_batch.py --render-log distillation/batches/<BATCH_ID>/logs/render_log.jsonl --teacher-dir distillation/batches/<BATCH_ID>/teacher_outputs --teacher-log distillation/batches/<BATCH_ID>/logs/teacher_log.jsonl --segnet-onnx models/onnx/segnet.onnx --model-dir models/onnx --continue-on-error
    python distillation/build_training_manifest.py --teacher-dir distillation/batches/<BATCH_ID>/teacher_outputs --out distillation/batches/<BATCH_ID>/training_manifest.jsonl --allow-empty
    python distillation/make_splits.py --manifest distillation/batches/<BATCH_ID>/training_manifest.jsonl --out-dir distillation/batches/<BATCH_ID>/splits --allow-missing-score-id
    python distillation/vocab.py --train-manifest distillation/batches/<BATCH_ID>/splits/training_manifest.train.jsonl --validate-manifest distillation/batches/<BATCH_ID>/splits/training_manifest.val.jsonl --out distillation/batches/<BATCH_ID>/vocab.json --encoded-out-dir distillation/batches/<BATCH_ID>

## Train and compare (before/after defense)

    python distillation/train_student.py --train-manifest <B>/training_manifest.train.encoded.jsonl --val-manifest <B>/training_manifest.val.encoded.jsonl --out-dir distillation/runs/clean_surrogate_large --epochs 40 --batch-size 4 --device cuda
    python distillation/train_student.py --train-manifest <B>/training_manifest.train.encoded.jsonl --val-manifest <B>/training_manifest.val.encoded.jsonl --out-dir distillation/runs/pgd_surrogate_large --epochs 40 --batch-size 4 --device cuda --adv-train --pgd-epsilon 0.02 --pgd-steps 10 --pgd-alpha 0.005
    python distillation/evaluate_surrogate.py --clean-checkpoint distillation/runs/clean_surrogate_large/best_clean.pt --pgd-checkpoint distillation/runs/pgd_surrogate_large/best_clean.pt --val-manifest <B>/training_manifest.val.encoded.jsonl --epsilon-grid 0.0 0.01 0.02 0.05 0.10 --out-dir results/surrogate_comparison_large --device cuda
    python distillation/autoattack_eval.py --clean-checkpoint distillation/runs/clean_surrogate_large/best_clean.pt --pgd-checkpoint distillation/runs/pgd_surrogate_large/best_clean.pt --val-manifest <B>/training_manifest.val.encoded.jsonl --epsilon-grid 0.0 0.01 0.02 0.05 0.10 --out-dir results/surrogate_comparison_large --device cuda --version standard

## Folder contents (on Drive under College/Projects/AI-AOL)

- gpu_config_backup/   580.82.07 driver libs + restore_gpu.sh + ld.so.conf snippet
- adversarial_homr_session/setup_session.sh    full environment restore (GPU + ORT-GPU + deps + models + code)
- adversarial_homr_session/run_large_pipeline.sh   one-shot data->train->compare
- adversarial_homr_session/CLAUDE.md            updated project guide
- adversarial_homr_session/code/                train_student.py, pgd_attack.py, evaluate_surrogate.py, autoattack_eval.py, flatten_renders.py
- adversarial_homr_session/models/             segnet.onnx, tromr_encoder.onnx, tromr_decoder.onnx, segnet_308 fp32/fp16
- adversarial_homr_session/batch_e2e_small/    processed demo batch (already runnable)
- adversarial_homr_session/runs/               demo + large checkpoints
- adversarial_homr_session/results/            comparison.json, autoattack_comparison.json (demo)
- adversarial_homr_session/results_large/      large-run comparisons (after running run_large_pipeline.sh)
