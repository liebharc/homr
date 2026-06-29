set -e

REPO=/content/adversarial-homr
BK="/content/drive/MyDrive/College/Projects/AI-AOL"
SESS="$BK/adversarial_homr_session"

echo "[1/8] Restoring L4 GPU userspace driver libs (580.82.07)"
bash "$BK/gpu_config_backup/restore_gpu.sh"

echo "[2/8] Installing system render dependencies (musescore3, xvfb)"
apt-get update -qq
apt-get install -y -qq musescore3 xvfb

echo "[3/8] Installing onnxruntime-gpu (CUDA 12) and autoattack"
pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
pip install -q "onnxruntime-gpu==1.20.1"
pip install -q git+https://github.com/fra31/auto-attack.git
cd "$REPO"
pip install -q -e . 2>/dev/null || poetry install 2>/dev/null || true

echo "[4/8] Registering CUDA libs for onnxruntime (cuDNN/cublas/cufft/curand + torch)"
PYSITE=$(python -c "import site;print(site.getsitepackages()[0])")
NV="$PYSITE/nvidia"
TORCHLIB=$(python -c "import os,torch;print(os.path.dirname(torch.__file__)+'/lib')")
{
  echo "/opt/nvidia-userspace"
  echo "$NV/cudnn/lib"
  echo "$NV/cublas/lib"
  echo "$NV/cuda_runtime/lib"
  echo "$NV/cufft/lib"
  echo "$NV/curand/lib"
  echo "$TORCHLIB"
} > /etc/ld.so.conf.d/001-ort-cuda.conf
ldconfig 2>/dev/null || true

echo "[5/8] Restoring ONNX models"
mkdir -p "$REPO/models/onnx"
cp -f "$SESS/models/segnet.onnx" "$REPO/models/onnx/segnet.onnx"
cp -f "$SESS/models/tromr_encoder.onnx" "$REPO/models/onnx/tromr_encoder.onnx"
cp -f "$SESS/models/tromr_decoder.onnx" "$REPO/models/onnx/tromr_decoder.onnx"
cp -f "$SESS/models/segnet_308"*.onnx "$REPO/homr/segmentation/" 2>/dev/null || true

echo "[6/8] Restoring distillation code (PGD defense + AutoAttack before/after)"
cp -f "$SESS/code/"*.py "$REPO/distillation/"
cp -f "$SESS/CLAUDE.md" "$REPO/CLAUDE.md" 2>/dev/null || true

echo "[7/8] Staging dataset pointers"
cp -f "$BK/datasets/PDMX.csv" "$REPO/dataset/PDMX.csv"
ln -sfn "$BK/datasets/mxl" "$REPO/dataset/mxl"

echo "[8/8] Restoring demo batch, checkpoints, results"
mkdir -p "$REPO/distillation/batches" "$REPO/distillation/runs" "$REPO/results"
cp -rf "$SESS/batch_e2e_small" "$REPO/distillation/batches/" 2>/dev/null || true
cp -rf "$SESS/runs/." "$REPO/distillation/runs/" 2>/dev/null || true
cp -rf "$SESS/results/." "$REPO/results/" 2>/dev/null || true

export PYTHONPATH="$REPO"
python -c "import torch, onnxruntime as ort; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); print('ort:', ort.__version__, ort.get_available_providers())"
echo "Session ready. Run: export PYTHONPATH=$REPO"
echo "Then: bash \"$SESS/run_large_pipeline.sh\" <N_SCORES> <EPOCHS>"
