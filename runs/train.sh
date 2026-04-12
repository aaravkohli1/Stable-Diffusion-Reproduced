#!/usr/bin/env bash

set -euo pipefail


MAX_HOURS=${MAX_HOURS:-75}      
HOURLY_RATE=${HOURLY_RATE:-1.00}     
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}  # auto-detect GPU count
SHARDS=${SHARDS:-69} 
MAX_LATENTS=${MAX_LATENTS:-200000} 
IMAGE_SIZE=${IMAGE_SIZE:-512}
LATENT_BS=${LATENT_BS:-32} 
CONFIG=${CONFIG:-configs/standard.yaml}
# -------------------------------------------------------------------------

BUDGET=$(awk "BEGIN{printf \"%.0f\", $MAX_HOURS * $HOURLY_RATE}")
echo "================================================================"
echo "  SD1.4 training — budget cap: ${MAX_HOURS}h ≈ \$${BUDGET}"
echo "================================================================"

START_TIME=$(date +%s)

elapsed_hours() {
  awk "BEGIN{printf \"%.1f\", ($(date +%s) - $START_TIME) / 3600}"
}

echo ""
echo "[0/3] Environment setup ($(elapsed_hours)h elapsed)"
echo "----------------------------------------------------------------"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"

# System deps
apt-get update -qq
apt-get install -y --no-install-recommends git tmux htop wget ca-certificates >/dev/null 2>&1

# Python deps
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

# Confirm CUDA
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible — check your instance"
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}, {props.total_mem / 1e9:.0f} GB")
print(f"  bf16={torch.cuda.is_bf16_supported()}, GPUs={torch.cuda.device_count()}")
PY

# Dirs
mkdir -p scripts/checkpoints scripts/latents scripts/unet_checkpoints

echo ""
echo "[1/3] Importing pretrained VAE ($(elapsed_hours)h elapsed)"
echo "----------------------------------------------------------------"

if [ -f scripts/checkpoints/vae_final.pt ]; then
  echo "  VAE checkpoint already exists, skipping."
else
  python scripts/import_vae.py
fi

echo ""
echo "[2/3] Caching latents from $SHARDS shards ($(elapsed_hours)h elapsed)"
echo "----------------------------------------------------------------"

LATENT_COUNT=$(find scripts/latents -name '*.pt' 2>/dev/null | wc -l)
if [ "$LATENT_COUNT" -ge "$MAX_LATENTS" ]; then
  echo "  $LATENT_COUNT latents already cached (>= $MAX_LATENTS), skipping."
else
  python scripts/compute_latents.py \
    --vae-ckpt scripts/checkpoints/vae_final.pt \
    --source hf --shards "$SHARDS" --max-samples "$MAX_LATENTS" \
    --image-size "$IMAGE_SIZE" --batch-size "$LATENT_BS" \
    --out-dir scripts/latents
fi

echo ""
echo "[3/3] Go!"
echo "----------------------------------------------------------------"

# Compute remaining seconds for the training timeout
ELAPSED_S=$(( $(date +%s) - START_TIME ))
BUDGET_S=$(( MAX_HOURS * 3600 ))
REMAINING_S=$(( BUDGET_S - ELAPSED_S ))

if [ "$REMAINING_S" -le 0 ]; then
  echo "ERROR: No time remaining after setup + caching. Increase MAX_HOURS."
  exit 1
fi

echo "  Training budget: $(awk "BEGIN{printf \"%.1f\", $REMAINING_S/3600}")h remaining"
echo "  Config: $CONFIG"
echo "  GPUs: $NUM_GPUS"
echo ""

# Use torchrun for multi-GPU, plain python for single-GPU.
if [ "$NUM_GPUS" -gt 1 ]; then
  TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS scripts/train_unet.py"
else
  TRAIN_CMD="python scripts/train_unet.py"
fi


timeout --signal=SIGINT --kill-after=30 "${REMAINING_S}s" \
  $TRAIN_CMD "$CONFIG" --resume latest \
  || EXIT_CODE=$?


if [ "${EXIT_CODE:-0}" -eq 124 ]; then
  echo ""
  echo "Budget reached (${MAX_HOURS}h). Training stopped gracefully."
elif [ "${EXIT_CODE:-0}" -ne 0 ]; then
  echo ""
  echo "Training exited with code ${EXIT_CODE}."
  exit "${EXIT_CODE}"
fi

# Done! :)
TOTAL_H=$(elapsed_hours)
COST=$(awk "BEGIN{printf \"%.2f\", $TOTAL_H * $HOURLY_RATE}")
echo ""
echo "================================================================"
echo "  Done. Wall time: ${TOTAL_H}h — estimated cost: \$${COST}"
echo "  Checkpoints: scripts/unet_checkpoints/"
echo "================================================================"
