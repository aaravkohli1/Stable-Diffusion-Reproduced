#!/usr/bin/env bash
# ==========================================================================
#  One-shot SD1.4 training on a vast.ai H100 80GB.
#
#  Combines environment setup, latent caching, and UNet training into a
#  single script with a wall-clock budget guard so you don't overspend.
#
#  Usage (on the remote box):
#    git clone https://github.com/aaravkohli1/Stable-Diffusion-Reproduced.git
#    cd Stable-Diffusion-Reproduced
#    tmux new -s train
#    bash runs/train.sh            # walks away
#    # Ctrl-b d to detach, `tmux a -t train` to reattach
#
#  Budget math (H100 80GB @ ~$1.50/hr on vast.ai):
#    $80 ÷ $1.50/hr ≈ 53 hours total wall-clock.
#    ~1 hr  setup + latent caching
#    ~50 hr UNet training
#    At ~0.5s/optimizer-step (bf16 + torch.compile), 50 hr ≈ 160k steps.
#    Adjust MAX_HOURS if your $/hr rate differs.
# ==========================================================================
set -euo pipefail

# --- Budget & knobs ------------------------------------------------------
MAX_HOURS=${MAX_HOURS:-50}            # wall-clock cap for the training phase
HOURLY_RATE=${HOURLY_RATE:-1.50}      # $/hr — H100 on vast.ai
SHARDS=${SHARDS:-69}                  # 69 shards available (~1M total)
MAX_LATENTS=${MAX_LATENTS:-200000}    # how many samples to cache
IMAGE_SIZE=${IMAGE_SIZE:-512}
LATENT_BS=${LATENT_BS:-32}            # VAE+CLIP forward batch size
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

# ==================== PHASE 0: Environment setup =========================
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
props = torch.cuda.get_device_properties(0)
print(f"  CUDA OK: {props.name}, {props.total_mem / 1e9:.0f} GB, bf16={torch.cuda.is_bf16_supported()}")
PY

# Dirs
mkdir -p scripts/checkpoints scripts/latents scripts/unet_checkpoints

# ==================== PHASE 1: Import pretrained VAE =====================
echo ""
echo "[1/3] Importing pretrained VAE ($(elapsed_hours)h elapsed)"
echo "----------------------------------------------------------------"

if [ -f scripts/checkpoints/vae_final.pt ]; then
  echo "  VAE checkpoint already exists, skipping."
else
  python scripts/import_vae.py
fi

# ==================== PHASE 2: Cache latents =============================
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

# ==================== PHASE 3: Train UNet ================================
echo ""
echo "[3/3] Training UNet — max ${MAX_HOURS}h ($(elapsed_hours)h elapsed so far)"
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
echo ""

# timeout sends SIGTERM, giving the training loop a chance to save a final
# checkpoint. The 30s --kill-after is a hard stop if it doesn't exit cleanly.
timeout --signal=SIGINT --kill-after=30 "${REMAINING_S}s" \
  python scripts/train_unet.py "$CONFIG" --resume latest \
  || EXIT_CODE=$?

# timeout returns 124 on expiry — that's expected, not an error
if [ "${EXIT_CODE:-0}" -eq 124 ]; then
  echo ""
  echo "Budget reached (${MAX_HOURS}h). Training stopped gracefully."
elif [ "${EXIT_CODE:-0}" -ne 0 ]; then
  echo ""
  echo "Training exited with code ${EXIT_CODE}."
  exit "${EXIT_CODE}"
fi

# ==================== Done ===============================================
TOTAL_H=$(elapsed_hours)
COST=$(awk "BEGIN{printf \"%.2f\", $TOTAL_H * $HOURLY_RATE}")
echo ""
echo "================================================================"
echo "  Done. Wall time: ${TOTAL_H}h — estimated cost: \$${COST}"
echo "  Checkpoints: scripts/unet_checkpoints/"
echo "================================================================"
