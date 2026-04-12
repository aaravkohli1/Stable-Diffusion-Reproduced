#!/usr/bin/env bash
# ==========================================================================
#  LoRA fine-tuning on axonometric/isometric data.
#
#  Usage (on vast.ai or any GPU box):
#    tmux new -s finetune
#    bash runs/finetune.sh
#
#  This is much cheaper than full training — LoRA fine-tuning only trains
#  ~1-2M parameters and converges in a few thousand steps. A single A100/H100
#  for 1-2 hours is typically enough.
# ==========================================================================
set -euo pipefail

CONFIG=${CONFIG:-configs/finetune.yaml}
MAX_HOURS=${MAX_HOURS:-4}
HOURLY_RATE=${HOURLY_RATE:-1.50}

BUDGET=$(awk "BEGIN{printf \"%.0f\", $MAX_HOURS * $HOURLY_RATE}")
echo "================================================================"
echo "  LoRA fine-tuning — budget cap: ${MAX_HOURS}h ≈ \$${BUDGET}"
echo "================================================================"

START_TIME=$(date +%s)

elapsed_hours() {
  awk "BEGIN{printf \"%.1f\", ($(date +%s) - $START_TIME) / 3600}"
}

# ==================== Setup ==============================================
echo ""
echo "[0/1] Environment setup ($(elapsed_hours)h elapsed)"
echo "----------------------------------------------------------------"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"

apt-get update -qq 2>/dev/null
apt-get install -y --no-install-recommends git tmux >/dev/null 2>&1 || true

python -m pip install --upgrade pip -q 2>/dev/null
python -m pip install -r requirements.txt -q 2>/dev/null

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible"
print(f"  CUDA OK: {torch.cuda.get_device_name(0)}, bf16={torch.cuda.is_bf16_supported()}")
PY

mkdir -p scripts/lora_checkpoints

# ==================== Fine-tune ==========================================
echo ""
echo "[1/1] LoRA fine-tuning ($(elapsed_hours)h elapsed)"
echo "----------------------------------------------------------------"

ELAPSED_S=$(( $(date +%s) - START_TIME ))
BUDGET_S=$(( MAX_HOURS * 3600 ))
REMAINING_S=$(( BUDGET_S - ELAPSED_S ))

if [ "$REMAINING_S" -le 0 ]; then
  echo "ERROR: No time remaining. Increase MAX_HOURS."
  exit 1
fi

echo "  Budget: $(awk "BEGIN{printf \"%.1f\", $REMAINING_S/3600}")h remaining"
echo "  Config: $CONFIG"
echo ""

timeout --signal=SIGINT --kill-after=30 "${REMAINING_S}s" \
  python scripts/finetune_lora.py "$CONFIG" --resume latest \
  || EXIT_CODE=$?

if [ "${EXIT_CODE:-0}" -eq 124 ]; then
  echo ""
  echo "Budget reached (${MAX_HOURS}h). Fine-tuning stopped gracefully."
elif [ "${EXIT_CODE:-0}" -ne 0 ]; then
  echo ""
  echo "Fine-tuning exited with code ${EXIT_CODE}."
  exit "${EXIT_CODE}"
fi

# ==================== Done ===============================================
TOTAL_H=$(elapsed_hours)
COST=$(awk "BEGIN{printf \"%.2f\", $TOTAL_H * $HOURLY_RATE}")
echo ""
echo "================================================================"
echo "  Done. Wall time: ${TOTAL_H}h — estimated cost: \$${COST}"
echo "  LoRA checkpoints: scripts/lora_checkpoints/"
echo "================================================================"
