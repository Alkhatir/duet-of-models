#!/bin/bash
set -euo pipefail

#set -a
#source .env
#set +a

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
RUN_GAP_SECONDS="${RUN_GAP_SECONDS:-30}"

TRAIN_LIST="data_reports/splits/train.txt"
VAL_LIST="data_reports/splits/val.txt"
MODEL_CFG="configs/model/xlstm/base.yaml"
TOKENIZER_CFG="configs/data/11.yaml"
OUTPUT_ROOT="experiments/xlstm-base-lr-sweep-tok11"

SWEEP_CONFIGS=(
  "configs/train/lr_sweep/batch_8_constant_1e-4.yaml"
  "configs/train/lr_sweep/batch_8_constant_3e-4.yaml"
  "configs/train/lr_sweep/batch_8_sched_peak_1e-4.yaml"
  "configs/train/lr_sweep/batch_8_sched_peak_2e-4.yaml"
  "configs/train/lr_sweep/batch_8_sched_peak_3e-4.yaml"
  "configs/train/lr_sweep/batch_8_sched_peak_5e-4.yaml"
  "configs/train/lr_sweep/batch_8_sched_warmup_100.yaml"
  "configs/train/lr_sweep/batch_8_sched_warmup_500.yaml"
  "configs/train/lr_sweep/batch_8_sched_decay_8000.yaml"
  "configs/train/lr_sweep/batch_8_sched_decay_12000.yaml"
)

between_runs() {
  echo "Waiting ${RUN_GAP_SECONDS}s before the next experiment..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
  sleep "$RUN_GAP_SECONDS"
}

mkdir -p "$OUTPUT_ROOT"

for index in "${!SWEEP_CONFIGS[@]}"; do
  TRAIN_CFG="${SWEEP_CONFIGS[$index]}"
  RUN_NAME="$(basename "$TRAIN_CFG" .yaml)"
  OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

  echo "Starting xLSTM base LR sweep run: ${RUN_NAME}"
  scripts/train_xlstm.sh \
    "$TRAIN_LIST" \
    "$VAL_LIST" \
    "$MODEL_CFG" \
    "$TOKENIZER_CFG" \
    "$OUTPUT_DIR" \
    "$TRAIN_CFG"

  if [[ "$index" -lt "$((${#SWEEP_CONFIGS[@]} - 1))" ]]; then
    between_runs
  fi
done

if [[ -n "${CONTAINER_ID:-}" ]] && command -v vastai >/dev/null 2>&1; then
  if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
    vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
  else
    vastai stop instance "$CONTAINER_ID"
  fi
else
  echo "Skipping Vast.ai shutdown: CONTAINER_ID is unset or vastai CLI is unavailable."
fi
