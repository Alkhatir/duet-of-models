#!/usr/bin/env bash
set -euo pipefail

# Train the medium xLSTM with the best tokenizer from the base tokenizer sweep.
#
# The train config runs for 15 epochs with three cosine LR cycles. The two
# cycle boundaries are based on tokenizer 14 at batch size 8:
#   4858 steps/epoch * 5 epochs  = 24290
#   4858 steps/epoch * 10 epochs = 48580
# Each new cosine cycle starts at the maximum LR again. The maximum LR is
# 1e-4, matching the best run from experiments/xlstm-base-lr-sweep-tok11.
#
# Optional controls:
#   OUTPUT_DIR=experiments/custom-run-name
#   RUN_GAP_SECONDS=30

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_LIST="${TRAIN_LIST:-data_reports/splits/train.txt}"
VAL_LIST="${VAL_LIST:-data_reports/splits/val.txt}"
MODEL_CFG="${MODEL_CFG:-configs/model/xlstm/medium.yaml}"
TOKENIZER_CFG="${TOKENIZER_CFG:-configs/data/14.yaml}"
TRAIN_CFG="${TRAIN_CFG:-configs/train/medium_tok14_15epoch_restarts/medium_15_epoch_cosine_restarts_batch_8.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/xlstm-medium-batch8-tok14-15epoch-cosine-restarts}"

echo "Starting medium xLSTM run"
echo "  model:     ${MODEL_CFG}"
echo "  tokenizer: ${TOKENIZER_CFG}"
echo "  train cfg: ${TRAIN_CFG}"
echo "  output:    ${OUTPUT_DIR}"

scripts/train_xlstm.sh \
  "$TRAIN_LIST" \
  "$VAL_LIST" \
  "$MODEL_CFG" \
  "$TOKENIZER_CFG" \
  "$OUTPUT_DIR" \
  "$TRAIN_CFG"

if [[ -n "${CONTAINER_ID:-}" ]] && command -v vastai >/dev/null 2>&1; then
  if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
    vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
  else
    vastai stop instance "$CONTAINER_ID"
  fi
else
  echo "Skipping Vast.ai shutdown: CONTAINER_ID is unset or vastai CLI is unavailable."
fi
