#!/bin/bash
set -euo pipefail

set -a
source .env
set +a

TRAIN_LIST="data_reports/splits/train.txt"
VAL_LIST="data_reports/splits/val.txt"
TOKENIZER_CFG="data_reports/tokenizer_info/tokenizers/11.yaml"

scripts/train_xlstm.sh \
  "$TRAIN_LIST" \
  "$VAL_LIST" \
  configs/model/xlstm/small.yaml \
  "$TOKENIZER_CFG" \
  experiments/xlstm-small-batch8-tok11 \
  configs/train/batch_8.yaml

scripts/train_xlstm.sh \
  "$TRAIN_LIST" \
  "$VAL_LIST" \
  configs/model/xlstm/small.yaml \
  "$TOKENIZER_CFG" \
  experiments/xlstm-small-batch16-tok11 \
  configs/train/batch_16.yaml

scripts/train_xlstm.sh \
  "$TRAIN_LIST" \
  "$VAL_LIST" \
  configs/model/xlstm/base.yaml \
  "$TOKENIZER_CFG" \
  experiments/xlstm-base-batch4-tok11 \
  configs/train/batch_4.yaml

scripts/train_xlstm.sh \
  "$TRAIN_LIST" \
  "$VAL_LIST" \
  configs/model/xlstm/base.yaml \
  "$TOKENIZER_CFG" \
  experiments/xlstm-base-batch8-tok11 \
  configs/train/batch_8.yaml

if [[ -n "${CONTAINER_ID:-}" ]] && command -v vastai >/dev/null 2>&1; then
  if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
    vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
  else
    vastai stop instance "$CONTAINER_ID"
  fi
else
  echo "Skipping Vast.ai shutdown: CONTAINER_ID is unset or vastai CLI is unavailable."
fi
