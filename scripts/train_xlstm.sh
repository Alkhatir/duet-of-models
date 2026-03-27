#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/train_xlstm.sh <data_list> <model_cfg> <tokenizer_cfg> [train_cfg]
DATA_LIST="${1:-data_reports/sampled_subset.txt}"
MODEL_CFG="${2:-configs/model/xlstm/basic.yaml}"
TOKENIZER_CFG="${3:-configs/data/tokenizer_config.yaml}"
TRAIN_CFG="${4:-configs/train/base.yaml}"

uv run --project envs/xlstm python -m src.models.xlstm.train \
  --cfg "$MODEL_CFG" \
  --train_cfg "$TRAIN_CFG" \
  --tok_cfg "$TOKENIZER_CFG" \
  --data_list "$DATA_LIST"
