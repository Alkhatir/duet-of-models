#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/train_xlstm.sh <train_list> <val_list> <test_list> <model_cfg> <tokenizer_cfg> <output_dir> [train_cfg]
TRAIN_LIST="${1:-data_reports/splits/train.txt}"
VAL_LIST="${2:-data_reports/splits/val.txt}"
TEST_LIST="${3:-data_reports/splits/test.txt}"
MODEL_CFG="${4:-configs/model/xlstm/basic.yaml}"
TOKENIZER_CFG="${5:-configs/data/tokenizer_config.yaml}"
OUTPUT_DIR="${6:?output_dir required}"
TRAIN_CFG="${7:-configs/train/base.yaml}"

uv run --project envs/xlstm python -m src.models.xlstm.train \
  --cfg "$MODEL_CFG" \
  --train_cfg "$TRAIN_CFG" \
  --tok_cfg "$TOKENIZER_CFG" \
  --train_list "$TRAIN_LIST" \
  --val_list "$VAL_LIST" \
  --test_list "$TEST_LIST" \
  --output_dir "$OUTPUT_DIR"
