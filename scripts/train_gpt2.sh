#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/train_gpt2.sh <train_list> <val_list> <test_list> <model_cfg> <tokenizer_cfg>
TRAIN_LIST="${1:-data_reports/splits/train.txt}"
VAL_LIST="${2:-data_reports/splits/val.txt}"
TEST_LIST="${3:-data_reports/splits/test.txt}"
MODEL_CFG="${4:-configs/model/gpt2/nano.yaml}"
TOKENIZER_CFG="${5:-configs/data/tokenizer_config.yaml}"

uv run --project envs/gpt2 python -m src.models.Transformer.train \
  --cfg "$MODEL_CFG" \
  --tok_cfg "$TOKENIZER_CFG" \
  --train_list "$TRAIN_LIST" \
  --val_list "$VAL_LIST" \
  --test_list "$TEST_LIST"
