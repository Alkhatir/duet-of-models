#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/train_gpt2.sh <data_dir> <model_cfg> <tokenizer_cfg>
DATA_DIR="${1:-data/lmd_matched}"
MODEL_CFG="${2:-configs/model/gpt2/nano.yaml}"
TOKENIZER_CFG="${3:-configs/data/tokenizer_config.yaml}"

uv run --project envs/gpt2 python train_transformer.py \
  --cfg "$MODEL_CFG" \
  --tok_cfg "$TOKENIZER_CFG" \
  --data_dir "$DATA_DIR"
