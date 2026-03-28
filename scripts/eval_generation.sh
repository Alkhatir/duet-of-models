#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/eval_generation.sh <model_type> <checkpoint> <model_cfg> <tokenizer_cfg> <data_list> <out_dir> [eval_cfg]
MODEL_TYPE="${1:?model_type required}"
CHECKPOINT="${2:?checkpoint required}"
MODEL_CFG="${3:?model_cfg required}"
TOKENIZER_CFG="${4:-configs/data/tokenizer_config.yaml}"
DATA_LIST="${5:?data_list required}"
OUT_DIR="${6:?out_dir required}"
EVAL_CFG="${7:-configs/eval/generation_shared.yaml}"

uv run --project envs/eval python -m src.evaluation.generate_and_score \
  --model_type "$MODEL_TYPE" \
  --checkpoint "$CHECKPOINT" \
  --model_cfg "$MODEL_CFG" \
  --tok_cfg "$TOKENIZER_CFG" \
  --data_list "$DATA_LIST" \
  --eval_cfg "$EVAL_CFG" \
  --out_dir "$OUT_DIR"
