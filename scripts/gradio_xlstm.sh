#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/gradio_xlstm.sh [checkpoint] [model_cfg] [tokenizer_cfg] [output_dir] [extra_args...]
CHECKPOINT="${1:-experiments/xlstm-small-batch8-tok11/best/checkpoint.pt}"
MODEL_CFG="${2:-configs/model/xlstm/small.yaml}"
TOKENIZER_CFG="${3:-configs/data/11.yaml}"
OUTPUT_DIR="${4:-experiments/gradio_xlstm_generations}"
shift $(( $# < 4 ? $# : 4 ))

GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}" \
PYTHONUNBUFFERED=1 \
uv run --project envs/xlstm python -m app.gradio_xlstm \
  --checkpoint "$CHECKPOINT" \
  --model_cfg "$MODEL_CFG" \
  --tok_cfg "$TOKENIZER_CFG" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
