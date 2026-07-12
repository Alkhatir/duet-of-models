#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/gradio_transformer.sh [checkpoint_dir] [model_cfg] [tokenizer_cfg] [output_dir] [extra_args...]
CHECKPOINT="${1:-experiments/transformer-tok14-batch16-sched-decay8000/gpt2}"
MODEL_CFG="${2:-configs/model/transformer/tok_14_gpt2_match.yaml}"
TOKENIZER_CFG="${3:-configs/data/14.yaml}"
OUTPUT_DIR="${4:-experiments/gradio_transformer_generations}"
shift $(( $# < 4 ? $# : 4 ))

GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}" \
PYTHONUNBUFFERED=1 \
uv run --project envs/gpt2 python -m app.gradio_transformer \
  --checkpoint "$CHECKPOINT" \
  --model_cfg "$MODEL_CFG" \
  --tok_cfg "$TOKENIZER_CFG" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
