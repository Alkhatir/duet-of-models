#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/eval_generation.sh <model_type> <checkpoint> <model_cfg> <tokenizer_cfg> <split_list> <out_dir> [eval_cfg]
MODEL_TYPE="${1:?model_type required}"
CHECKPOINT="${2:?checkpoint required}"
MODEL_CFG="${3:?model_cfg required}"
TOKENIZER_CFG="${4:-configs/data/tokenizer_config.yaml}"
DATA_LIST="${5:?split_list required}"
OUT_DIR="${6:?out_dir required}"
EVAL_CFG="${7:-configs/eval/generation_shared.yaml}"

case "$MODEL_TYPE" in
  transformer|gpt2)
    PROJECT="envs/gpt2"
    MODULE="src.models.Transformer.generate_samples"
    ;;
  xlstm)
    PROJECT="envs/xlstm"
    MODULE="src.models.xlstm.generate_samples"
    ;;
  *)
    echo "ERROR: model_type must be one of: transformer, gpt2, xlstm" >&2
    exit 2
    ;;
esac

uv run --project "$PROJECT" python -m "$MODULE" \
  --checkpoint "$CHECKPOINT" \
  --model_cfg "$MODEL_CFG" \
  --tok_cfg "$TOKENIZER_CFG" \
  --data_list "$DATA_LIST" \
  --eval_cfg "$EVAL_CFG" \
  --out_dir "$OUT_DIR"

uv run --project envs/data python -m src.evaluation.score_generated_midis \
  --samples_dir "$OUT_DIR" \
  --eval_cfg "$EVAL_CFG" \
  --model_type "$MODEL_TYPE" \
  --checkpoint "$CHECKPOINT"
