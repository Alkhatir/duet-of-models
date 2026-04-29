#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/report_split_sizes_matrix.sh <out_dir> <train_cfg> <train_list> <val_list> <test_list> <tok_cfg...>
#
# Example:
#   scripts/report_split_sizes_matrix.sh \
#     data_reports/split_size_reports \
#     configs/train/base.yaml \
#     data_reports/splits/train.txt \
#     data_reports/splits/val.txt \
#     data_reports/splits/test.txt \
#     data_reports/autotune_miditok_remi_test/*.yaml

OUT_DIR="${1:?output directory required}"
TRAIN_CFG="${2:-configs/train/base.yaml}"
TRAIN_LIST="${3:-data_reports/splits/train.txt}"
VAL_LIST="${4:-data_reports/splits/val.txt}"
TEST_LIST="${5:-data_reports/splits/test.txt}"
shift $(($# < 5 ? $# : 5))

if [[ "$#" -eq 0 ]]; then
  echo "error: provide at least one tokenizer config path" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
MANIFEST="$OUT_DIR/manifest.jsonl"
: >"$MANIFEST"

for TOK_CFG in "$@"; do
  if [[ ! -f "$TOK_CFG" ]]; then
    echo "warning: skipping missing tokenizer config: $TOK_CFG" >&2
    continue
  fi

  BASENAME="$(basename "$TOK_CFG")"
  STEM="${BASENAME%.*}"
  JSON_OUT="$OUT_DIR/${STEM}.json"

  echo "Running split-size report for $TOK_CFG -> $JSON_OUT" >&2

  uv run --project envs/xlstm python -m src.data.report_split_sizes \
    --tok_cfg "$TOK_CFG" \
    --train_cfg "$TRAIN_CFG" \
    --train_list "$TRAIN_LIST" \
    --val_list "$VAL_LIST" \
    --test_list "$TEST_LIST" \
    >"$JSON_OUT"

  printf '{"tokenizer_config":"%s","report_json":"%s"}\n' "$TOK_CFG" "$JSON_OUT" >>"$MANIFEST"
done

echo "Wrote reports to $OUT_DIR" >&2
echo "Manifest: $MANIFEST" >&2
