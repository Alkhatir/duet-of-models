#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/create_dataset_splits.sh <sampled_list> [out_dir] [seed]
SAMPLED_LIST="${1:-data_reports/sampled_subset.txt}"
OUT_DIR="${2:-data_reports/splits}"
SEED="${3:-42}"

uv run --project envs/data python -m src.data.create_dataset_splits \
  --input-list "$SAMPLED_LIST" \
  --out-dir "$OUT_DIR" \
  --seed "$SEED" \
  --manifest-cache data_reports/splits_cache.jsonl
