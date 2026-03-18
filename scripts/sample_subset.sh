#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/sample_subset.sh <input_dir> <n> <out_list> [seed] [jobs]
INPUT_DIR="${1:-data/raw}"
N="${2:-2000}"
OUT_LIST="${3:-data_reports/sampled_subset.txt}"
SEED="${4:-42}"
JOBS="${5:-0}"

REPORT="data_reports/sampled_subset_report.json"
CACHE="data_reports/sampled_subset_cache.jsonl"

uv run --project envs/data python -m src.data.sample_midi_subset \
  --in "$INPUT_DIR" \
  --n "$N" \
  --seed "$SEED" \
  --jobs "$JOBS" \
  --out-list "$OUT_LIST" \
  --report "$REPORT" \
  --manifest-cache "$CACHE"
