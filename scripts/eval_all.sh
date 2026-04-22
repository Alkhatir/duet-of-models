#!/usr/bin/env bash
set -euo pipefail

# Example evaluation sweep over tokenizer configs using onset/chroma metrics.
DATA_DIR="${1:-data/lmd_matched}"
OUT_DIR="${2:-experiments/$(date +%F)-eval}"

mkdir -p "$OUT_DIR"
uv run --project envs/data python scripts/autotune_miditok_remi.py \
  --midi-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR"
