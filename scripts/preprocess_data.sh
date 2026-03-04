#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/preprocess_data.sh <input_dir> <output_dir>
INPUT_DIR="${1:-data/raw}"
OUTPUT_DIR="${2:-data/clean}"

uv run --project envs/data python preprocessing.py \
  --in "$INPUT_DIR" \
  --out "$OUTPUT_DIR" \
  --save-config configs/data/capture.yaml \
  --write-manifest data_reports/clean_manifest.jsonl
