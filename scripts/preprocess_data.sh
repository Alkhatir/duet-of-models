#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/preprocess_data.sh <input_dir> <output_dir> [input_list] [jobs]
INPUT_DIR="${1:-data/raw}"
OUTPUT_DIR="${2:-data/clean}"
INPUT_LIST="${3:-}"
JOBS="${4:-0}"

CMD=(uv run --project envs/data python -m src.data.midi_preprocess
  --in "$INPUT_DIR"
  --out "$OUTPUT_DIR"
  --jobs "$JOBS"
  --save-config configs/data/capture.yaml
  --write-manifest data_reports/clean_manifest.jsonl
)

if [ -n "$INPUT_LIST" ]; then
  CMD+=(--input-list "$INPUT_LIST")
fi

"${CMD[@]}"
