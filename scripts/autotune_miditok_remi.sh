#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/autotune_miditok_remi.sh [input_list] [out_dir] [max_files] [grid_file] [extra_args...]
INPUT_LIST="${1:-data_reports/sampled_subset.txt}"
OUT_DIR="${2:-data_reports/autotune_miditok_remi}"
MAX_FILES="${3:-200}"
GRID_FILE="${4:-}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
QUIET="${QUIET:-0}"
JOBS="${JOBS:-0}"
shift $(( $# < 4 ? $# : 4 ))

CMD=(uv run --project envs/data python -m src.data.autotune_miditok_remi
  --input-list "$INPUT_LIST"
  --max-files "$MAX_FILES"
  --outdir "$OUT_DIR"
  --log-level "$LOG_LEVEL"
  --jobs "$JOBS"
)

if [[ -n "$GRID_FILE" ]]; then
  CMD+=(--grid-file "$GRID_FILE")
fi

if [[ "$QUIET" == "1" || "$QUIET" == "true" || "$QUIET" == "yes" ]]; then
  CMD+=(--quiet)
fi

"${CMD[@]}" "$@"
