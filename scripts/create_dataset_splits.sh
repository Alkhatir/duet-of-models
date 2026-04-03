#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/create_dataset_splits.sh <subset_size> [genre_csv] [out_dir] [seed] [jobs] [genres_csv]
SUBSET_SIZE="${1:?subset_size required}"
GENRE_CSV="${2:-data_reports/genre_dist.csv}"
OUT_DIR="${3:-data_reports/splits}"
SEED="${4:-42}"
JOBS="${5:-0}"
GENRES_CSV="${6:-}"
SUBSET_LIST_OUT="$(dirname "$OUT_DIR")/sampled_subset.txt"

GENRE_ARGS=()
if [[ -n "$GENRES_CSV" ]]; then
  IFS=',' read -r -a GENRE_LIST <<< "$GENRES_CSV"
  GENRE_ARGS=(--genres "${GENRE_LIST[@]}")
fi

uv run --project envs/data python -m src.data.create_dataset_splits \
  --genre-csv "$GENRE_CSV" \
  --subset-size "$SUBSET_SIZE" \
  --subset-list-out "$SUBSET_LIST_OUT" \
  --out-dir "$OUT_DIR" \
  --seed "$SEED" \
  --jobs "$JOBS" \
  --manifest-cache data_reports/splits_cache.jsonl \
  "${GENRE_ARGS[@]}"
