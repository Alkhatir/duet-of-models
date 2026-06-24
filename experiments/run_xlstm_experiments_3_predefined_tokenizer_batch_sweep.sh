#!/usr/bin/env bash
set -euo pipefail

# Batch sweep for an explicit tokenizer set.
#
# This script does not inspect local summary.json files. It runs every
# tokenizer x batch combination listed below, which makes it suitable for
# remote training where W&B summaries are downloaded later.
#
# Optional controls:
#   TOKENIZER_CONFIGS="configs/data/11.yaml configs/data/15.yaml"
#   BATCH_SIZES="4 8 16"
#   RUN_GAP_SECONDS=30

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_GAP_SECONDS="${RUN_GAP_SECONDS:-30}"

TRAIN_LIST="data_reports/splits/train.txt"
VAL_LIST="data_reports/splits/val.txt"
MODEL_CFG="configs/model/xlstm/base.yaml"
BASE_TRAIN_CFG="configs/train/lr_sweep/batch_8_sched_decay_8000.yaml"
OUTPUT_ROOT="experiments/xlstm-base-predefined-tokenizer-batch-sweep-sched-decay-8000"
GENERATED_CFG_DIR="${OUTPUT_ROOT}/generated_train_configs"

DEFAULT_TOKENIZERS=(
  "configs/data/11.yaml"
  "configs/data/15.yaml"
  "configs/data/11_chords.yaml"
  "configs/data/15_chords.yaml"
)

DEFAULT_BATCHES=(4 8 16)

between_runs() {
  echo "Waiting ${RUN_GAP_SECONDS}s before the next experiment..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
  sleep "$RUN_GAP_SECONDS"
}

tokenizer_name() {
  basename "$1" .yaml
}

make_batch_train_cfg() {
  local batch_size="$1"
  local out_cfg="${GENERATED_CFG_DIR}/batch_${batch_size}_sched_decay_8000.yaml"

  mkdir -p "$GENERATED_CFG_DIR"
  sed -E \
    -e "s/^([[:space:]]*per_device_train_batch_size: ).*/\1${batch_size}/" \
    -e "s/^([[:space:]]*per_device_eval_batch_size: ).*/\1${batch_size}/" \
    "$BASE_TRAIN_CFG" >"$out_cfg"

  echo "$out_cfg"
}

load_tokenizers() {
  if [[ -n "${TOKENIZER_CONFIGS:-}" ]]; then
    read -r -a tokenizers <<<"$TOKENIZER_CONFIGS"
  else
    tokenizers=("${DEFAULT_TOKENIZERS[@]}")
  fi
}

load_batches() {
  if [[ -n "${BATCH_SIZES:-}" ]]; then
    read -r -a batches <<<"$BATCH_SIZES"
  else
    batches=("${DEFAULT_BATCHES[@]}")
  fi
}

load_tokenizers
load_batches

total_runs=$((${#tokenizers[@]} * ${#batches[@]}))
run_index=0

mkdir -p "$OUTPUT_ROOT"

for tokenizer_cfg in "${tokenizers[@]}"; do
  if [[ ! -f "$tokenizer_cfg" ]]; then
    echo "Tokenizer config not found: ${tokenizer_cfg}" >&2
    exit 2
  fi

  tokenizer_label="$(tokenizer_name "$tokenizer_cfg")"

  for batch_size in "${batches[@]}"; do
    run_index=$((run_index + 1))
    train_cfg="$(make_batch_train_cfg "$batch_size")"
    output_dir="${OUTPUT_ROOT}/tok_${tokenizer_label}_batch_${batch_size}"

    echo "Starting run ${run_index}/${total_runs}: tokenizer=${tokenizer_cfg}, batch=${batch_size}"
    scripts/train_xlstm.sh \
      "$TRAIN_LIST" \
      "$VAL_LIST" \
      "$MODEL_CFG" \
      "$tokenizer_cfg" \
      "$output_dir" \
      "$train_cfg"

    if [[ "$run_index" -lt "$total_runs" ]]; then
      between_runs
    fi
  done
done

if [[ -n "${CONTAINER_ID:-}" ]] && command -v vastai >/dev/null 2>&1; then
  if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
    vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
  else
    vastai stop instance "$CONTAINER_ID"
  fi
else
  echo "Skipping Vast.ai shutdown: CONTAINER_ID is unset or vastai CLI is unavailable."
fi
