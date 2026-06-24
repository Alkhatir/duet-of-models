#!/usr/bin/env bash
set -euo pipefail

# Tokenizer sweep with an optional explicit batch-size sweep.
#
# This uses the best scheduled LR config from the existing base LR sweep:
# configs/train/lr_sweep/batch_8_sched_decay_8000.yaml
#
# Optional controls:
#   PHASE=tokenizer|batch|all          Default: all
#   TOKENIZER_SET=focused|full         Default: full
#   BATCH_SWEEP_TOKENIZER_CFG=...      Required by PHASE=batch
#   RUN_GAP_SECONDS=30                 Delay between experiments
#
# PHASE=all intentionally runs only the tokenizer sweep. It does not inspect
# summary.json files or pick a best tokenizer automatically.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PHASE="${PHASE:-all}"
TOKENIZER_SET="${TOKENIZER_SET:-full}"
RUN_GAP_SECONDS="${RUN_GAP_SECONDS:-30}"

TRAIN_LIST="data_reports/splits/train.txt"
VAL_LIST="data_reports/splits/val.txt"
MODEL_CFG="configs/model/xlstm/base.yaml"
BASE_TRAIN_CFG="configs/train/lr_sweep/batch_8_sched_decay_8000.yaml"
OUTPUT_ROOT="experiments/xlstm-base-tokenizer-batch-sweep-sched-decay-8000"
TOKENIZER_OUTPUT_ROOT="${OUTPUT_ROOT}/tokenizer_sweep"
BATCH_OUTPUT_ROOT="${OUTPUT_ROOT}/batch_sweep"
GENERATED_CFG_DIR="${OUTPUT_ROOT}/generated_train_configs"

FOCUSED_TOKENIZERS=(
  "configs/data/11.yaml"
  "configs/data/15.yaml"
  "configs/data/11_chords.yaml"
  "configs/data/15_chords.yaml"
)

FULL_TOKENIZERS=(
  "configs/data/10.yaml"
  "configs/data/11.yaml"
  "configs/data/14.yaml"
  "configs/data/15.yaml"
  "configs/data/10_chords.yaml"
  "configs/data/11_chords.yaml"
  "configs/data/14_chords.yaml"
  "configs/data/15_chords.yaml"
)

BATCHES=(4 8 16)

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

select_tokenizers() {
  case "$TOKENIZER_SET" in
  focused)
    printf "%s\n" "${FOCUSED_TOKENIZERS[@]}"
    ;;
  full)
    printf "%s\n" "${FULL_TOKENIZERS[@]}"
    ;;
  *)
    echo "Unknown TOKENIZER_SET=${TOKENIZER_SET}. Expected focused or full." >&2
    exit 2
    ;;
  esac
}

run_tokenizer_sweep() {
  mkdir -p "$TOKENIZER_OUTPUT_ROOT"

  mapfile -t tokenizers < <(select_tokenizers)
  for index in "${!tokenizers[@]}"; do
    local tokenizer_cfg="${tokenizers[$index]}"
    local run_name="tok_$(tokenizer_name "$tokenizer_cfg")"
    local output_dir="${TOKENIZER_OUTPUT_ROOT}/${run_name}"

    echo "Starting tokenizer sweep run: ${run_name}"
    scripts/train_xlstm.sh \
      "$TRAIN_LIST" \
      "$VAL_LIST" \
      "$MODEL_CFG" \
      "$tokenizer_cfg" \
      "$output_dir" \
      "$BASE_TRAIN_CFG"

    if [[ "$index" -lt "$((${#tokenizers[@]} - 1))" ]]; then
      between_runs
    fi
  done
}

run_batch_sweep() {
  local tokenizer_cfg="${BATCH_SWEEP_TOKENIZER_CFG:-}"
  if [[ -z "$tokenizer_cfg" ]]; then
    echo "BATCH_SWEEP_TOKENIZER_CFG is required for PHASE=batch." >&2
    echo "Example: PHASE=batch BATCH_SWEEP_TOKENIZER_CFG=configs/data/11.yaml $0" >&2
    exit 3
  fi

  if [[ ! -f "$tokenizer_cfg" ]]; then
    echo "Tokenizer config not found: ${tokenizer_cfg}" >&2
    exit 3
  fi

  mkdir -p "$BATCH_OUTPUT_ROOT"

  local tokenizer_label
  tokenizer_label="$(tokenizer_name "$tokenizer_cfg")"

  for index in "${!BATCHES[@]}"; do
    local batch_size="${BATCHES[$index]}"
    local train_cfg
    train_cfg="$(make_batch_train_cfg "$batch_size")"
    local output_dir="${BATCH_OUTPUT_ROOT}/tok_${tokenizer_label}_batch_${batch_size}"

    echo "Starting batch sweep run: tokenizer=${tokenizer_cfg}, batch=${batch_size}"
    scripts/train_xlstm.sh \
      "$TRAIN_LIST" \
      "$VAL_LIST" \
      "$MODEL_CFG" \
      "$tokenizer_cfg" \
      "$output_dir" \
      "$train_cfg"

    if [[ "$index" -lt "$((${#BATCHES[@]} - 1))" ]]; then
      between_runs
    fi
  done
}

case "$PHASE" in
tokenizer)
  run_tokenizer_sweep
  ;;
batch)
  run_batch_sweep
  ;;
all)
  run_tokenizer_sweep
  ;;
*)
  echo "Unknown PHASE=${PHASE}. Expected tokenizer, batch, or all." >&2
  exit 2
  ;;
esac

if [[ -n "${CONTAINER_ID:-}" ]] && command -v vastai >/dev/null 2>&1; then
  if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
    vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
  else
    vastai stop instance "$CONTAINER_ID"
  fi
else
  echo "Skipping Vast.ai shutdown: CONTAINER_ID is unset or vastai CLI is unavailable."
fi
