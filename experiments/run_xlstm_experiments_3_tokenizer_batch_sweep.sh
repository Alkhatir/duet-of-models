#!/usr/bin/env bash
set -euo pipefail

# Tokenizer sweep followed by a batch-size sweep on the best tokenizer.
#
# This uses the best scheduled LR config from the existing base LR sweep:
# configs/train/lr_sweep/batch_8_sched_decay_8000.yaml
#
# Optional controls:
#   PHASE=tokenizer|batch|all          Default: all
#   TOKENIZER_SET=focused|full         Default: focused
#   BATCH_SWEEP_TOKENIZER_CFG=...      Override tokenizer used by PHASE=batch
#   RUN_GAP_SECONDS=30                 Delay between experiments

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

best_tokenizer_from_summaries() {
  mapfile -t tokenizers < <(select_tokenizers)

  local best_cfg=""
  local best_loss=""

  for tokenizer_cfg in "${tokenizers[@]}"; do
    local run_name="tok_$(tokenizer_name "$tokenizer_cfg")"
    local summary_path="${TOKENIZER_OUTPUT_ROOT}/${run_name}/summary.json"

    if [[ ! -f "$summary_path" ]]; then
      echo "Missing summary for ${run_name}: ${summary_path}" >&2
      continue
    fi

    local loss
    loss="$(jq -r '."best/val_loss" // .best_val_loss // empty' "$summary_path")"
    if [[ -z "$loss" ]]; then
      echo "Could not read best validation loss from ${summary_path}" >&2
      continue
    fi

    if [[ -z "$best_loss" ]] || awk -v loss="$loss" -v best="$best_loss" 'BEGIN { exit !(loss < best) }'; then
      best_loss="$loss"
      best_cfg="$tokenizer_cfg"
    fi
  done

  if [[ -z "$best_cfg" ]]; then
    echo "No tokenizer summary files were available to select a best tokenizer." >&2
    exit 3
  fi

  echo "$best_cfg"
}

run_batch_sweep() {
  local tokenizer_cfg="${BATCH_SWEEP_TOKENIZER_CFG:-}"
  if [[ -z "$tokenizer_cfg" ]]; then
    tokenizer_cfg="$(best_tokenizer_from_summaries)"
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
  between_runs
  run_batch_sweep
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
