#!/usr/bin/env bash
set -euo pipefail

# Train matched GPT-2 and LLaMA decoder baselines with tokenizer 14.
#
# Optional controls:
#   TRAIN_LIST=data_reports/splits/train.txt
#   VAL_LIST=data_reports/splits/val.txt
#   TEST_LIST=data_reports/splits/test.txt
#   TOKENIZER_CFG=configs/data/14.yaml
#   TRAIN_CFG=configs/train/transformer_tok14_batch16_sched_decay8000.yaml
#   OUTPUT_ROOT=experiments/transformer-tok14-batch16-sched-decay8000
#   RUN_GAP_SECONDS=30

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_LIST="${TRAIN_LIST:-data_reports/splits/train.txt}"
VAL_LIST="${VAL_LIST:-data_reports/splits/val.txt}"
TEST_LIST="${TEST_LIST:-data_reports/splits/test.txt}"
TOKENIZER_CFG="${TOKENIZER_CFG:-configs/data/14.yaml}"
TRAIN_CFG="${TRAIN_CFG:-configs/train/transformer_tok14_batch16_sched_decay8000.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/transformer-tok14-batch16-sched-decay8000}"
RUN_GAP_SECONDS="${RUN_GAP_SECONDS:-30}"

declare -a RUNS=(
  "gpt2 configs/model/transformer/tok_14_gpt2_match.yaml ${OUTPUT_ROOT}/gpt2"
  "llama configs/model/transformer/tok_14_llama_match.yaml ${OUTPUT_ROOT}/llama"
)

between_runs() {
  echo "Waiting ${RUN_GAP_SECONDS}s before the next experiment..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
  sleep "$RUN_GAP_SECONDS"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 2
  fi
}

require_file "$TRAIN_LIST"
require_file "$VAL_LIST"
require_file "$TEST_LIST"
require_file "$TOKENIZER_CFG"
require_file "$TRAIN_CFG"

mkdir -p "$OUTPUT_ROOT"

total_runs="${#RUNS[@]}"
run_index=0

for run_spec in "${RUNS[@]}"; do
  read -r run_name model_cfg output_dir <<<"$run_spec"
  require_file "$model_cfg"
  run_index=$((run_index + 1))

  echo "Starting transformer run ${run_index}/${total_runs}: ${run_name}"
  echo "  model:     ${model_cfg}"
  echo "  tokenizer: ${TOKENIZER_CFG}"
  echo "  train cfg: ${TRAIN_CFG}"
  echo "  output:    ${output_dir}"

  scripts/train_gpt2.sh \
    "$TRAIN_LIST" \
    "$VAL_LIST" \
    "$TEST_LIST" \
    "$model_cfg" \
    "$TOKENIZER_CFG" \
    "$TRAIN_CFG" \
    "$output_dir"

  if [[ "$run_index" -lt "$total_runs" ]]; then
    between_runs
  fi
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
