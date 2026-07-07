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
#   EVAL_CFG=configs/eval/transformer_full_context_test.yaml
#   OUTPUT_ROOT=experiments/transformer-tok14-batch16-sched-decay8000
#   GENERATED_MIDI_ROOT=${OUTPUT_ROOT}/generated_midis
#   RUN_GAP_SECONDS=30
#   DELETE_GPT2_VENV_AFTER_GENERATION=1

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_LIST="${TRAIN_LIST:-data_reports/splits/train.txt}"
VAL_LIST="${VAL_LIST:-data_reports/splits/val.txt}"
TEST_LIST="${TEST_LIST:-data_reports/splits/test.txt}"
TOKENIZER_CFG="${TOKENIZER_CFG:-configs/data/14.yaml}"
TRAIN_CFG="${TRAIN_CFG:-configs/train/transformer_tok14_batch16_sched_decay8000.yaml}"
EVAL_CFG="${EVAL_CFG:-configs/eval/transformer_full_context_test.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/transformer-tok14-batch16-sched-decay8000}"
GENERATED_MIDI_ROOT="${GENERATED_MIDI_ROOT:-${OUTPUT_ROOT}/generated_midis}"
RUN_GAP_SECONDS="${RUN_GAP_SECONDS:-30}"
DELETE_GPT2_VENV_AFTER_GENERATION="${DELETE_GPT2_VENV_AFTER_GENERATION:-1}"
GPT2_VENV_DIR="${GPT2_VENV_DIR:-envs/gpt2/.venv}"

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
require_file "$EVAL_CFG"

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$GENERATED_MIDI_ROOT"

declare -a SCORE_SPECS=()

generate_transformer_samples() {
  local run_name="$1"
  local model_cfg="$2"
  local checkpoint_dir="$3"
  local generation_dir="$4"

  echo "Generating full-context test samples for ${run_name}"
  echo "  checkpoint: ${checkpoint_dir}"
  echo "  eval cfg:   ${EVAL_CFG}"
  echo "  output:     ${generation_dir}"

  uv run --project envs/gpt2 python -m src.models.Transformer.generate_samples \
    --checkpoint "$checkpoint_dir" \
    --model_cfg "$model_cfg" \
    --tok_cfg "$TOKENIZER_CFG" \
    --data_list "$TEST_LIST" \
    --eval_cfg "$EVAL_CFG" \
    --out_dir "$generation_dir"
}

score_generated_samples() {
  local run_name="$1"
  local checkpoint_dir="$2"
  local generation_dir="$3"

  echo "Scoring generated MIDIs for ${run_name}"
  echo "  samples:    ${generation_dir}"
  echo "  checkpoint: ${checkpoint_dir}"

  uv run --project envs/data python -m src.evaluation.score_generated_midis \
    --samples_dir "$generation_dir" \
    --eval_cfg "$EVAL_CFG" \
    --model_type transformer \
    --checkpoint "$checkpoint_dir"
}

collect_generated_midis() {
  local run_name="$1"
  local generation_dir="$2"
  local run_generated_dir="${generation_dir}/generated_midis"
  local collected_dir="${GENERATED_MIDI_ROOT}/${run_name}"

  if [[ ! -d "$run_generated_dir" ]]; then
    echo "No generated MIDI directory found for ${run_name}: ${run_generated_dir}" >&2
    exit 2
  fi
  if [[ -e "$collected_dir" ]]; then
    echo "Collected generated MIDI directory already exists: ${collected_dir}" >&2
    echo "Remove it or set GENERATED_MIDI_ROOT to a fresh directory before rerunning." >&2
    exit 2
  fi

  mv "$run_generated_dir" "$collected_dir"
  ln -s "$(realpath -m "$collected_dir")" "$run_generated_dir"
  echo "Generated MIDIs for ${run_name} collected at ${collected_dir}"
}

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

  generation_dir="${output_dir}/generation_full_context_test"
  generate_transformer_samples "$run_name" "$model_cfg" "$output_dir" "$generation_dir"
  collect_generated_midis "$run_name" "$generation_dir"
  SCORE_SPECS+=("${run_name} ${output_dir} ${generation_dir}")

  if [[ "$run_index" -lt "$total_runs" ]]; then
    between_runs
  fi
done

if [[ "$DELETE_GPT2_VENV_AFTER_GENERATION" == "1" ]]; then
  if [[ -d "$GPT2_VENV_DIR" ]]; then
    echo "Deleting ${GPT2_VENV_DIR} before envs/data scoring."
    rm -rf "$GPT2_VENV_DIR"
  else
    echo "Skipping GPT-2 venv deletion: ${GPT2_VENV_DIR} does not exist."
  fi
else
  echo "Skipping GPT-2 venv deletion: DELETE_GPT2_VENV_AFTER_GENERATION=${DELETE_GPT2_VENV_AFTER_GENERATION}."
fi

for score_spec in "${SCORE_SPECS[@]}"; do
  read -r run_name output_dir generation_dir <<<"$score_spec"
  score_generated_samples "$run_name" "$output_dir" "$generation_dir"
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
