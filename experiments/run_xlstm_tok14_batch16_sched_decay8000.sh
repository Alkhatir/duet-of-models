#!/usr/bin/env bash
set -euo pipefail

# Train the base xLSTM with the same tokenizer and training config as the
# matched GPT-2/LLaMA transformer launcher.
#
# Optional controls:
#   TRAIN_LIST=data_reports/splits/train.txt
#   VAL_LIST=data_reports/splits/val.txt
#   TEST_LIST=data_reports/splits/test.txt
#   MODEL_CFG=configs/model/xlstm/base.yaml
#   TOKENIZER_CFG=configs/data/14.yaml
#   TRAIN_CFG=configs/train/transformer_tok14_batch16_sched_decay8000.yaml
#   EVAL_CFG=configs/eval/transformer_full_context_test.yaml
#   OUTPUT_DIR=experiments/xlstm-tok14-batch16-sched-decay8000
#   GENERATED_MIDI_ROOT=${OUTPUT_DIR}/generated_midis
#   DELETE_XLSTM_VENV_BEFORE_TEST_EVAL=1
#   DELETE_XLSTM_VENV_AFTER_GENERATION=1
#   XLSTM_VENV_DIR=envs/xlstm/.venv
#   GENERATION_DEVICE=cuda
#   START_AT=train
#
# Resume controls:
#   START_AT=train      Train xLSTM, generate full-context test samples, then score.
#   START_AT=generate   Skip training, generate from OUTPUT_DIR/best/checkpoint.pt, then score.
#   START_AT=score      Skip training/generation and only score existing generated samples.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DELETE_XLSTM_VENV_BEFORE_TEST_EVAL="${DELETE_XLSTM_VENV_BEFORE_TEST_EVAL:-1}"
export XLSTM_VENV_DIR="${XLSTM_VENV_DIR:-envs/xlstm/.venv}"

TRAIN_LIST="${TRAIN_LIST:-data_reports/splits/train.txt}"
VAL_LIST="${VAL_LIST:-data_reports/splits/val.txt}"
TEST_LIST="${TEST_LIST:-data_reports/splits/test.txt}"
MODEL_CFG="${MODEL_CFG:-configs/model/xlstm/base.yaml}"
TOKENIZER_CFG="${TOKENIZER_CFG:-configs/data/14.yaml}"
TRAIN_CFG="${TRAIN_CFG:-configs/train/transformer_tok14_batch16_sched_decay8000.yaml}"
EVAL_CFG="${EVAL_CFG:-configs/eval/transformer_full_context_test.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/xlstm-tok14-batch16-sched-decay8000}"
GENERATED_MIDI_ROOT="${GENERATED_MIDI_ROOT:-${OUTPUT_DIR}/generated_midis}"
DELETE_XLSTM_VENV_AFTER_GENERATION="${DELETE_XLSTM_VENV_AFTER_GENERATION:-1}"
GENERATION_DEVICE="${GENERATION_DEVICE:-cuda}"
START_AT="${START_AT:-train}"
CHECKPOINT_PATH="${OUTPUT_DIR}/best/checkpoint.pt"
GENERATION_DIR="${OUTPUT_DIR}/generation_full_context_test"

declare -A STAGE_ORDER=(
  [train]=0
  [generate]=1
  [score]=2
)

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
require_file "$MODEL_CFG"
require_file "$TOKENIZER_CFG"
require_file "$TRAIN_CFG"
require_file "$EVAL_CFG"

if [[ -z "${STAGE_ORDER[$START_AT]+x}" ]]; then
  echo "Unknown START_AT value: ${START_AT}" >&2
  echo "Expected one of: train, generate, score" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$GENERATED_MIDI_ROOT"

stage_enabled() {
  local stage="$1"
  [[ "${STAGE_ORDER[$stage]}" -ge "${STAGE_ORDER[$START_AT]}" ]]
}

generate_xlstm_samples() {
  echo "Generating full-context test samples for xLSTM"
  echo "  checkpoint: ${CHECKPOINT_PATH}"
  echo "  eval cfg:   ${EVAL_CFG}"
  echo "  device:     ${GENERATION_DEVICE}"
  echo "  output:     ${GENERATION_DIR}"

  require_file "$CHECKPOINT_PATH"

  uv run --project envs/xlstm python -m src.models.xlstm.generate_samples \
    --checkpoint "$CHECKPOINT_PATH" \
    --model_cfg "$MODEL_CFG" \
    --tok_cfg "$TOKENIZER_CFG" \
    --data_list "$TEST_LIST" \
    --eval_cfg "$EVAL_CFG" \
    --device "$GENERATION_DEVICE" \
    --out_dir "$GENERATION_DIR"
}

score_generated_samples() {
  echo "Scoring generated MIDIs for xLSTM"
  echo "  samples:    ${GENERATION_DIR}"
  echo "  checkpoint: ${CHECKPOINT_PATH}"

  uv run --project envs/data python -m src.evaluation.score_generated_midis \
    --samples_dir "$GENERATION_DIR" \
    --eval_cfg "$EVAL_CFG" \
    --model_type xlstm \
    --checkpoint "$CHECKPOINT_PATH"
}

collect_generated_midis() {
  local run_generated_dir="${GENERATION_DIR}/generated_midis"
  local collected_dir="${GENERATED_MIDI_ROOT}/xlstm"

  if [[ -L "$run_generated_dir" && -e "$collected_dir" ]]; then
    echo "Generated MIDIs for xLSTM already collected at ${collected_dir}"
    return
  fi
  if [[ ! -d "$run_generated_dir" && -d "$collected_dir" ]]; then
    mkdir -p "$GENERATION_DIR"
    ln -s "$(realpath -m "$collected_dir")" "$run_generated_dir"
    echo "Restored generated MIDI symlink for xLSTM: ${run_generated_dir} -> ${collected_dir}"
    return
  fi
  if [[ ! -d "$run_generated_dir" ]]; then
    echo "No generated MIDI directory found for xLSTM: ${run_generated_dir}" >&2
    exit 2
  fi
  if [[ -e "$collected_dir" ]]; then
    echo "Collected generated MIDI directory already exists: ${collected_dir}" >&2
    echo "Remove it or set GENERATED_MIDI_ROOT to a fresh directory before rerunning." >&2
    exit 2
  fi

  mv "$run_generated_dir" "$collected_dir"
  ln -s "$(realpath -m "$collected_dir")" "$run_generated_dir"
  echo "Generated MIDIs for xLSTM collected at ${collected_dir}"
}

cleanup_xlstm_venv_after_generation() {
  if [[ "$DELETE_XLSTM_VENV_AFTER_GENERATION" == "1" ]]; then
    if [[ -d "$XLSTM_VENV_DIR" ]]; then
      echo "Deleting ${XLSTM_VENV_DIR} before envs/data scoring."
      rm -rf "$XLSTM_VENV_DIR"
    else
      echo "Skipping xLSTM venv deletion: ${XLSTM_VENV_DIR} does not exist."
    fi
  else
    echo "Skipping xLSTM venv deletion: DELETE_XLSTM_VENV_AFTER_GENERATION=${DELETE_XLSTM_VENV_AFTER_GENERATION}."
  fi
}

echo "Starting xLSTM run"
echo "  START_AT:   ${START_AT}"
echo "  model:     ${MODEL_CFG}"
echo "  tokenizer: ${TOKENIZER_CFG}"
echo "  train cfg: ${TRAIN_CFG}"
echo "  eval cfg:  ${EVAL_CFG}"
echo "  test list: ${TEST_LIST}"
echo "  output:    ${OUTPUT_DIR}"
echo "  generated: ${GENERATED_MIDI_ROOT}"
echo "  cleanup:   DELETE_XLSTM_VENV_BEFORE_TEST_EVAL=${DELETE_XLSTM_VENV_BEFORE_TEST_EVAL}"
echo "  cleanup:   DELETE_XLSTM_VENV_AFTER_GENERATION=${DELETE_XLSTM_VENV_AFTER_GENERATION}"
echo "  cleanup:   XLSTM_VENV_DIR=${XLSTM_VENV_DIR}"

if stage_enabled train; then
  scripts/train_xlstm.sh \
    "$TRAIN_LIST" \
    "$VAL_LIST" \
    "$MODEL_CFG" \
    "$TOKENIZER_CFG" \
    "$OUTPUT_DIR" \
    "$TRAIN_CFG" \
    "$TEST_LIST"
else
  echo "Skipping xLSTM training because START_AT=${START_AT}."
  require_file "$CHECKPOINT_PATH"
fi

if stage_enabled generate; then
  generate_xlstm_samples
  collect_generated_midis
else
  echo "Skipping xLSTM generation because START_AT=${START_AT}."
  collect_generated_midis
fi

cleanup_xlstm_venv_after_generation
score_generated_samples

if [[ -n "${CONTAINER_ID:-}" ]] && command -v vastai >/dev/null 2>&1; then
  if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
    vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
  else
    vastai stop instance "$CONTAINER_ID"
  fi
else
  echo "Skipping Vast.ai shutdown: CONTAINER_ID is unset or vastai CLI is unavailable."
fi
