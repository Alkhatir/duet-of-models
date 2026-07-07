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
#   OUTPUT_DIR=experiments/xlstm-tok14-batch16-sched-decay8000
#   DELETE_XLSTM_VENV_BEFORE_TEST_EVAL=1
#   XLSTM_VENV_DIR=envs/xlstm/.venv

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DELETE_XLSTM_VENV_BEFORE_TEST_EVAL="${DELETE_XLSTM_VENV_BEFORE_TEST_EVAL:-1}"
export XLSTM_VENV_DIR="${XLSTM_VENV_DIR:-envs/xlstm/.venv}"

TRAIN_LIST="${TRAIN_LIST:-data_reports/splits/train.txt}"
VAL_LIST="${VAL_LIST:-data_reports/splits/val.txt}"
TEST_LIST="${TEST_LIST:-data_reports/splits/test.txt}"
MODEL_CFG="${MODEL_CFG:-configs/model/xlstm/base.yaml}"
TOKENIZER_CFG="${TOKENIZER_CFG:-configs/data/14.yaml}"
TRAIN_CFG="${TRAIN_CFG:-configs/train/transformer_tok14_batch16_sched_decay8000.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/xlstm-tok14-batch16-sched-decay8000}"

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

echo "Starting xLSTM run"
echo "  model:     ${MODEL_CFG}"
echo "  tokenizer: ${TOKENIZER_CFG}"
echo "  train cfg: ${TRAIN_CFG}"
echo "  test list: ${TEST_LIST}"
echo "  output:    ${OUTPUT_DIR}"
echo "  cleanup:   DELETE_XLSTM_VENV_BEFORE_TEST_EVAL=${DELETE_XLSTM_VENV_BEFORE_TEST_EVAL}"
echo "  cleanup:   XLSTM_VENV_DIR=${XLSTM_VENV_DIR}"

scripts/train_xlstm.sh \
  "$TRAIN_LIST" \
  "$VAL_LIST" \
  "$MODEL_CFG" \
  "$TOKENIZER_CFG" \
  "$OUTPUT_DIR" \
  "$TRAIN_CFG" \
  "$TEST_LIST"

if [[ -n "${CONTAINER_ID:-}" ]] && command -v vastai >/dev/null 2>&1; then
  if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
    vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
  else
    vastai stop instance "$CONTAINER_ID"
  fi
else
  echo "Skipping Vast.ai shutdown: CONTAINER_ID is unset or vastai CLI is unavailable."
fi
