#!/bin/bash
set -euo pipefail

set -a
source .env
set +a

TRAIN_LIST="data_reports/splits/train.txt"
VAL_LIST="data_reports/splits/val.txt"
TOKENIZER_CFG="data_reports/tokenizer_info/tokenizers/11.yaml"

scripts/train_xlstm.sh \
  "$TRAIN_LIST" \
  "$VAL_LIST" \
  configs/model/xlstm/small.yaml \
  "$TOKENIZER_CFG" \
  experiments/xlstm-small-batch8-tok11 \
  configs/train/batch_8.yaml

scripts/train_xlstm.sh \
  "$TRAIN_LIST" \
  "$VAL_LIST" \
  configs/model/xlstm/small.yaml \
  "$TOKENIZER_CFG" \
  experiments/xlstm-small-batch16-tok11 \
  configs/train/batch_16.yaml

scripts/train_xlstm.sh \
  "$TRAIN_LIST" \
  "$VAL_LIST" \
  configs/model/xlstm/base.yaml \
  "$TOKENIZER_CFG" \
  experiments/xlstm-base-batch4-tok11 \
  configs/train/batch_4.yaml

scripts/train_xlstm.sh \
  "$TRAIN_LIST" \
  "$VAL_LIST" \
  configs/model/xlstm/base.yaml \
  "$TOKENIZER_CFG" \
  experiments/xlstm-base-batch8-tok11 \
  configs/train/batch_8.yaml
