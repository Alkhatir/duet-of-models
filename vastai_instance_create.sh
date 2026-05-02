#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-alkhatir/duet-of-models:cuda12.1-xlstm-vast}"
REPO_URL="${REPO_URL:-https://github.com/alkhatir/duet-of-models.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
DISK_GB="${DISK_GB:-32}"
RUN_AFTER_SETUP="${RUN_AFTER_SETUP:-experiments/run_xlstm_experiments_1.sh}"

OFFER_ID=$(vastai search offers \
  'gpu_name in [RTX_3090_Ti,RTX_3090,RTX_3080_Ti,RTX_3080,RTX_3070_Ti,RTX_3070,RTX_3070_Laptop,RTX_3060_Ti,RTX_3060,RTX_3060_Laptop,RTX_3050,RTX_4090_D,RTX_4090,RTX_4080_Super,RTX_4080,RTX_4070_Ti_Super,RTX_4070_Ti,RTX_4070_Super,RTX_4070,RTX_4070_Laptop,RTX_4060_Ti,RTX_4060,RTX_4060_Laptop,RTX_5090,RTX_5080,RTX_5070_Ti,RTX_5070,RTX_5060_Ti,RTX_5060,RTX_6000_Ada_Generation,RTX_5880_Ada_Generation,RTX_5000_Ada_Generation,RTX_4500_Ada_Generation,RTX_4000_Ada_Generation,RTX_A6000,RTX_A5000,RTX_A4500,RTX_A4000,RTX_A2000,A800_PCIE,A100_PCIE,A100_SXM4,A100_SXM,A100X,GH200_SXM,H100_NVL,H100_PCIE,H100_SXM,H200_NVL,H200_SXM] gpu_ram>=12 disk_space>=32 duration>=7 rentable=true verified=true reliability>=0.9 cuda_vers>=12.1' \
  -o 'dph' --raw | jq -r '.[0].id')

if [[ -z "$OFFER_ID" || "$OFFER_ID" == "null" ]]; then
  echo "No matching Vast.ai offer found." >&2
  exit 1
fi

vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --ssh \
  --direct \
  --env "-e REPO_BRANCH=$REPO_BRANCH -e RUN_AFTER_SETUP=$RUN_AFTER_SETUP" \
  --onstart-cmd "bash /entry.sh"
