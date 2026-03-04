#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/train_xlstm.sh <config_path>
CFG="${1:-configs/model/xlstm/basic.yaml}"

uv run --project envs/xlstm python -m src.models.xlstm.train --config "$CFG"
