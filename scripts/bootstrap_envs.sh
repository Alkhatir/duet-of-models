#!/usr/bin/env bash
set -euo pipefail

uv sync --project envs/data
uv sync --project envs/gpt2
uv sync --project envs/xlstm
