#!/usr/bin/env bash
set -uo pipefail

UV_BIN="${UV_BIN:-uv}"

if [ "$UV_BIN" = "uv" ] && ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
fi

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$UV_BIN' not found in PATH."
  echo "Tip: run this from your zsh session or set UV_BIN to the full uv path."
  exit 127
fi

commands=(
  "$UV_BIN run --project envs/data python -m src.data.midi_preprocess --help"
  "$UV_BIN run --project envs/data python -m src.data.create_dataset_splits --help"
  "$UV_BIN run --project envs/data python -m src.data.tokenizer --help"
  "$UV_BIN run --project envs/data python -m src.evaluation.score_generated_midis --help"
  "$UV_BIN run --project envs/gpt2 python -m src.models.Transformer.train --help"
  "$UV_BIN run --project envs/gpt2 python -m src.models.Transformer.generate_samples --help"
  "$UV_BIN run --project envs/xlstm python -m src.models.xlstm.train --help"
  "$UV_BIN run --project envs/xlstm python -m src.models.xlstm.generate_samples --help"
)

failures=0
for cmd in "${commands[@]}"; do
  echo "==> $cmd"
  if eval "$cmd" >/dev/null; then
    echo "PASS"
  else
    echo "FAIL"
    failures=$((failures + 1))
  fi
  echo
 done

if [ "$failures" -eq 0 ]; then
  echo "All environment checks passed."
  exit 0
fi

echo "$failures check(s) failed."
exit 1
