# Environment Layout (uv)

This repository uses three isolated `uv` projects:

- `envs/data`: dataset preprocessing, tokenization, and evaluation utilities.
- `envs/gpt2`: GPT-2 training stack.
- `envs/xlstm`: xLSTM training stack.

## Sync environments

```bash
uv sync --project envs/data
uv sync --project envs/gpt2
uv sync --project envs/xlstm
```

Or run all three:

```bash
scripts/bootstrap_envs.sh
```

Smoke-check all environments:

```bash
scripts/check_envs.sh
```

## Run commands by environment

```bash
uv run --project envs/data python -m src.data.midi_preprocess --help
uv run --project envs/data python -m src.data.tokenizer --help
uv run --project envs/gpt2 python -m src.models.gpt2.train --help
uv run --project envs/xlstm python -m src.models.xlstm.train --help
```

## Notes

- Keep shared code in `src/`.
- Add dependencies to the matching `envs/*/pyproject.toml`, not to the repository root.
- If a library is needed by multiple workflows, duplicate it in each relevant env.
