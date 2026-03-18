# duet-of-models

Repository for running controlled experiments on simplified GPT-2 and xLSTM models for symbolic music generation.

## Setup

This project uses three separate `uv` environments:

- `envs/data` for preprocessing, tokenization, and evaluation
- `envs/gpt2` for transformer training
- `envs/xlstm` for xLSTM training

```bash
scripts/bootstrap_envs.sh
scripts/check_envs.sh
```

Or sync individually:

```bash
uv sync --project envs/data
uv sync --project envs/gpt2
uv sync --project envs/xlstm
```

## Core Workflow

1. Preprocess MIDI data.
2. Build/freeze tokenizer config.
3. Train GPT-2 and xLSTM with comparable settings.
4. Evaluate generated MIDI with shared metrics.
5. Store all run artifacts under `experiments/<date>-<run_name>/`.

## Commands

```bash
# representative subset sampling (recommended before preprocess on huge corpora)
scripts/sample_subset.sh <data_dir> <n> <out_list> [seed] [jobs]

# data preprocessing
scripts/preprocess_data.sh <input_dir> <output_dir> [input_list] [jobs]

# GPT-2 training
scripts/train_gpt2.sh <data_dir> <model_cfg> <tokenizer_cfg>

# xLSTM training
scripts/train_xlstm.sh <config_path>

# evaluation sweep (tokenizer + round-trip metrics)
scripts/eval_all.sh <data_dir> <out_dir>
```

Module entrypoints are used directly from `src/` (no root-level wrappers).

## Project Structure

```text
duet-of-models/
  README.md
  pyproject.toml
  uv.lock

  envs/
    data/
      pyproject.toml
    gpt2/
      pyproject.toml
    xlstm/
      pyproject.toml

  configs/
    model/
      gpt2/
      xlstm/
    data/
      tokenizer_config.yaml
      maestro.yaml
    train/
      base.yaml
    eval/
      music_metrics.yaml
    experiment/
      gpt2_baseline.yaml
      xlstm_baseline.yaml

  src/
    data/
      midi_preprocess.py
      tokenizer.py
    models/
      gpt2/
        train.py
      xlstm/
        train.py
    evaluation/
      music_metrics.py
    training/
    utils/
      midi_utils.py

  scripts/
    preprocess_data.sh
    train_gpt2.sh
    train_xlstm.sh
    eval_all.sh
    autotune_miditok_remi.py

  experiments/
    .templates/

  docs/
    environments.md
    experiment_protocol.md
    metric_definitions.md

  tests/
    test_metrics_smoke.py
```

## Notes

- Shared code lives under `src/`; model-specific code is isolated in `src/models/gpt2` and `src/models/xlstm`.
- Evaluation logic is centralized in `src/evaluation/music_metrics.py` so both models are scored consistently.
- Runtime dependencies are isolated per workflow in `envs/*/pyproject.toml`.
- Existing notebooks and `data_reports/` are kept for exploration and analysis.
- For very large MIDI corpora (e.g. 170k files), run subset sampling first and pass the generated list into preprocessing.
