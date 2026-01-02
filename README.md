# duet-of-models
This repository contains tooling for preparing MIDI datasets and tokenizer configurations for duet-model experiments. It includes:

- **`preprocessing.py`**: a command-line pipeline for cleaning raw MIDI files. It maps drum pitches, trims empty tracks, filters files that are too short, deduplicates MIDIs based on a quantized signature, and writes cleaned outputs alongside optional manifest and deduplication indexes.
- **`tokenization.py`**: a builder for MidiTok YAML configs. It exposes every tokenizer option, adds tokenizer-specific extras (e.g., REMI bar tokens, MIDILike max duration, MMM density bins, PerTok microtiming controls), validates compatibility, and can scan MIDI folders to populate configuration values.
- **`utils.py`**: shared utilities, currently providing recursive MIDI discovery used by both the preprocessing and tokenization scripts.
- **Configuration files** in `configs/` for reference tokenizers and model settings, including a REMI tokenizer YAML and an xLSTM model preset.
- **Notebooks** for exploratory work (`data_exploring.ipynb`) and MIDI reproduction experiments (`reproducing_midi.ipynb`).

## Installation with `uv`

`uv` can create and manage the project virtual environment from the provided `pyproject.toml` and `uv.lock` files. To install dependencies:

```bash
uv sync
```

This will create a `.venv` folder and install all locked packages. To run project tools without activating the environment explicitly, prefix commands with `uv run`:

```bash
uv run python preprocessing.py --help
uv run python tokenization.py --help
```

## Preprocessing pipeline

The cleaning script focuses on light sanitation while leaving quantization and velocity handling to the tokenizer. Key steps include:

1. Load each MIDI file with `miditoolkit`.
2. Optionally remap drum pitches to coarse classes (or keep originals).
3. Trim empty instruments and filter out files with too few notes or bars.
4. Compute a PPQ-independent signature for deduplication based on quantized note tuples.
5. Write cleaned MIDIs, append manifest entries, and optionally save dedupe indexes.

Example usage cleaning a dataset and writing a manifest:

```bash
uv run python preprocessing.py \
  --in /path/to/raw_midis \
  --out ./clean_midis \
  --write-manifest ./manifests/clean.jsonl \
  --dedupe-index ./manifests/dedupe.json
```

Use `--save-config` or `--load-config` to persist or reuse capture settings such as drum mapping, dedupe behavior, and minimum file size thresholds.

## Tokenizer configuration builder

`tokenization.py` helps assemble MidiTok tokenizer YAML files without manually editing long dictionaries. Highlights:

- Supports REMI, MIDILike, TSD, Structured, CPWord, Octuple, MuMIDI, MMM, and PerTok tokenizers.
- Stores the full `TokenizerConfig` fields with sensible defaults and allows overriding via CLI or YAML.
- Provides tokenizer-specific extras such as bar embeddings for REMI, max duration for MIDILike, density bins for MMM, and microtiming options for PerTok.
- Can iterate over MIDI folders (using `iter_midi_paths`) to derive useful metadata when populating configs.

Generate or edit a tokenizer config by running:

```bash
uv run python tokenization.py --help
```

The generated YAML files can be saved alongside preprocessing configs for reproducibility.

## Project structure

- `configs/preprocessing/tokenizer_config.yaml`: Example REMI tokenizer configuration produced by the builder.
- `configs/models/xlstm/1.yaml`: Example model hyperparameters for an xLSTM variant.
- `data_reports/`: Space for generated reports or manifests.
- `data_exploring.ipynb`, `reproducing_midi.ipynb`: Jupyter notebooks for analysis and experimentation.

## Development notes

- The scripts target Python 3.10+.
- Dependencies are managed through `uv` using `pyproject.toml` and `uv.lock` for reproducible environments.
- To add new packages, run `uv add <package>` and commit the updated lockfile.