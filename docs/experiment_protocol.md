# Experiment Protocol

1. Preprocess raw MIDI files with `scripts/preprocess_data.sh`.
2. Freeze tokenizer and model configs under `configs/`.
3. Train GPT-2 and xLSTM with matching dataset splits and seed.
4. Save artifacts in `experiments/<date>-<run_name>/`.
5. Run evaluation metrics and store outputs as JSON/CSV in the same run folder.
