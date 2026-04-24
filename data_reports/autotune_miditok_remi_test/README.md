# Auto-tuning results

- `results_ranked.csv`: ranked metrics per config with `score` (lower is better)
- `best_config.json`: JSON-safe winning TokenizerConfig dict for inspection
- `best_config.yaml`: winning TokenizerConfig dict with Python tuple keys preserved
- `best_tokenizer.yaml`: winning config in this repo's MidiTokBuilder format

Edit weights via `--weights` to reflect your priorities.

## Info about the run

This run was done with 50 files sampled from the `test` split