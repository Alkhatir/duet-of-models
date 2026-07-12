# MusPy Test-Set Reference Baseline

This directory contains a MusPy-only scorer run over every MIDI listed in `data_reports/splits/test.txt`. It is a reference-equals-generated baseline: `generated.mid` is hardlinked or copied from `reference.mid`, so MusPy absolute differences should be zero for successfully scored samples. FMD is disabled.

Run summary:

- `num_samples`: 200
- `score_success_rate`: 1.0
- `fmd_enabled`: false
- `roundtrip.enabled`: false
- Aggregate results: `aggregate_metrics.json`
- Per-sample results: `per_sample_metrics.json`
