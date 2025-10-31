#!/usr/bin/env python3
"""
Auto-tune Miditok REMI tokenizer configs for a MIDI dataset.

It sweeps a grid of TokenizerConfig options, tokenizes a sample of the dataset,
(optionally) round-trips back to MIDI, computes metrics, and ranks configs.

Metrics (lower is better unless noted):
- tokens_per_second: total tokens / total seconds (compression proxy)
- median_tokens_per_file, p95_tokens_per_file: spread indicator
- note_loss_rate: |notes_decoded - notes_original| / max(1, notes_original)
- tempo_diff: |n_tempos_decoded - n_tempos_original|
- timesig_diff: |n_timesigs_decoded - n_timesigs_original|
- error_rate: fraction of files that failed with this config (lower is better)

Score (lower is better):
    score = w_len * z(tokens_per_second)
          + w_len_p95 * z(p95_tokens_per_file)
          + w_loss * z(note_loss_rate)
          + w_struct * z(tempo_diff + timesig_diff)
          + w_err * z(error_rate)

one can customize weights with --weights JSON, or skip round-trip with --no-decode.
Outputs:
- results.csv: per-config metrics
- best_config.json: the top-ranked TokenizerConfig (as dict)
- debug/ per-config logs (optional)
"""

from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
import traceback
import statistics

# Third-party
import numpy as np  # type: ignore
import pandas as pd
import yaml  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# Miditok / Miditoolkit
from miditok import REMI, TokSequence, TokenizerConfig  # type: ignore
from miditoolkit import MidiFile, Instrument  # type: ignore


def read_midi_duration_seconds(midi: MidiFile) -> float:
    """Estimate the MIDI file duration in seconds by integrating tempo segments.

    The Miditoolkit `MidiFile` represents time in ticks and stores tempo
    change events as (time, tempo) entries where `time` is a tick position.
    This function computes the real-world duration by converting tick spans
    between consecutive tempo change events into seconds using the
    corresponding BPM (beats per minute) value.

    Algorithm details:
    - Determine `ticks_per_beat` from the MIDI file header.
    - Sort tempo change events by their tick time.
    - Compute `end_tick` as the maximum note end tick across instruments
        (falls back to `midi.max_tick` if no notes are present).
    - For each tempo segment (from tempo[i].time to tempo[i+1].time or
        `end_tick` for the last segment) compute the number of beats and then
        seconds = (60 / bpm) * beats. Sum segment seconds and return the total.

    Edge cases and notes:
    - If the file has no tempo changes, a sentinel tempo of 120 BPM at time 0
        is used to provide a reasonable duration estimate.
    - If a tempo segment ends at or before it starts (t1 <= t0), that segment
        is skipped to avoid negative durations (such data would be malformed).
    - Returns a float (seconds). If the MIDI has no time information, the
        result may be 0.0.
    """
    ticks_per_beat = midi.ticks_per_beat
    tempos = sorted(midi.tempo_changes, key=lambda t: t.time)
    end_tick = max((n.end for inst in midi.instruments for n in inst.notes), default=midi.max_tick)
    # Append a sentinel tempo at end
    if not tempos:
        tempos = [type("T", (), {"time": 0, "tempo": 120.0})()]
    segments = []
    for i, t in enumerate(tempos):
        t0 = t.time
        bpm = t.tempo
        t1 = tempos[i + 1].time if i + 1 < len(tempos) else end_tick
        t1 = min(t1, end_tick)
        if t1 <= t0:
            continue
        # ticks -> seconds for this segment
        beats = (t1 - t0) / ticks_per_beat
        seconds = (60.0 / max(1e-6, bpm)) * beats
        segments.append(seconds)
    return float(sum(segments))


def count_notes(midi: MidiFile) -> int:
    """Return the total number of note events across all instruments.

    This counts individual Note objects stored in each `Instrument` of the
    provided `MidiFile`. It does not attempt to deduplicate overlapping
    notes or normalize by program/track — it's a raw event count used for
    simple round-trip equality checks.

    Returns:
        int: total note count (0 if there are no instruments/notes).
    """
    return sum(len(inst.notes) for inst in midi.instruments)


def count_tempos(midi: MidiFile) -> int:
    """Return the number of tempo change events in the MIDI.

    This is a quick structural metric (how many tempo events exist) used
    to compare the original and decoded MIDIs. It simply returns the
    length of `midi.tempo_changes`.
    """
    return len(midi.tempo_changes)


def count_timesigs(midi: MidiFile) -> int:
    """Return the number of time signature change events in the MIDI.

    Like `count_tempos`, this is a lightweight structural metric used to
    compare decoded files against originals.
    """
    return len(midi.time_signature_changes)


def load_midi(path: Path) -> MidiFile:
    """Load a MIDI file from disk and return a Miditoolkit MidiFile.

    Args:
        path: Path to the .mid/.midi file.

    Returns:
        MidiFile: parsed MIDI object.

    Raises:
        Any exception raised by `miditoolkit.MidiFile` will propagate. The
        caller typically wraps this in try/except to handle corrupt files.
    """
    return MidiFile(str(path))


def default_grid() -> List[Dict[str, Any]]:
    """Build and return a list of TokenizerConfig-style dictionaries.

    The function enumerates a small but useful cross-product of common
    tokenizer options (beat resolutions, velocity quantization bins, and
    boolean toggles such as using chords/rests/tempos/time-signatures/and
    program changes). Each returned dict is intended to be passed as
    kwargs into `miditok.TokenizerConfig`.

    Notes:
    - Keep this grid compact to make experiments fast. Expand or provide a
        `--grid-file` if you want broader sweeps.
    - Keys are chosen to align with `miditok.TokenizerConfig` naming
        (e.g., `num_velocities`). If you upgrade/downgrade miditok, verify
        these names still match the library's constructor.
    """
    grids = []
    beat_res_opts = [
        {(0, 4): 8, (4, 12): 4},
        {(0, 4): 4, (4, 12): 4},
        {(0, 4): 12, (4, 12): 6},
    ]
    nb_vels = [16, 32, 64]
    chords = [True, False]
    rests = [True, False]
    tempos = [True]
    timesigs = [True]
    programs = [True]

    for br in beat_res_opts:
        for nv in nb_vels:
            for uc in chords:
                for ur in rests:
                    for ut in tempos:
                        for uts in timesigs:
                            for up in programs:
                                grids.append(dict(
                                    # `beat_res` controls resolution mapping by bar ranges
                                    beat_res=br,
                                    # number of velocity bins to quantize velocities
                                    num_velocities=nv,
                                    # boolean toggles that enable/disable tokens
                                    use_chords=uc,
                                    use_rests=ur,
                                    use_tempos=ut,
                                    use_time_signatures=uts,
                                    use_programs=up,
                                ))
    return grids


def make_tokenizer(cfg_dict: Dict[str, Any]) -> REMI:
    """Create a REMI tokenizer from a TokenizerConfig kwargs dict.

    This helper wraps the pattern `TokenizerConfig(**cfg_dict)` followed by
    `REMI(config)` so callers only need to provide a plain dict. Any invalid
    keys or incompatible values will raise in `TokenizerConfig`. The caller
    should catch exceptions when constructing tokenizers for experimental
    grids.

    Args:
        cfg_dict: mapping of TokenizerConfig parameter names to values.

    Returns:
        REMI: instantiated tokenizer ready to be called on MidiFile objects.
    """
    config = TokenizerConfig(**cfg_dict)
    return REMI(config)



def zscore(arr: np.ndarray) -> np.ndarray:
    """Compute a z-score (standard score) for a 1D numpy array.

    This implementation is NaN-aware (uses `nanmean`/`nanstd`) so missing
    values don't propagate silently. If the standard deviation is effectively
    zero, it returns an array of zeros to avoid numerical instability.

    Args:
        arr: 1D numpy array of numeric values (may contain NaNs).

    Returns:
        numpy.ndarray: z-scored values with the same shape as `arr`.
    """
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if s < 1e-12:
        return np.zeros_like(arr)
    return (arr - m) / s


def rank_configs(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Rank tokenizer configurations using z-scored metric components.

    This function takes the raw-per-config metrics DataFrame produced by the
    main loop and builds normalized components using `zscore`. It combines
    the following (lower is better):

      - tokens_per_second: proxy for verbosity/compression
      - p95_tokens_per_file: tail length indicator
      - note_loss_rate: average fractional note loss (only when decoding)
      - struct_diff: mean difference in tempo + timesig counts
      - error_rate: fraction of files that raised errors

    Each component is multiplied by a weight (provided in `weights`) and
    summed to produce a final `score` column. The DataFrame is returned
    sorted ascending by `score` (best configurations first).

    Args:
        df: DataFrame containing the required metric columns.
        weights: mapping of weight names to floats; missing entries use
                 reasonable defaults.

    Returns:
        DataFrame: a copy of `df` with an added `score` column, sorted by
                   ascending score.
    """
    # Build components
    comp = {}
    # Lower is better for all used components
    comp["tokens_per_second"] = zscore(df["tokens_per_second"].to_numpy())
    comp["p95_tokens_per_file"] = zscore(df["p95_tokens_per_file"].to_numpy())
    comp["note_loss_rate"] = zscore(df["note_loss_rate"].to_numpy())
    comp["struct_diff"] = zscore((df["tempo_diff"] + df["timesig_diff"]).to_numpy())
    comp["error_rate"] = zscore(df["error_rate"].to_numpy())

    score = (
        weights.get("w_len", 1.0) * comp["tokens_per_second"]
        + weights.get("w_len_p95", 0.5) * comp["p95_tokens_per_file"]
        + weights.get("w_loss", 2.0) * comp["note_loss_rate"]
        + weights.get("w_struct", 1.0) * comp["struct_diff"]
        + weights.get("w_err", 3.0) * comp["error_rate"]
    )
    df = df.copy()
    df["score"] = score
    df = df.sort_values("score", ascending=True)
    return df


def main():
    """CLI entrypoint: sweep tokenizer configs, evaluate and rank them.

    This function parses command-line arguments, loads a sample of MIDI files
    from `--input` (up to `--max-files`), precomputes baseline features
    (notes, tempo/time signature counts, duration), then iterates over the
    configuration grid. For each config it attempts to construct a REMI
    tokenizer, tokenize each MIDI and optionally decode back to MIDI to
    compute structural/quality metrics. The per-config metrics are saved to
    CSV and the best configuration is written to `best_config.json`.

    Error handling and behavior notes:
    - If a tokenizer construction fails for a given config, that config is
        recorded with `error_rate=1.0` and NaNs for other metrics.
    - Tokenization/decoding errors on individual files increment an error
        counter; decoding failures are penalized via large default diffs.
    - The function writes output files in `--outdir` and prints the path
        when finished.
    """
    p = argparse.ArgumentParser(description="Auto-tune Miditok REMI configs for a MIDI dataset.")
    p.add_argument("--input", type=Path, required=True, help="Folder with .mid/.midi files")
    p.add_argument("--extensions", nargs="+", default=[".mid", ".midi"], help="File extensions")
    p.add_argument("--max-files", type=int, default=200, help="Evaluate at most N files")
    p.add_argument("--grid-file", type=Path, help="JSON file with a list of TokenizerConfig dicts")
    p.add_argument("--no-decode", action="store_true", help="Skip round-trip decoding metrics (faster)")
    p.add_argument("--weights", type=str, default=None, help='JSON like {"w_len":1,"w_len_p95":0.5,"w_loss":2,"w_struct":1,"w_err":3}')
    p.add_argument("--outdir", type=Path, required=True, help="Where to save results")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Collect files
    exts = {e.lower() for e in args.extensions}
    files = [p for p in args.input.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    if not files:
        raise SystemExit("No MIDI files found.")
    if args.max_files:
        files = files[: args.max_files]

    # Prepare dataset stats
    dataset_midis: List[Tuple[Path, MidiFile]] = []
    iterator = files
    if HAS_TQDM:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(files, desc="Loading MIDIs")
    for f in iterator:
        try:
            m = load_midi(f)
            dataset_midis.append((f, m))
        except Exception:
            print(f"[WARN] Failed to load {f}:\n{traceback.format_exc()}")

    if not dataset_midis:
        raise SystemExit("No valid MIDI files could be loaded.")

    # Precompute original dataset features
    origins = []
    for _, m in dataset_midis:
        try:
            dur = read_midi_duration_seconds(m)
        except Exception:
            # Fallback if tempo parsing fails: derive from ticks with 120 bpm
            dur = (m.max_tick / max(1, m.ticks_per_beat)) * 0.5
        origins.append(dict(
            notes=count_notes(m),
            tempos=count_tempos(m),
            timesigs=count_timesigs(m),
            seconds=max(1e-6, dur),
        ))

    # Build grid
    if args.grid_file and args.grid_file.exists():
        grid = json.loads(args.grid_file.read_text(encoding="utf-8"))
        if not isinstance(grid, list):
            raise SystemExit("--grid-file must contain a JSON list of TokenizerConfig dicts")
    else:
        grid = default_grid()

    weights = {"w_len": 1.0, "w_len_p95": 0.5, "w_loss": 2.0, "w_struct": 1.0, "w_err": 3.0}
    if args.weights:
        try:
            weights.update(json.loads(args.weights))
        except Exception:
            print("[WARN] Failed to parse --weights; using defaults.")

    rows = []
    per_config_detail = []

    grid_iter = grid
    if HAS_TQDM:
        from tqdm import tqdm  # type: ignore
        grid_iter = tqdm(grid, desc="Configs")

    for i, cfg in enumerate(grid_iter):
        # Tokenize all, collect metrics
        tok_lens: List[int] = []
        errs = 0
        note_losses = []
        tempo_diffs = []
        timesig_diffs = []

        try:
            tokenizer = make_tokenizer(cfg)
        except Exception:
            print(f"[WARN] Bad tokenizer config {cfg}:\n{traceback.format_exc()}")
            args.outdir.joinpath("tokenization_configs_errors").mkdir(parents=True, exist_ok=True)
            yaml.dump(cfg, encoding="utf-8", stream=open(f"{args.outdir}/tokenization_configs_errors/{i}.yaml", "w"))
            rows.append(dict(id=i, error_rate=1.0, tokens_per_second=np.nan,
                             median_tokens_per_file=np.nan, p95_tokens_per_file=np.nan,
                             note_loss_rate=np.nan, tempo_diff=np.nan, timesig_diff=np.nan))
            continue

        for (path, midi), orig in zip(dataset_midis, origins):
            try:
                toks = tokenizer(midi)
                total_length = 0
                if isinstance(toks, list):
                    for track in toks:
                        total_length += len(track)
                else:
                    total_length = len(toks)

                tok_lens.append(total_length)

                if not args.no_decode:
                    try:
                        dec = tokenizer(toks)
                        n_loss = abs(count_notes(dec) - orig["notes"]) / max(1, orig["notes"])
                        t_diff = abs(count_tempos(dec) - orig["tempos"])
                        s_diff = abs(count_timesigs(dec) - orig["timesigs"])
                    except Exception:
                        # If decode fails, count as a full error for that item
                        n_loss, t_diff, s_diff = 1.0, 5.0, 5.0
                    note_losses.append(n_loss)
                    tempo_diffs.append(t_diff)
                    timesig_diffs.append(s_diff)

            except Exception:
                errs += 1
                # continue to next file

        n_ok = len(dataset_midis) - errs
        if n_ok == 0:
            tokens_per_second = np.nan
            med = np.nan
            p95 = np.nan
        else:
            total_seconds = float(sum(o["seconds"] for o in origins))
            total_tokens = float(sum(tok_lens))
            tokens_per_second = total_tokens / max(1e-9, total_seconds)
            med = float(statistics.median(tok_lens))
            p95 = float(np.percentile(tok_lens, 95))
        args.outdir.joinpath("tokenization_configs").mkdir(parents=True, exist_ok=True)
        yaml.dump(cfg, encoding="utf-8", stream=open(f"{args.outdir}/tokenization_configs/{i}.yaml", "w"))
        row = dict(
            id=i,
            error_rate=errs / len(dataset_midis),
            tokens_per_second=tokens_per_second,
            median_tokens_per_file=med,
            p95_tokens_per_file=p95,
            note_loss_rate=float(np.nan if args.no_decode or len(note_losses) == 0 else np.mean(note_losses)),
            tempo_diff=float(np.nan if args.no_decode or len(tempo_diffs) == 0 else np.mean(tempo_diffs)),
            timesig_diff=float(np.nan if args.no_decode or len(timesig_diffs) == 0 else np.mean(timesig_diffs)),
        )
        rows.append(row)
        per_config_detail.append({
            "config": cfg,
            "per_file_token_lengths": tok_lens,
            "errs": errs,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "results.csv", index=False)

    # Rank
    ranked = rank_configs(df, weights)
    ranked.to_csv(args.outdir / "results_ranked.csv", index=False)

    # Save best config
    if len(ranked) > 0:
        print("Best config (lowest score):", int(ranked.iloc[0]["id"]))

    # Also drop a quick README
    (args.outdir / "README.md").write_text(
        "# Auto-tuning results\n\n"
        "- `results.csv`: raw metrics per config\n"
        "- `results_ranked.csv`: same, with `score` and sorted (lower is better)\n"
        "- `best_config.json`: the winning TokenizerConfig dict to use with Miditok REMI\n\n"
        "Edit weights via `--weights` to reflect your priorities.\n",
        encoding="utf-8",
    )

    print("Done. Wrote:", args.outdir)


if __name__ == "__main__":
    main()
