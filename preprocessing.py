#!/usr/bin/env python3
# lmd_prep_slim.py
# Minimal, tokenizer-friendly LMD preprocessor
#
# This version purposely avoids steps that Miditok already handles well
# (velocity binning/normalization, note/time quantization, tempo discretization),
# and focuses on light sanitation + dedupe. Use the tokenizer config to control
# resolution (beat_res), velocity bins, tempo bins, bar tokens, etc.
#
# Usage examples:
#   python lmd_prep_slim.py --in /path/to/LMD_matched --out ./LMD_CLEAN \
#       --save-config ./capture.yaml --write-manifest ./manifest.jsonl
#
#   python lmd_prep_slim.py --in ./raw --out ./clean --load-config ./capture.yaml

from __future__ import annotations
import argparse, json, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils import iter_midi_paths

import yaml
from tqdm import tqdm
from miditoolkit import MidiFile
from miditoolkit.midi.containers import TimeSignature

# ---------------------------
# Part 1: Capture config (model+tokenizer contract)
# ---------------------------


@dataclass
class ModelCaptureConfig:
    """Configuration of the capture/cleaning step.


    The goal is to explicitly record choices that affect the data distribution
    *before* tokenization, while avoiding edits better handled by the tokenizer
    itself (e.g., quantization, velocity discretization).


    Attributes
    ----------
    drum_map: str
    Drum pitch mapping scheme: 'gm_coarse' folds GM drums into coarse
    classes; 'none' retains original pitches.


    """

    drum_map: str = "gm_coarse"  # ('gm_coarse' or 'none')

    # FILTERS
    min_note_ons: int = 50
    min_bars: int = 4

    # DEDUPLICATION
    dedupe_by_signature: bool = True
    signature_include_programs: bool = True
    signature_include_durations: bool = True
    signature_grid: int = 8  # steps per beat used for signature invariance
    swing_tolerance_steps: int = 0

    # PPQ: do NOT rebase by default (tokenizer handles timing)
    ppq_target: Optional[int] = None


def save_config(cfg: ModelCaptureConfig, path: Path) -> None:
    """Serialize the capture configuration to a YAML file.


    Parameters
    ----------
    cfg : ModelCaptureConfig
    The configuration object to persist.
    path : Path
    Destination path (parent directories are created if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def load_config(path: Path) -> ModelCaptureConfig:
    """Load configuration from YAML, merged over defaults.


    The file may omit fields; any missing values fall back to the dataclass
    defaults to maintain forward/backward compatibility.


    Parameters
    ----------
    path : Path
    Path to the YAML configuration.


    Returns
    -------
    ModelCaptureConfig
    A fully-populated configuration object.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    defaults = asdict(ModelCaptureConfig())
    defaults.update(data or {})
    return ModelCaptureConfig(**defaults)


# ---------------------------
# Helpers (Cleaning)
# ---------------------------

GM_COARSE_DRUM_MAP = {
    35: 36,
    36: 36,  # kicks
    38: 38,
    40: 38,
    37: 38,
    39: 38,  # snares
    42: 42,
    44: 42,
    46: 46,  # hats
    41: 45,
    43: 45,
    45: 45,
    47: 47,
    48: 47,
    50: 47,  # toms
    49: 49,
    57: 49,
    55: 49,
    52: 49,
    51: 49,
    53: 49,  # cymbals
}


def map_drum_pitch(pitch: int, scheme: str) -> int:
    """Map a raw drum MIDI pitch to a coarse class, if enabled.


    Parameters
    ----------
    pitch : int
    Original MIDI note number for a drum hit.
    scheme : str
    Mapping scheme name: 'gm_coarse' applies `GM_COARSE_DRUM_MAP`; 'none'
    returns the original pitch unchanged.


    Returns
    -------
    int
    Mapped pitch. If a pitch is unknown under the coarse scheme, it falls
    back to 49 (crash cymbal), providing a consistent default class.
    """
    if scheme == "none":
        return pitch
    return GM_COARSE_DRUM_MAP.get(pitch, 49)  # default to crash


def clamp(v, lo, hi):
    """Clamp a value to the inclusive range [lo, hi]."""
    return lo if v < lo else hi if v > hi else v


def nearest_time_sig(
    num: int, den: int, candidates: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """Find the allowed time signature closest in bar length.


    Distance is computed in *quarter-note units per bar*, so 3/4 and 12/8 both
    measure as length 3.0 and are considered equivalent distances.


    Parameters
    ----------
    num, den : int
    Numerator and denominator of the target time signature.
    candidates : list[(int, int)]
    Allowed (numerator, denominator) pairs to select from.


    Returns
    -------
    (int, int)
    The candidate (num, den) with minimal absolute difference in bar length.
    """

    def bar_len(n, d):
        return n * (4.0 / d)  # in quarter-note units

    target = bar_len(num, den)
    best = min(candidates, key=lambda x: abs(bar_len(*x) - target))
    return best


def compute_bar_ticks(ppq: int, ts: TimeSignature) -> int:
    """Compute the number of ticks per bar given PPQ and a time signature.


    Parameters
    ----------
    ppq : int
    Pulses (ticks) per quarter note for this MIDI file.
    ts : TimeSignature
    The time signature to evaluate.


    Returns
    -------
    int
    Ticks per bar. For example, with PPQ=480 and 4/4, returns 1920.
    """
    quarter_per_beat = 4 / ts.denominator
    return int(ts.numerator * quarter_per_beat * ppq)


def remap_drums(mid: MidiFile, scheme: str) -> None:
    """Apply a drum pitch mapping scheme to all drum instruments in-place.


    If `scheme` is 'none', this is a no-op. Otherwise, each drum note's pitch is
    replaced by the mapped class (e.g., kick/sare/hat/tom/cymbal classes).
    """
    if scheme == "none":
        return
    for ins in mid.instruments:
        if ins.is_drum:
            for n in ins.notes:
                n.pitch = map_drum_pitch(n.pitch, scheme)


def filter_small_or_broken(mid: MidiFile, cfg: ModelCaptureConfig) -> bool:
    """Return True if the MIDI passes basic size/length sanity checks.


    Criteria
    --------
    1) Total note count across all instruments must be >= `min_note_ons`.
    2) Estimated number of bars must be >= `min_bars`. The bar length is
    computed from the **first** time signature only (for simplicity).


    Notes
    -----
    - `max_tick` is the latest note end across the file.
    - `bar_ticks` converts the first TS to ticks-per-bar using the file's PPQ.
    - If the file has no TS events, 4/4 at t=0 is assumed.
    """
    note_count = sum(len(i.notes) for i in mid.instruments)
    if note_count < cfg.min_note_ons:
        return False
    ts0 = (
        sorted(mid.time_signature_changes, key=lambda x: x.time)[0]
        if mid.time_signature_changes
        else TimeSignature(4, 4, 0)
    )
    bar_ticks = compute_bar_ticks(mid.ticks_per_beat, ts0)
    max_tick = max([0] + [n.end for ins in mid.instruments for n in ins.notes])
    n_bars = (max_tick // max(1, bar_ticks)) if bar_ticks > 0 else 0
    if n_bars < cfg.min_bars:
        return False
    return True


def signature_for_dedupe(mid: MidiFile, cfg: ModelCaptureConfig) -> str:
    """Compute a PPQ-independent, grid-based signature for de-duplication.


    The signature is a SHA-1 hash over sorted note tuples of the form:
    (program_or_sentinel, is_drum, pitch_mapped, onset_step, dur_step)


    where onset/duration are quantized to a grid derived from the file's PPQ:
    step = round(PPQ / signature_grid) # steps per beat


    Invariance & sensitivity
    ------------------------
    + Invariant to absolute PPQ and note ordering
    + Coalesces drum pitches via the selected drum_map
    + Timing differences smaller than ~half a step will round to the same step
    - Sensitive to transposition (pitches differ)
    - Sensitive to program numbers for non-drums if `signature_include_programs`
    - Sensitive to durations unless `signature_include_durations` is False


    Notes
    -----
    The current implementation's program handling effectively includes program
    numbers for non-drum tracks regardless of `signature_include_programs`.
    See the inline comment below if you intend to change that behavior.
    """
    # PPQ-independent grid based on "signature_grid" steps per beat.
    ppq = mid.ticks_per_beat
    step = max(1, int(round(ppq / cfg.signature_grid)))

    items: List[Tuple[int, int, int, int, int]] = (
        []
    )  # (prog_or_-1, is_drum, pitch, onset_step, dur_step)

    for ins in mid.instruments:
        prog = (
            int(ins.program)
            if (cfg.signature_include_programs and not ins.is_drum)
            else (-1 if ins.is_drum else int(ins.program))
        )
        for n in ins.notes:
            onset = int(round(n.start / step))
            dur = max(1, int(round((n.end - n.start) / step)))
            pitch = (
                n.pitch if not ins.is_drum else map_drum_pitch(n.pitch, cfg.drum_map)
            )
            items.append(
                (
                    prog,
                    1 if ins.is_drum else 0,
                    pitch,
                    onset,
                    dur if cfg.signature_include_durations else 1,
                )
            )

    items.sort()
    blob = json.dumps(items, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def trim_empty_tracks(mid: MidiFile) -> None:
    """Remove instruments that contain no notes after cleaning steps."""
    mid.instruments = [ins for ins in mid.instruments if len(ins.notes) > 0]


# ---------------------------
# Core pipeline (no quantize, no velocity edits)
# ---------------------------


def process_file(
    path: Path, out_dir: Path, cfg: ModelCaptureConfig
) -> Optional[Tuple[str, int, int]]:
    """Clean a single MIDI file and (optionally) dedupe-sign it.


    Steps
    -----
    1) Load MIDI with miditoolkit.
    2) Drop/keep controllers per config.
    3) Normalize TS; sanitize tempos.
    4) Optionally remap drum pitches or drop drum tracks entirely.
    5) Trim empty instruments.
    6) Filter out tiny/short files using note and bar thresholds.
    7) Compute a PPQ-independent grid signature for dedupe.
    8) Write cleaned MIDI to `out_dir/<stem>.mid`.


    Returns
    -------
    (signature, n_notes, ppq) if the file is kept, else None.


    Notes
    -----
    - The main loop performs dedupe *after* writing; duplicates are unlinked.
    This simplifies control flow but incurs extra I/O for duplicates.
    - PPQ is not rebased; tokenizer is expected to handle any timing grid.
    """
    try:
        mid = MidiFile(str(path))
    except Exception:
        return None

    # Do NOT rebase PPQ — tokenizer will quantize/normalize time
    # if cfg.ppq_target:  # left here as a no-op placeholder
    #     pass

    # Drum mapping (tokenizer does not collapse drum pitches)
    remap_drums(mid, cfg.drum_map)

    # No note quantization here — tokenizer handles it
    # No velocity clipping/binning here — tokenizer handles it

    trim_empty_tracks(mid)

    if not filter_small_or_broken(mid, cfg):
        return None

    sig = signature_for_dedupe(mid, cfg)

    # Write cleaned MIDI
    rel = path.stem
    out_path = out_dir / f"{rel}.mid"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mid.dump(str(out_path))
    except Exception:
        return None

    n_notes = sum(len(i.notes) for i in mid.instruments)
    return (sig, n_notes, mid.ticks_per_beat)


# ---------------------------
# CLI / Orchestration
# ---------------------------


def main():
    """Parse CLI arguments and orchestrate the batch cleaning pipeline.


    Command-line options:
    --in PATH Input root directory to search for MIDIs
    --out PATH Output directory for cleaned MIDIs
    --save-config PATH Save the effective capture config to YAML
    --load-config PATH Load capture config from YAML (overrides defaults)
    --write-manifest PATH Append per-file JSON lines with metadata
    --dedupe-index PATH Write JSON mapping signature → first kept filename
    --limit N Limit number of input files processed (0 = all)


    The pipeline:
    - Materialize the input file list (respecting --limit)
    - Process each file, collecting signature, notes, and PPQ
    - If dedupe is enabled, delete just-written duplicates
    - Optionally write a JSONL manifest and a dedupe index JSON
    - Print a final summary of processed/kept/unique counts
    """
    ap = argparse.ArgumentParser(description="LMD-matched cleaner (tokenizer-friendly)")
    ap.add_argument(
        "--in",
        dest="inp",
        required=True,
        type=Path,
        help="Input root folder (LMD matched)",
    )
    ap.add_argument(
        "--out",
        dest="out",
        required=True,
        type=Path,
        help="Output folder for cleaned MIDIs",
    )
    ap.add_argument(
        "--save-config", type=Path, help="Write the chosen capture config to YAML"
    )
    ap.add_argument(
        "--load-config", type=Path, help="Load config from YAML (overrides defaults)"
    )
    ap.add_argument(
        "--write-manifest",
        type=Path,
        help="Write JSONL manifest with per-file metadata",
    )
    ap.add_argument(
        "--dedupe-index",
        type=Path,
        help="Write JSON file of kept signatures (for reuse)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of files to process (0 = all)",
    )
    args = ap.parse_args()

    cfg = load_config(args.load_config) if args.load_config else ModelCaptureConfig()

    if args.save_config:
        save_config(cfg, args.save_config)
        print(f"[config] saved to {args.save_config}")

    args.out.mkdir(parents=True, exist_ok=True)

    seen: Dict[str, str] = {}
    kept = 0
    total = 0

    man_fp = open(args.write_manifest, "w") if args.write_manifest else None

    midi_paths = list(iter_midi_paths(args.inp))
    if args.limit and args.limit > 0:
        midi_paths = midi_paths[: args.limit]

    for p in tqdm(midi_paths, desc="Cleaning MIDIs"):
        total += 1
        res = process_file(p, args.out, cfg)
        if res is None:
            continue
        sig, n_notes, ppq = res
        if cfg.dedupe_by_signature:
            if sig in seen:
                try:
                    (args.out / f"{p.stem}.mid").unlink(missing_ok=True)
                except Exception:
                    pass
                continue
            seen[sig] = p.name

        kept += 1
        if man_fp:
            record = {
                "src_path": str(p),
                "out_path": str(args.out / f"{p.stem}.mid"),
                "notes": n_notes,
                "signature": sig,
                "ppq": ppq,  # actual PPQ (not rebased)
                "drum_map": cfg.drum_map,
            }
            man_fp.write(json.dumps(record) + "\n")

    if man_fp:
        man_fp.close()

    if args.dedupe_index:
        with open(args.dedupe_index, "w") as f:
            json.dump(seen, f, indent=2)

    print(f"[done] processed={total}, kept={kept}, deduped={len(seen)}")


if __name__ == "__main__":
    main()
