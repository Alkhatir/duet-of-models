#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from miditoolkit import MidiFile
from tqdm import tqdm


GM_FAMILY_NAMES: List[str] = [
    "piano",
    "chromatic_percussion",
    "organ",
    "guitar",
    "bass",
    "strings",
    "ensemble",
    "brass",
    "reed",
    "pipe",
    "synth_lead",
    "synth_pad",
    "synth_effects",
    "ethnic",
    "percussive",
    "sound_effects",
]
DRUMS_FAMILY = "drums"
FAMILY_KEYS: List[str] = GM_FAMILY_NAMES + [DRUMS_FAMILY]
FAMILY_PRIORITY: List[str] = FAMILY_KEYS[:]  # deterministic tie-break


def parse_bool(value: str) -> bool:
    val = value.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value '{value}'. Use true/false.")


def resolve_jobs(jobs: int) -> int:
    if jobs > 0:
        return jobs
    cpu = os.cpu_count() or 1
    return max(4, cpu - 2)


def normalize_exts(exts: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for ext in exts:
        e = ext.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = f".{e}"
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out


def iter_midi_paths_with_ext(root: Path, exts: List[str]) -> Iterable[Path]:
    for ext in exts:
        yield from root.rglob(f"*{ext}")


def gm_program_to_family(program: int) -> str:
    p = max(0, min(127, int(program)))
    return GM_FAMILY_NAMES[p // 8]


def primary_family_from_counts(counts: Dict[str, int]) -> str:
    max_count = max(counts.values())
    tied = {k for k, v in counts.items() if v == max_count}
    for family in FAMILY_PRIORITY:
        if family in tied:
            return family
    return FAMILY_PRIORITY[0]


def empty_counts() -> Dict[str, int]:
    return {k: 0 for k in FAMILY_KEYS}


def analyze_midi_path(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    try:
        midi = MidiFile(str(path))
    except Exception:
        return {
            "path": str(path),
            "valid": False,
            "primary_family": None,
            "family_counts": empty_counts(),
            "error_type": "parse_error",
        }

    counts = empty_counts()
    total_notes = 0
    for inst in midi.instruments:
        note_count = len(inst.notes)
        if note_count <= 0:
            continue
        total_notes += note_count
        if inst.is_drum:
            counts[DRUMS_FAMILY] += note_count
        else:
            fam = gm_program_to_family(inst.program)
            counts[fam] += note_count

    if total_notes <= 0:
        return {
            "path": str(path),
            "valid": False,
            "primary_family": None,
            "family_counts": counts,
            "error_type": "empty",
        }

    return {
        "path": str(path),
        "valid": True,
        "primary_family": primary_family_from_counts(counts),
        "family_counts": counts,
        "error_type": None,
    }


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache_path = rec.get("path")
            if not isinstance(cache_path, str):
                continue
            out[cache_path] = rec
    return out


def write_cache(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in sorted(records, key=lambda x: x["path"]):
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def allocate_quotas(capacities: Dict[str, int], n: int) -> Dict[str, int]:
    if n <= 0:
        raise ValueError("n must be > 0")
    total_capacity = sum(capacities.values())
    if total_capacity < n:
        raise ValueError("Not enough available files to satisfy requested sample size.")

    exact = {k: (n * capacities[k] / total_capacity) for k in capacities}
    quotas = {k: min(capacities[k], math.floor(exact[k])) for k in capacities}

    assigned = sum(quotas.values())
    if assigned < n:
        remainders = sorted(
            capacities.keys(),
            key=lambda k: (exact[k] - math.floor(exact[k]), capacities[k], k),
            reverse=True,
        )
        for k in remainders:
            if assigned >= n:
                break
            if quotas[k] < capacities[k]:
                quotas[k] += 1
                assigned += 1

    if assigned < n:
        spare = sorted(
            capacities.keys(),
            key=lambda k: (capacities[k] - quotas[k], capacities[k], k),
            reverse=True,
        )
        while assigned < n:
            progressed = False
            for k in spare:
                if quotas[k] < capacities[k]:
                    quotas[k] += 1
                    assigned += 1
                    progressed = True
                    if assigned >= n:
                        break
            if not progressed:
                raise ValueError("Unable to allocate quotas due to exhausted capacities.")

    return quotas


def sample_by_strata(
    strata_to_paths: Dict[str, List[str]],
    quotas: Dict[str, int],
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    selected: List[str] = []
    for family in sorted(strata_to_paths.keys()):
        candidates = sorted(strata_to_paths[family])
        rng.shuffle(candidates)
        selected.extend(candidates[: quotas.get(family, 0)])
    return sorted(selected)


def proportion(counter: Dict[str, int], total: int) -> Dict[str, float]:
    if total <= 0:
        return {k: 0.0 for k in FAMILY_KEYS}
    return {k: counter.get(k, 0) / total for k in FAMILY_KEYS}


def format_relative_path(path: str, base_dir: Path) -> str:
    return os.path.relpath(path, start=base_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Sample a representative MIDI subset by instrument-family strata."
    )
    ap.add_argument("--in", dest="inp", required=True, type=Path, help="Input MIDI root")
    ap.add_argument("--n", required=True, type=int, help="Subset size")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed")
    ap.add_argument("--out-list", required=True, type=Path, help="Output newline list")
    ap.add_argument("--report", type=Path, default=None, help="Output JSON report")
    ap.add_argument(
        "--manifest-cache",
        type=Path,
        default=None,
        help="Optional JSONL metadata cache for faster reruns",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Worker process count (0 = auto max(4, cpu-2))",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=64,
        help="Task chunksize for process pool worker map",
    )
    ap.add_argument(
        "--extensions",
        nargs="+",
        default=[".mid", ".midi"],
        help="MIDI file extensions",
    )
    ap.add_argument(
        "--fail-on-too-few",
        type=parse_bool,
        default=True,
        help="Fail if valid MIDI files are fewer than requested n (true/false)",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be > 0")

    exts = normalize_exts(args.extensions)
    all_paths = sorted({p.resolve() for p in iter_midi_paths_with_ext(args.inp, exts)})
    if not all_paths:
        raise SystemExit("No MIDI files found for provided root/extensions.")

    cache: Dict[str, Dict[str, Any]] = {}
    if args.manifest_cache:
        cache = load_cache(args.manifest_cache)

    reusable: Dict[str, Dict[str, Any]] = {}
    to_analyze: List[Path] = []
    for p in all_paths:
        p_str = str(p)
        cached = cache.get(p_str)
        stat = p.stat()
        if (
            cached
            and cached.get("mtime") == stat.st_mtime_ns
            and cached.get("size") == stat.st_size
            and "valid" in cached
            and "family_counts" in cached
            and "error_type" in cached
        ):
            reusable[p_str] = cached
        else:
            to_analyze.append(p)

    jobs = resolve_jobs(args.jobs)
    analyzed_records: Dict[str, Dict[str, Any]] = {}
    if to_analyze:
        worker_input = [str(p) for p in to_analyze]
        if jobs <= 1:
            it = (analyze_midi_path(p) for p in worker_input)
            for rec in tqdm(it, total=len(worker_input), desc="Analyzing MIDIs"):
                analyzed_records[rec["path"]] = rec
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                it = ex.map(analyze_midi_path, worker_input, chunksize=max(1, args.chunksize))
                for rec in tqdm(it, total=len(worker_input), desc="Analyzing MIDIs"):
                    analyzed_records[rec["path"]] = rec

    final_records: List[Dict[str, Any]] = []
    for p in all_paths:
        p_str = str(p)
        base = analyzed_records.get(p_str) or reusable.get(p_str)
        if base is None:
            # should not happen, but keep a stable fallback
            base = {
                "path": p_str,
                "valid": False,
                "primary_family": None,
                "family_counts": empty_counts(),
                "error_type": "missing_record",
            }
        stat = p.stat()
        final_records.append(
            {
                "path": p_str,
                "mtime": stat.st_mtime_ns,
                "size": stat.st_size,
                "valid": bool(base.get("valid", False)),
                "primary_family": base.get("primary_family"),
                "family_counts": base.get("family_counts", empty_counts()),
                "error_type": base.get("error_type"),
            }
        )

    if args.manifest_cache:
        write_cache(args.manifest_cache, final_records)

    valid_records = [r for r in final_records if r["valid"]]
    if len(valid_records) < args.n:
        msg = (
            f"Requested n={args.n} but only {len(valid_records)} valid MIDI files found."
        )
        if args.fail_on_too_few:
            raise SystemExit(msg)
        print(f"[warn] {msg} Sampling all valid files instead.")

    target_n = min(args.n, len(valid_records))

    strata_to_paths: Dict[str, List[str]] = defaultdict(list)
    full_strata_counter = Counter()
    error_counter = Counter()
    for rec in final_records:
        err = rec.get("error_type")
        if err:
            error_counter[err] += 1
        if not rec["valid"]:
            continue
        family = str(rec["primary_family"])
        strata_to_paths[family].append(rec["path"])
        full_strata_counter[family] += 1

    capacities = {k: len(v) for k, v in strata_to_paths.items()}
    quotas = allocate_quotas(capacities, target_n)
    selected_paths = sample_by_strata(strata_to_paths, quotas, args.seed)

    out_list = args.out_list
    out_list.parent.mkdir(parents=True, exist_ok=True)
    with open(out_list, "w", encoding="utf-8") as f:
        for p in selected_paths:
            f.write(f"{format_relative_path(p, out_list.parent)}\n")

    family_by_path = {str(r["path"]): str(r["primary_family"]) for r in valid_records}
    sample_counter = Counter()
    for p in selected_paths:
        sample_counter[family_by_path[p]] += 1

    full_prop = proportion(dict(full_strata_counter), sum(full_strata_counter.values()))
    sample_prop = proportion(dict(sample_counter), sum(sample_counter.values()))
    max_abs_delta = max(abs(full_prop[k] - sample_prop[k]) for k in FAMILY_KEYS)

    report = {
        "input_root": str(args.inp.resolve()),
        "n_requested": args.n,
        "n_selected": len(selected_paths),
        "seed": args.seed,
        "jobs": jobs,
        "chunksize": args.chunksize,
        "extensions": exts,
        "totals": {
            "files_discovered": len(all_paths),
            "files_valid": len(valid_records),
            "files_invalid": len(all_paths) - len(valid_records),
        },
        "errors": dict(error_counter),
        "full_distribution_files": {
            "counts": {k: int(full_strata_counter.get(k, 0)) for k in FAMILY_KEYS},
            "proportions": full_prop,
        },
        "sample_distribution_files": {
            "counts": {k: int(sample_counter.get(k, 0)) for k in FAMILY_KEYS},
            "proportions": sample_prop,
        },
        "distribution_quality": {
            "max_abs_delta": max_abs_delta,
        },
        "quotas": {k: int(v) for k, v in sorted(quotas.items())},
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    print(
        "[done] "
        f"discovered={len(all_paths)} valid={len(valid_records)} "
        f"selected={len(selected_paths)} jobs={jobs} "
        f"max_abs_delta={max_abs_delta:.4f}"
    )


if __name__ == "__main__":
    main()
