#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from miditoolkit import MidiFile
from tqdm import tqdm


GENRE_PREFIX = "genre:"
INSTRUMENT_PREFIX = "instrument:"
DRUMS_LABEL = f"{INSTRUMENT_PREFIX}drums"
DEFAULT_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
CSV_FIXED_COLUMNS = {"id", "midi_path"}


@dataclass(frozen=True)
class MidiAnalysis:
    path: str
    mtime: int
    size: int
    valid: bool
    top_instruments: list[str]
    error_type: str | None


@dataclass(frozen=True)
class MidiRow:
    music_id: str
    midi_path: Path
    genre_labels: tuple[str, ...]
    instrument_labels: tuple[str, ...]

    @property
    def all_labels(self) -> tuple[str, ...]:
        return self.genre_labels + self.instrument_labels


@dataclass(frozen=True)
class StratItem:
    key: str
    size: int
    label_counts: dict[str, int]


@dataclass
class GroupSelection:
    music_id: str
    rows: list[MidiRow]

    @property
    def size(self) -> int:
        return len(self.rows)

    @property
    def label_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        genre_labels: set[str] = set()
        for row in self.rows:
            genre_labels.update(row.genre_labels)
            counts.update(row.instrument_labels)
        counts.update(genre_labels)
        return dict(counts)


def resolve_jobs(jobs: int) -> int:
    if jobs > 0:
        return jobs
    cpu = os.cpu_count() or 1
    return max(4, cpu - 2)


def normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return {name: value / total for name, value in ratios.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample MIDI rows from genre_dist.csv and create leakage-safe train/val/test splits."
    )
    parser.add_argument(
        "--genre-csv",
        type=Path,
        default=Path("data_reports/genre_dist.csv"),
        help="CSV file containing id, midi_path, and boolean genre columns.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        required=True,
        help="Target number of MIDI rows to include in the sampled subset.",
    )
    parser.add_argument(
        "--subset-list-out",
        type=Path,
        default=Path("data_reports/sampled_subset.txt"),
        help="Output text file for the sampled MIDI subset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_reports/splits"),
        help="Directory where train/val/test split lists and report are written.",
    )
    parser.add_argument(
        "--manifest-cache",
        type=Path,
        default=Path("data_reports/splits_cache.jsonl"),
        help="Optional JSONL cache for MIDI top-instrument analysis.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic assignment seed.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS["train"],
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS["val"],
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS["test"],
        help="Test split ratio.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Number of worker processes for MIDI analysis (0 = auto).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=64,
        help="Task chunksize for the process pool.",
    )
    parser.add_argument(
        "--genres",
        nargs="+",
        default=None,
        help="Optional list of genre columns to use for filtering, stratification, and reporting.",
    )
    return parser


def hash_key(seed: int, *parts: str) -> str:
    return sha256(":".join((str(seed), *parts)).encode("utf8")).hexdigest()


def read_genre_columns(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: '{csv_path}'.")
        missing = CSV_FIXED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns {sorted(missing)}.")
        return [name for name in reader.fieldnames if name not in CSV_FIXED_COLUMNS]


def select_genre_columns(available_genres: list[str], requested_genres: list[str] | None) -> list[str]:
    if not requested_genres:
        return available_genres
    requested = []
    seen: set[str] = set()
    for genre in requested_genres:
        if genre in seen:
            continue
        seen.add(genre)
        requested.append(genre)
    missing = sorted(set(requested) - set(available_genres))
    if missing:
        raise ValueError(
            f"Requested genres are not present in the CSV header: {missing}. "
            f"Available genres: {sorted(available_genres)}"
        )
    return requested


def parse_bool_cell(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"1", "1.0", "true", "t", "yes"}


def resolve_csv_midi_path(raw_path: str, csv_path: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    cwd_resolved = (Path.cwd() / candidate).resolve()
    if cwd_resolved.is_file():
        return cwd_resolved
    csv_relative = (csv_path.parent / candidate).resolve()
    if csv_relative.is_file():
        return csv_relative
    return cwd_resolved


def analyze_midi_instruments(path_str: str) -> MidiAnalysis:
    path = Path(path_str)
    stat = path.stat()
    try:
        midi = MidiFile(str(path))
    except Exception:
        return MidiAnalysis(
            path=str(path),
            mtime=stat.st_mtime_ns,
            size=stat.st_size,
            valid=False,
            top_instruments=[],
            error_type="parse_error",
        )

    counts: Counter[str] = Counter()
    for instrument in midi.instruments:
        note_count = len(instrument.notes)
        if note_count <= 0:
            continue
        if instrument.is_drum:
            counts[DRUMS_LABEL] += note_count
        else:
            counts[f"{INSTRUMENT_PREFIX}prog_{int(instrument.program):03d}"] += note_count

    if not counts:
        return MidiAnalysis(
            path=str(path),
            mtime=stat.st_mtime_ns,
            size=stat.st_size,
            valid=False,
            top_instruments=[],
            error_type="empty",
        )

    top_instruments = [
        label
        for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:4]
    ]
    return MidiAnalysis(
        path=str(path),
        mtime=stat.st_mtime_ns,
        size=stat.st_size,
        valid=True,
        top_instruments=top_instruments,
        error_type=None,
    )


def load_analysis_cache(path: Path) -> dict[str, MidiAnalysis]:
    if not path.exists():
        return {}
    cached: dict[str, MidiAnalysis] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            required = {"path", "mtime", "size", "valid", "top_instruments", "error_type"}
            if not required.issubset(payload):
                continue
            cached[payload["path"]] = MidiAnalysis(
                path=str(payload["path"]),
                mtime=int(payload["mtime"]),
                size=int(payload["size"]),
                valid=bool(payload["valid"]),
                top_instruments=list(payload["top_instruments"]),
                error_type=payload["error_type"],
            )
    return cached


def write_analysis_cache(path: Path, records: Iterable[MidiAnalysis]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in sorted(records, key=lambda item: item.path):
            fh.write(json.dumps(record.__dict__, sort_keys=True) + "\n")


def analyze_paths(paths: list[Path], cache_path: Path | None, jobs: int, chunksize: int) -> dict[str, MidiAnalysis]:
    cached = load_analysis_cache(cache_path) if cache_path is not None else {}
    results: dict[str, MidiAnalysis] = {}
    to_analyze: list[str] = []

    for path in paths:
        stat = path.stat()
        cached_record = cached.get(str(path))
        if (
            cached_record is not None
            and cached_record.mtime == stat.st_mtime_ns
            and cached_record.size == stat.st_size
        ):
            results[str(path)] = cached_record
        else:
            to_analyze.append(str(path))

    if to_analyze:
        if jobs <= 1:
            iterator = (analyze_midi_instruments(path_str) for path_str in to_analyze)
            for record in tqdm(iterator, total=len(to_analyze), desc="Analyzing MIDI instruments"):
                results[record.path] = record
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                iterator = executor.map(
                    analyze_midi_instruments,
                    to_analyze,
                    chunksize=max(1, chunksize),
                )
                for record in tqdm(iterator, total=len(to_analyze), desc="Analyzing MIDI instruments"):
                    results[record.path] = record

    if cache_path is not None:
        write_analysis_cache(cache_path, results.values())
    return results


def load_candidate_rows(
    csv_path: Path,
    analysis_by_path: dict[str, MidiAnalysis],
    genre_columns: list[str],
) -> tuple[list[MidiRow], list[dict[str, str]]]:
    valid_rows: list[MidiRow] = []
    invalid_rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            assert row["id"] is not None
            assert row["midi_path"] is not None
            resolved_path = resolve_csv_midi_path(row["midi_path"], csv_path)
            analysis = analysis_by_path.get(str(resolved_path))
            if analysis is None or not analysis.valid:
                invalid_rows.append(
                    {
                        "id": row["id"],
                        "midi_path": str(resolved_path),
                        "error_type": None if analysis is None else str(analysis.error_type),
                    }
                )
                continue

            genre_labels = tuple(
                f"{GENRE_PREFIX}{column}"
                for column in genre_columns
                if row.get(column) is not None and parse_bool_cell(row[column])
            )
            if genre_columns and not genre_labels:
                invalid_rows.append(
                    {
                        "id": row["id"],
                        "midi_path": str(resolved_path),
                        "error_type": "missing_selected_genre",
                    }
                )
                continue
            valid_rows.append(
                MidiRow(
                    music_id=row["id"],
                    midi_path=resolved_path,
                    genre_labels=genre_labels,
                    instrument_labels=tuple(analysis.top_instruments),
                )
            )
    return valid_rows, invalid_rows


def make_row_item(row: MidiRow) -> StratItem:
    counts = Counter(row.all_labels)
    return StratItem(key=str(row.midi_path), size=1, label_counts=dict(counts))


def make_group_items(rows: list[MidiRow]) -> list[tuple[StratItem, GroupSelection]]:
    grouped: dict[str, list[MidiRow]] = defaultdict(list)
    for row in rows:
        grouped[row.music_id].append(row)

    items: list[tuple[StratItem, GroupSelection]] = []
    for music_id, group_rows in grouped.items():
        selection = GroupSelection(music_id=music_id, rows=sorted(group_rows, key=lambda item: str(item.midi_path)))
        items.append(
            (
                StratItem(key=music_id, size=len(group_rows), label_counts=selection.label_counts),
                selection,
            )
        )
    return items


def sort_items_for_assignment(
    items: list[StratItem],
    total_label_counts: Counter[str],
    seed: int,
) -> list[StratItem]:
    def rarity(item: StratItem) -> tuple[int, int]:
        if not item.label_counts:
            return (10**9, 0)
        return (
            min(total_label_counts[label] for label in item.label_counts),
            -sum(item.label_counts.values()),
        )

    return sorted(
        items,
        key=lambda item: (
            rarity(item)[0],
            rarity(item)[1],
            -item.size,
            hash_key(seed, item.key),
        ),
    )


def assign_items(
    items: list[StratItem],
    targets: dict[str, int],
    seed: int,
) -> dict[str, list[StratItem]]:
    assignments = {split_name: [] for split_name in targets}
    current_sizes = Counter()
    current_label_counts = {split_name: Counter() for split_name in targets}
    total_label_counts: Counter[str] = Counter()
    total_size = sum(item.size for item in items)
    for item in items:
        total_label_counts.update(item.label_counts)

    target_label_counts = {
        split_name: {
            label: total_label_counts[label] * (targets[split_name] / max(1, total_size))
            for label in total_label_counts
        }
        for split_name in targets
    }

    for item in sort_items_for_assignment(items, total_label_counts, seed):
        best_split: str | None = None
        best_score: tuple[float, float, float, str] | None = None
        split_order = sorted(targets, key=lambda split_name: hash_key(seed, item.key, split_name))
        for split_name in split_order:
            new_size = current_sizes[split_name] + item.size
            overflow = max(0, new_size - targets[split_name])
            size_cost = ((new_size - targets[split_name]) ** 2) / max(1, targets[split_name])
            label_cost = 0.0
            for label, count in item.label_counts.items():
                before = current_label_counts[split_name][label] - target_label_counts[split_name].get(label, 0.0)
                after = current_label_counts[split_name][label] + count - target_label_counts[split_name].get(label, 0.0)
                label_cost += ((after * after) - (before * before)) / max(1, total_label_counts[label])
            score = (
                (1000.0 * overflow) + size_cost + label_cost,
                current_sizes[split_name] / max(1, targets[split_name]),
                -sum(item.label_counts.values()),
                split_name,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        assert best_split is not None
        assignments[best_split].append(item)
        current_sizes[best_split] += item.size
        current_label_counts[best_split].update(item.label_counts)

    return assignments


def subset_targets(total_rows: int, subset_size: int) -> dict[str, int]:
    if subset_size <= 0:
        raise ValueError("subset_size must be > 0.")
    if subset_size > total_rows:
        raise ValueError(
            f"Requested subset_size={subset_size}, but only {total_rows} valid MIDI rows are available."
        )
    return {"selected": subset_size, "remainder": total_rows - subset_size}


def split_targets(total_rows: int, ratios: dict[str, float]) -> dict[str, int]:
    raw_targets = {split_name: total_rows * ratio for split_name, ratio in ratios.items()}
    targets = {split_name: int(math.floor(value)) for split_name, value in raw_targets.items()}
    positive_splits = [split_name for split_name, ratio in ratios.items() if ratio > 0]
    if total_rows >= len(positive_splits):
        for split_name in positive_splits:
            if targets[split_name] == 0:
                targets[split_name] = 1
    assigned = sum(targets.values())
    while assigned > total_rows:
        donor = max(
            (split_name for split_name in targets if targets[split_name] > 1),
            key=lambda split_name: (targets[split_name] - raw_targets[split_name], split_name),
            default=None,
        )
        if donor is None:
            break
        targets[donor] -= 1
        assigned -= 1
    for split_name, _ in sorted(
        raw_targets.items(),
        key=lambda item: (item[1] - math.floor(item[1]), item[0]),
        reverse=True,
    ):
        if assigned >= total_rows:
            break
        targets[split_name] += 1
        assigned += 1
    return targets


def relative_to_list(path: Path, list_path: Path) -> str:
    return os.path.relpath(path, start=list_path.parent)


def write_path_list(list_path: Path, midi_paths: list[Path]) -> None:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", encoding="utf-8") as fh:
        for midi_path in midi_paths:
            fh.write(f"{relative_to_list(midi_path, list_path)}\n")


def count_prefixed_labels(rows: list[MidiRow], prefix: str) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        for label in row.all_labels:
            if label.startswith(prefix):
                counts[label] += 1
    return dict(sorted(counts.items()))


def count_id_level_genre_labels(rows: list[MidiRow]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for grouped_rows in split_row_lists_by_music_id(rows).values():
        genre_labels: set[str] = set()
        for row in grouped_rows:
            genre_labels.update(row.genre_labels)
        counts.update(genre_labels)
    return dict(sorted(counts.items()))


def proportions_from_counts(counts: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {label: 0.0 for label in sorted(counts)}
    return {label: counts[label] / total for label in sorted(counts)}


def genre_proportion_deltas(
    baseline_counts: dict[str, int],
    baseline_total: int,
    split_counts: dict[str, int],
    split_total: int,
) -> dict[str, float]:
    all_labels = sorted(set(baseline_counts) | set(split_counts))
    if baseline_total <= 0 or split_total <= 0:
        return {label: 0.0 for label in all_labels}
    return {
        label: abs((split_counts.get(label, 0) / split_total) - (baseline_counts.get(label, 0) / baseline_total))
        for label in all_labels
    }


def split_row_lists_by_music_id(rows: list[MidiRow]) -> dict[str, list[MidiRow]]:
    grouped: dict[str, list[MidiRow]] = defaultdict(list)
    for row in rows:
        grouped[row.music_id].append(row)
    return grouped


def exact_id_overlap(split_rows: dict[str, list[MidiRow]]) -> dict[str, list[str]]:
    split_to_ids = {
        split_name: set(grouped.keys())
        for split_name, grouped in (
            (split_name, split_row_lists_by_music_id(rows))
            for split_name, rows in split_rows.items()
        )
    }
    overlaps: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left}_{right}"] = sorted(split_to_ids[left] & split_to_ids[right])
    return overlaps


def main() -> None:
    args = build_arg_parser().parse_args()
    ratios = normalize_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    jobs = resolve_jobs(args.jobs)

    genre_columns = select_genre_columns(read_genre_columns(args.genre_csv), args.genres)
    candidate_paths: list[Path] = []
    with args.genre_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            assert row["midi_path"] is not None
            resolved = resolve_csv_midi_path(row["midi_path"], args.genre_csv)
            if resolved.is_file():
                candidate_paths.append(resolved)

    candidate_paths = sorted({path for path in candidate_paths})
    analysis_by_path = analyze_paths(candidate_paths, args.manifest_cache, jobs, args.chunksize)
    valid_rows, invalid_rows = load_candidate_rows(args.genre_csv, analysis_by_path, genre_columns)
    if not valid_rows:
        raise SystemExit("No valid MIDI rows were found in the genre CSV.")

    row_items = [make_row_item(row) for row in valid_rows]
    subset_assignment = assign_items(row_items, subset_targets(len(valid_rows), args.subset_size), args.seed)
    selected_path_keys = {item.key for item in subset_assignment["selected"]}
    selected_rows = sorted(
        [row for row in valid_rows if str(row.midi_path) in selected_path_keys],
        key=lambda row: (row.music_id, str(row.midi_path)),
    )

    group_items = make_group_items(selected_rows)
    group_item_lookup = {item.key: selection for item, selection in group_items}
    split_assignment = assign_items(
        [item for item, _ in group_items],
        split_targets(len(selected_rows), ratios),
        args.seed,
    )

    split_rows: dict[str, list[MidiRow]] = {}
    for split_name, items in split_assignment.items():
        rows: list[MidiRow] = []
        for item in items:
            rows.extend(group_item_lookup[item.key].rows)
        split_rows[split_name] = sorted(rows, key=lambda row: (row.music_id, str(row.midi_path)))

    overlaps = exact_id_overlap(split_rows)
    if any(overlaps.values()):
        raise SystemExit("ID leakage detected across train/val/test splits.")

    selected_subset_paths = [row.midi_path for row in selected_rows]
    write_path_list(args.subset_list_out, selected_subset_paths)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        write_path_list(args.out_dir / f"{split_name}.txt", [row.midi_path for row in split_rows[split_name]])

    selected_grouped = split_row_lists_by_music_id(selected_rows)
    selected_genre_id_counts = count_id_level_genre_labels(selected_rows)
    selected_genre_id_proportions = proportions_from_counts(selected_genre_id_counts, len(selected_grouped))
    report = {
        "genre_csv": str(args.genre_csv.resolve()),
        "seed": args.seed,
        "genres_used": genre_columns,
        "subset_size_requested": args.subset_size,
        "subset_size_selected": len(selected_rows),
        "ratios": ratios,
        "totals": {
            "candidate_rows": len(valid_rows) + len(invalid_rows),
            "valid_rows": len(valid_rows),
            "invalid_rows": len(invalid_rows),
            "selected_rows": len(selected_rows),
            "selected_ids": len(selected_grouped),
        },
        "selected_subset": {
            "genre_id_counts": selected_genre_id_counts,
            "genre_id_proportions": selected_genre_id_proportions,
            "genre_midi_counts": count_prefixed_labels(selected_rows, GENRE_PREFIX),
            "instrument_label_counts": count_prefixed_labels(selected_rows, INSTRUMENT_PREFIX),
        },
        "splits": {},
        "id_overlap": overlaps,
        "invalid_row_counts": dict(Counter(entry["error_type"] for entry in invalid_rows)),
    }
    for split_name in ("train", "val", "test"):
        split_grouped = split_row_lists_by_music_id(split_rows[split_name])
        split_genre_id_counts = count_id_level_genre_labels(split_rows[split_name])
        split_genre_id_proportions = proportions_from_counts(split_genre_id_counts, len(split_grouped))
        deltas = genre_proportion_deltas(
            selected_genre_id_counts,
            len(selected_grouped),
            split_genre_id_counts,
            len(split_grouped),
        )
        report["splits"][split_name] = {
            "rows": len(split_rows[split_name]),
            "ids": len(split_grouped),
            "genre_id_counts": split_genre_id_counts,
            "genre_id_proportions": split_genre_id_proportions,
            "genre_id_max_abs_delta": max(deltas.values(), default=0.0),
            "genre_midi_counts": count_prefixed_labels(split_rows[split_name], GENRE_PREFIX),
            "instrument_label_counts": count_prefixed_labels(split_rows[split_name], INSTRUMENT_PREFIX),
        }

    with (args.out_dir / "split_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print(
        "[done] "
        f"candidate_rows={len(valid_rows)} selected_rows={len(selected_rows)} "
        f"selected_ids={len(selected_grouped)} "
        f"train={len(split_rows['train'])} val={len(split_rows['val'])} test={len(split_rows['test'])}"
    )


if __name__ == "__main__":
    main()
