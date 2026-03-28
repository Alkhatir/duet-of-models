#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from miditoolkit import MidiFile

from src.data.midi_preprocess import (
    ModelCaptureConfig,
    remap_drums,
    signature_for_dedupe,
    trim_empty_tracks,
)
from src.data.sample_midi_subset import FAMILY_KEYS, resolve_jobs
from src.utils.midi_utils import load_midi_paths_from_list


DEFAULT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


@dataclass(frozen=True)
class MidiMetadata:
    path: str
    mtime: int
    size: int
    valid: bool
    signature: str | None
    labels: list[str]
    error_type: str | None


@dataclass
class GroupRecord:
    signature: str
    paths: list[Path]
    labels: set[str]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe train/val/test splits from a sampled MIDI list."
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        required=True,
        help="Text file with sampled MIDI paths.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_reports/splits"),
        help="Directory where train/val/test lists and report are written.",
    )
    parser.add_argument(
        "--manifest-cache",
        type=Path,
        default=None,
        help="Optional JSONL metadata cache for reruns.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic assignment seed.")
    parser.add_argument(
        "--rare-threshold",
        type=float,
        default=0.02,
        help="Families present in <= threshold fraction of groups are treated as rare.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_RATIOS["train"],
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_RATIOS["val"],
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_RATIOS["test"],
        help="Test split ratio.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Reserved for interface consistency; metadata analysis runs in-process for sampled lists.",
    )
    return parser


def analyze_midi_for_split(path: Path) -> MidiMetadata:
    stat = path.stat()
    try:
        midi = MidiFile(str(path))
    except Exception:
        return MidiMetadata(
            path=str(path),
            mtime=stat.st_mtime_ns,
            size=stat.st_size,
            valid=False,
            signature=None,
            labels=[],
            error_type="parse_error",
        )

    trim_empty_tracks(midi)
    if not midi.instruments:
        return MidiMetadata(
            path=str(path),
            mtime=stat.st_mtime_ns,
            size=stat.st_size,
            valid=False,
            signature=None,
            labels=[],
            error_type="empty",
        )

    labels: set[str] = set()
    for instrument in midi.instruments:
        if not instrument.notes:
            continue
        if instrument.is_drum:
            labels.add("drums")
        else:
            labels.add(FAMILY_KEYS[max(0, min(127, int(instrument.program))) // 8])

    if not labels:
        return MidiMetadata(
            path=str(path),
            mtime=stat.st_mtime_ns,
            size=stat.st_size,
            valid=False,
            signature=None,
            labels=[],
            error_type="empty",
        )

    # Align the leakage key with preprocessing dedupe semantics.
    dedupe_cfg = ModelCaptureConfig()
    remap_drums(midi, dedupe_cfg.drum_map)
    signature = signature_for_dedupe(midi, dedupe_cfg)

    return MidiMetadata(
        path=str(path),
        mtime=stat.st_mtime_ns,
        size=stat.st_size,
        valid=True,
        signature=signature,
        labels=sorted(labels),
        error_type=None,
    )


def load_cache(path: Path) -> dict[str, MidiMetadata]:
    if not path.exists():
        return {}
    records: dict[str, MidiMetadata] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records[payload["path"]] = MidiMetadata(**payload)
    return records


def write_cache(path: Path, records: list[MidiMetadata]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in sorted(records, key=lambda item: item.path):
            fh.write(json.dumps(record.__dict__, sort_keys=True) + "\n")


def collect_metadata(paths: list[Path], cache_path: Path | None) -> list[MidiMetadata]:
    cached = load_cache(cache_path) if cache_path else {}
    records: list[MidiMetadata] = []

    for path in paths:
        stat = path.stat()
        cache_key = str(path)
        cached_record = cached.get(cache_key)
        if (
            cached_record is not None
            and cached_record.mtime == stat.st_mtime_ns
            and cached_record.size == stat.st_size
        ):
            records.append(cached_record)
            continue
        records.append(analyze_midi_for_split(path))

    if cache_path:
        write_cache(cache_path, records)
    return records


def normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return {name: value / total for name, value in ratios.items()}


def group_records(records: list[MidiMetadata]) -> dict[str, GroupRecord]:
    grouped: dict[str, GroupRecord] = {}
    for record in records:
        if not record.valid or record.signature is None:
            continue
        group = grouped.setdefault(
            record.signature,
            GroupRecord(signature=record.signature, paths=[], labels=set()),
        )
        group.paths.append(Path(record.path))
        group.labels.update(record.labels)
    return grouped


def compute_rare_labels(groups: dict[str, GroupRecord], threshold: float) -> set[str]:
    label_counts = Counter()
    for group in groups.values():
        for label in group.labels:
            label_counts[label] += 1
    min_groups = max(1, math.ceil(len(groups) * threshold))
    return {label for label, count in label_counts.items() if count <= min_groups}


def build_group_order(
    groups: dict[str, GroupRecord],
    rare_labels: set[str],
    seed: int,
) -> list[GroupRecord]:
    rarity_counts = Counter()
    for group in groups.values():
        rarity_counts[group.signature] = sum(1 for label in group.labels if label in rare_labels)

    return sorted(
        groups.values(),
        key=lambda group: (
            -rarity_counts[group.signature],
            -len(group.labels),
            -len(group.paths),
            sha256(f"{seed}:{group.signature}".encode("utf8")).hexdigest(),
        ),
    )


def target_file_counts(groups: dict[str, GroupRecord], ratios: dict[str, float]) -> dict[str, int]:
    total_files = sum(len(group.paths) for group in groups.values())
    raw_targets = {name: total_files * ratio for name, ratio in ratios.items()}
    targets = {name: int(math.floor(value)) for name, value in raw_targets.items()}
    assigned = sum(targets.values())
    for name, _ in sorted(
        raw_targets.items(),
        key=lambda item: (item[1] - math.floor(item[1]), item[0]),
        reverse=True,
    ):
        if assigned >= total_files:
            break
        targets[name] += 1
        assigned += 1
    return targets


def target_label_counts(groups: dict[str, GroupRecord], ratios: dict[str, float]) -> dict[str, dict[str, float]]:
    total_labels = Counter()
    for group in groups.values():
        for label in group.labels:
            total_labels[label] += 1
    return {
        split: {label: total_labels[label] * ratio for label in FAMILY_KEYS}
        for split, ratio in ratios.items()
    }


def assign_groups(
    groups: dict[str, GroupRecord],
    ratios: dict[str, float],
    rare_labels: set[str],
    seed: int,
) -> dict[str, list[GroupRecord]]:
    assignments: dict[str, list[GroupRecord]] = {"train": [], "val": [], "test": []}
    current_files = Counter()
    current_labels: dict[str, Counter] = {name: Counter() for name in assignments}
    file_targets = target_file_counts(groups, ratios)
    label_targets = target_label_counts(groups, ratios)
    label_to_groups: dict[str, list[str]] = defaultdict(list)
    for group in groups.values():
        for label in group.labels:
            label_to_groups[label].append(group.signature)

    max_test_groups = max(1, round(len(groups) * ratios["test"]))
    singleton_rare_labels = {
        label for label, signatures in label_to_groups.items() if label in rare_labels and len(signatures) == 1
    }
    preassigned_test: set[str] = set()
    uncovered_singletons = set(singleton_rare_labels)
    while uncovered_singletons and len(preassigned_test) < max_test_groups:
        candidate = min(
            (
                group
                for group in groups.values()
                if group.signature not in preassigned_test and group.labels & uncovered_singletons
            ),
            key=lambda group: (
                -len(group.labels & uncovered_singletons),
                len(group.paths),
                len(group.labels),
                sha256(f"{seed}:{group.signature}".encode("utf8")).hexdigest(),
            ),
        )
        preassigned_test.add(candidate.signature)
        uncovered_singletons -= candidate.labels

    ordered_groups = build_group_order(groups, rare_labels, seed)
    uncovered_test_labels = set(rare_labels)

    for group in ordered_groups:
        if group.signature in preassigned_test:
            assignments["test"].append(group)
            current_files["test"] += len(group.paths)
            for label in group.labels:
                current_labels["test"][label] += 1
            uncovered_test_labels -= group.labels

    for group in ordered_groups:
        if group.signature in preassigned_test:
            continue
        best_split: str | None = None
        best_score: tuple[float, float, float, str] | None = None
        split_order = sorted(
            ("test", "val", "train"),
            key=lambda split_name: sha256(f"{seed}:{group.signature}:{split_name}".encode("utf8")).hexdigest(),
        )
        for split_name in split_order:
            new_file_total = current_files[split_name] + len(group.paths)
            overflow = max(0, new_file_total - file_targets[split_name])
            file_cost = ((new_file_total - file_targets[split_name]) ** 2) / max(1, file_targets[split_name])
            label_cost = 0.0
            for label in group.labels:
                new_label_total = current_labels[split_name][label] + 1
                label_cost += (
                    (new_label_total - label_targets[split_name].get(label, 0.0)) ** 2
                )

            missing_rare = len(group.labels & uncovered_test_labels)
            rare_bonus = 0.0
            if split_name == "test" and missing_rare > 0:
                rare_bonus = -5.0 * missing_rare
                if current_files["test"] < file_targets["test"]:
                    rare_bonus -= 5.0 * missing_rare
            score = ((25.0 * overflow) + file_cost + rare_bonus, label_cost, -missing_rare, split_name)
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        assert best_split is not None
        assignments[best_split].append(group)
        current_files[best_split] += len(group.paths)
        for label in group.labels:
            current_labels[best_split][label] += 1
        if best_split == "test":
            uncovered_test_labels -= group.labels

    return assignments


def expand_assignments(assignments: dict[str, list[GroupRecord]]) -> dict[str, list[Path]]:
    expanded: dict[str, list[Path]] = {}
    for split_name, groups in assignments.items():
        paths: list[Path] = []
        for group in groups:
            paths.extend(group.paths)
        expanded[split_name] = sorted(paths)
    return expanded


def exact_overlap_signatures(assignments: dict[str, list[GroupRecord]]) -> dict[str, list[str]]:
    split_to_signatures = {
        split_name: {group.signature for group in groups}
        for split_name, groups in assignments.items()
    }
    overlaps: dict[str, list[str]] = {}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for left, right in pairs:
        overlaps[f"{left}_{right}"] = sorted(split_to_signatures[left] & split_to_signatures[right])
    return overlaps


def label_presence_counts(groups: list[GroupRecord]) -> dict[str, int]:
    counts = Counter()
    for group in groups:
        for label in group.labels:
            counts[label] += 1
    return {label: counts.get(label, 0) for label in FAMILY_KEYS}


def file_label_presence(paths: list[Path], record_by_path: dict[str, MidiMetadata]) -> dict[str, int]:
    counts = Counter()
    for path in paths:
        for label in record_by_path[str(path)].labels:
            counts[label] += 1
    return {label: counts.get(label, 0) for label in FAMILY_KEYS}


def write_split_list(path: Path, midi_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for midi_path in midi_paths:
            fh.write(f"{midi_path}\n")


def main() -> None:
    args = build_arg_parser().parse_args()
    ratios = normalize_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    _ = resolve_jobs(args.jobs)

    sampled_paths = load_midi_paths_from_list(args.input_list)
    metadata = collect_metadata(sampled_paths, args.manifest_cache)
    valid_metadata = [record for record in metadata if record.valid]
    if not valid_metadata:
        raise SystemExit("No valid MIDI files were found in the sampled list.")

    groups = group_records(valid_metadata)
    if not groups:
        raise SystemExit("No leakage-safe groups could be created from the sampled list.")

    rare_labels = compute_rare_labels(groups, args.rare_threshold)
    assignments = assign_groups(groups, ratios, rare_labels, args.seed)
    expanded = expand_assignments(assignments)
    overlaps = exact_overlap_signatures(assignments)
    if any(overlaps.values()):
        raise SystemExit("Signature leakage detected across splits.")

    record_by_path = {record.path: record for record in valid_metadata}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_paths in expanded.items():
        write_split_list(args.out_dir / f"{split_name}.txt", split_paths)

    total_groups = len(groups)
    rare_coverage = {
        label: any(label in group.labels for group in assignments["test"])
        for label in sorted(rare_labels)
    }
    report = {
        "input_list": str(args.input_list.resolve()),
        "seed": args.seed,
        "ratios": ratios,
        "rare_threshold": args.rare_threshold,
        "totals": {
            "sampled_files": len(sampled_paths),
            "valid_files": len(valid_metadata),
            "groups": total_groups,
        },
        "splits": {
            split_name: {
                "files": len(expanded[split_name]),
                "groups": len(assignments[split_name]),
                "group_label_presence": label_presence_counts(assignments[split_name]),
                "file_label_presence": file_label_presence(expanded[split_name], record_by_path),
            }
            for split_name in ("train", "val", "test")
        },
        "rare_labels": sorted(rare_labels),
        "rare_label_test_coverage": rare_coverage,
        "invalid_records": [
            {"path": record.path, "error_type": record.error_type}
            for record in metadata
            if not record.valid
        ],
        "signature_overlap": overlaps,
    }

    with (args.out_dir / "split_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print(
        "[done] "
        f"files={len(valid_metadata)} groups={total_groups} "
        f"train={len(expanded['train'])} val={len(expanded['val'])} test={len(expanded['test'])}"
    )


if __name__ == "__main__":
    main()
