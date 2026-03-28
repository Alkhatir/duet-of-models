from __future__ import annotations

import json
import sys
from pathlib import Path

from miditoolkit import Instrument, MidiFile, Note

from src.data.create_dataset_splits import main as split_main


def _write_midi(
    path: Path,
    instruments: list[tuple[int, bool]],
    *,
    transpose: int = 0,
    notes_per_track: int = 64,
) -> None:
    midi = MidiFile()
    for index, (program, is_drum) in enumerate(instruments):
        instrument = Instrument(program=program, is_drum=is_drum, name=f"track_{index}")
        start = 0
        for note_idx in range(notes_per_track):
            pitch = 36 if is_drum else 60 + transpose + (note_idx % 7)
            instrument.notes.append(
                Note(velocity=80, pitch=pitch, start=start, end=start + 100)
            )
            start += 140
        midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(path))


def _run_split(sampled_list: Path, out_dir: Path, seed: int = 42, rare_threshold: float = 0.2) -> dict:
    sys.argv = [
        "create_dataset_splits",
        "--input-list",
        str(sampled_list),
        "--out-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--rare-threshold",
        str(rare_threshold),
        "--manifest-cache",
        str(out_dir / "split_cache.jsonl"),
    ]
    split_main()
    return json.loads((out_dir / "split_report.json").read_text(encoding="utf-8"))


def test_grouped_split_has_no_signature_leakage_and_covers_rare_test(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    sampled_list = tmp_path / "sampled.txt"
    out_dir = tmp_path / "splits"

    paths = [
        raw / "piano_a.mid",
        raw / "piano_b_dup.mid",
        raw / "piano_c.mid",
        raw / "piano_d.mid",
        raw / "piano_e.mid",
        raw / "piano_f.mid",
        raw / "drums_only.mid",
    ]
    _write_midi(paths[0], [(0, False)])
    _write_midi(paths[1], [(0, False)])
    _write_midi(paths[2], [(0, False)], transpose=2)
    _write_midi(paths[3], [(0, False)], transpose=4)
    _write_midi(paths[4], [(0, False)], transpose=5)
    _write_midi(paths[5], [(0, False)], transpose=7)
    _write_midi(paths[6], [(0, True)])

    sampled_list.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")

    report = _run_split(sampled_list, out_dir, seed=11, rare_threshold=0.2)

    assert (out_dir / "train.txt").exists()
    assert (out_dir / "val.txt").exists()
    assert (out_dir / "test.txt").exists()
    assert report["signature_overlap"] == {
        "train_test": [],
        "train_val": [],
        "val_test": [],
    }
    assert report["rare_label_test_coverage"]["drums"] is True
    assert report["splits"]["test"]["files"] >= 1


def test_grouped_split_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    sampled_list = tmp_path / "sampled.txt"
    out_a = tmp_path / "splits_a"
    out_b = tmp_path / "splits_b"

    paths = []
    for idx, program in enumerate((0, 24, 40, 56, 64, 72, 80, 88, 96, 104)):
        path = raw / f"midi_{idx}.mid"
        _write_midi(path, [(program, False)])
        paths.append(path)

    sampled_list.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")

    _run_split(sampled_list, out_a, seed=42)
    _run_split(sampled_list, out_b, seed=42)

    assert (out_a / "train.txt").read_text(encoding="utf-8") == (
        out_b / "train.txt"
    ).read_text(encoding="utf-8")
    assert (out_a / "val.txt").read_text(encoding="utf-8") == (
        out_b / "val.txt"
    ).read_text(encoding="utf-8")
    assert (out_a / "test.txt").read_text(encoding="utf-8") == (
        out_b / "test.txt"
    ).read_text(encoding="utf-8")
