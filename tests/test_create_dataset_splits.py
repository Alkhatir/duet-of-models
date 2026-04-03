from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from miditoolkit import Instrument, MidiFile, Note

from src.data.create_dataset_splits import main as split_main


def _write_midi(
    path: Path,
    instruments: list[tuple[int, bool]],
    *,
    note_multiplier: int = 1,
) -> None:
    midi = MidiFile()
    for track_idx, (program, is_drum) in enumerate(instruments):
        instrument = Instrument(program=program, is_drum=is_drum, name=f"track_{track_idx}")
        start = 0
        for note_idx in range(64 * note_multiplier):
            pitch = 36 if is_drum else 60 + (note_idx % 7)
            instrument.notes.append(
                Note(velocity=80, pitch=pitch, start=start, end=start + 100)
            )
            start += 140
        midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(path))


def _write_genre_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["id", "rock", "jazz", "electronic", "midi_path"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_split(
    csv_path: Path,
    out_dir: Path,
    subset_size: int,
    seed: int = 42,
    genres: list[str] | None = None,
) -> dict:
    subset_list = out_dir.parent / "sampled_subset.txt"
    sys.argv = [
        "create_dataset_splits",
        "--genre-csv",
        str(csv_path),
        "--subset-size",
        str(subset_size),
        "--subset-list-out",
        str(subset_list),
        "--out-dir",
        str(out_dir),
        "--manifest-cache",
        str(out_dir.parent / "splits_cache.jsonl"),
        "--seed",
        str(seed),
        "--jobs",
        "1",
    ]
    if genres:
        sys.argv.extend(["--genres", *genres])
    split_main()
    return json.loads((out_dir / "split_report.json").read_text(encoding="utf-8"))


def test_create_dataset_splits_is_deterministic_and_relative(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "lmd_matched"
    csv_path = tmp_path / "data_reports" / "genre_dist.csv"
    out_a = tmp_path / "data_reports" / "splits_a"
    out_b = tmp_path / "data_reports" / "splits_b"

    rows: list[dict[str, str]] = []
    for idx in range(8):
        midi_path = raw / f"track_{idx}.mid"
        _write_midi(midi_path, [(idx * 8, False)])
        rows.append(
            {
                "id": f"TRACK{idx:03d}",
                "rock": "1.0",
                "jazz": "1.0" if idx % 3 == 0 else "0.0",
                "electronic": "1.0" if idx % 4 == 0 else "0.0",
                "midi_path": str(midi_path),
            }
        )

    _write_genre_csv(csv_path, rows)

    report_a = _run_split(csv_path, out_a, subset_size=6, seed=7)
    report_b = _run_split(csv_path, out_b, subset_size=6, seed=7)

    assert report_a["subset_size_selected"] == 6
    assert report_a["id_overlap"] == {"train_test": [], "train_val": [], "val_test": []}
    assert (out_a.parent / "sampled_subset.txt").read_text(encoding="utf-8").startswith("../data/")
    assert (out_a / "train.txt").read_text(encoding="utf-8").startswith("../../data/")
    assert (out_a.parent / "sampled_subset.txt").read_text(encoding="utf-8") == (
        out_b.parent / "sampled_subset.txt"
    ).read_text(encoding="utf-8")
    assert (out_a / "train.txt").read_text(encoding="utf-8") == (
        out_b / "train.txt"
    ).read_text(encoding="utf-8")


def test_same_music_id_never_leaks_across_splits(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "lmd_matched"
    csv_path = tmp_path / "data_reports" / "genre_dist.csv"
    out_dir = tmp_path / "data_reports" / "splits"

    first = raw / "song_a_1.mid"
    second = raw / "song_a_2.mid"
    third = raw / "song_b_1.mid"
    fourth = raw / "song_c_1.mid"
    _write_midi(first, [(0, False), (24, False)], note_multiplier=2)
    _write_midi(second, [(0, False), (40, False)], note_multiplier=1)
    _write_midi(third, [(56, False)], note_multiplier=1)
    _write_midi(fourth, [(0, True), (64, False)], note_multiplier=1)

    rows = [
        {
            "id": "SONG_A",
            "rock": "1.0",
            "jazz": "0.0",
            "electronic": "0.0",
            "midi_path": str(first),
        },
        {
            "id": "SONG_A",
            "rock": "1.0",
            "jazz": "0.0",
            "electronic": "0.0",
            "midi_path": str(second),
        },
        {
            "id": "SONG_B",
            "rock": "0.0",
            "jazz": "1.0",
            "electronic": "0.0",
            "midi_path": str(third),
        },
        {
            "id": "SONG_C",
            "rock": "0.0",
            "jazz": "0.0",
            "electronic": "1.0",
            "midi_path": str(fourth),
        },
    ]
    _write_genre_csv(csv_path, rows)

    report = _run_split(csv_path, out_dir, subset_size=4, seed=13)

    assert report["id_overlap"] == {"train_test": [], "train_val": [], "val_test": []}
    combined_lists = {
        "train": (out_dir / "train.txt").read_text(encoding="utf-8").splitlines(),
        "val": (out_dir / "val.txt").read_text(encoding="utf-8").splitlines(),
        "test": (out_dir / "test.txt").read_text(encoding="utf-8").splitlines(),
    }
    memberships = {
        split_name: {entry for entry in entries if "song_a_" in entry}
        for split_name, entries in combined_lists.items()
    }
    non_empty = [split_name for split_name, entries in memberships.items() if entries]
    assert len(non_empty) == 1


def test_selected_genres_are_filtered_and_reported_over_ids(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "lmd_matched"
    csv_path = tmp_path / "data_reports" / "genre_dist.csv"
    out_dir = tmp_path / "data_reports" / "splits"

    midi_paths = [raw / f"song_{idx}.mid" for idx in range(6)]
    for idx, midi_path in enumerate(midi_paths):
        _write_midi(midi_path, [(idx * 8, False)])

    rows = [
        {"id": "A", "rock": "1.0", "jazz": "0.0", "electronic": "0.0", "midi_path": str(midi_paths[0])},
        {"id": "A", "rock": "1.0", "jazz": "0.0", "electronic": "1.0", "midi_path": str(midi_paths[1])},
        {"id": "B", "rock": "1.0", "jazz": "1.0", "electronic": "0.0", "midi_path": str(midi_paths[2])},
        {"id": "C", "rock": "0.0", "jazz": "1.0", "electronic": "0.0", "midi_path": str(midi_paths[3])},
        {"id": "D", "rock": "0.0", "jazz": "0.0", "electronic": "1.0", "midi_path": str(midi_paths[4])},
        {"id": "E", "rock": "1.0", "jazz": "0.0", "electronic": "0.0", "midi_path": str(midi_paths[5])},
    ]
    _write_genre_csv(csv_path, rows)

    report = _run_split(csv_path, out_dir, subset_size=4, seed=9, genres=["rock", "jazz"])

    assert report["genres_used"] == ["rock", "jazz"]
    assert report["invalid_row_counts"]["missing_selected_genre"] == 1
    assert report["selected_subset"]["genre_id_counts"]["genre:rock"] >= report["selected_subset"]["genre_id_counts"]["genre:jazz"]
    assert "genre:electronic" not in report["selected_subset"]["genre_id_counts"]
    for split_name in ("train", "val", "test"):
        split_info = report["splits"][split_name]
        assert "genre_id_counts" in split_info
        assert "genre_id_proportions" in split_info
        assert "genre_id_max_abs_delta" in split_info
        assert "genre:electronic" not in split_info["genre_id_counts"]
