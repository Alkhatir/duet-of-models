from __future__ import annotations

import json
import sys
from pathlib import Path

from miditoolkit import Instrument, MidiFile, Note

from src.data.sample_midi_subset import (
    allocate_quotas,
    gm_program_to_family,
    main as sample_main,
    primary_family_from_counts,
)


def _write_midi(path: Path, *, program: int = 0, is_drum: bool = False, notes: int = 64) -> None:
    midi = MidiFile()
    inst = Instrument(program=program, is_drum=is_drum, name="track")
    start = 0
    step = 140
    dur = 80
    for i in range(notes):
        pitch = 36 if is_drum else (60 + (i % 12))
        inst.notes.append(Note(velocity=80, pitch=pitch, start=start, end=start + dur))
        start += step
    midi.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(path))


def test_mapping_and_quota_allocation() -> None:
    assert gm_program_to_family(0) == "piano"
    assert gm_program_to_family(24) == "guitar"
    assert gm_program_to_family(127) == "sound_effects"

    counts = {"piano": 10, "guitar": 10, "drums": 5}
    primary = primary_family_from_counts({k: counts.get(k, 0) for k in [
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
        "drums",
    ]})
    assert primary == "piano"

    quotas = allocate_quotas({"piano": 8, "guitar": 4, "drums": 4}, 8)
    assert sum(quotas.values()) == 8
    assert quotas["piano"] >= quotas["guitar"]


def test_sampling_determinism_and_cache(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    for i in range(8):
        _write_midi(root / f"piano_{i}.mid", program=0)
    for i in range(4):
        _write_midi(root / f"guitar_{i}.mid", program=24)
    for i in range(4):
        _write_midi(root / f"drums_{i}.mid", is_drum=True)

    out1 = tmp_path / "subset_seed7_a.txt"
    out2 = tmp_path / "subset_seed7_b.txt"
    out3 = tmp_path / "subset_seed8.txt"
    report = tmp_path / "report.json"
    cache = tmp_path / "cache.jsonl"

    sys.argv = [
        "sample_midi_subset",
        "--in",
        str(root),
        "--n",
        "8",
        "--seed",
        "7",
        "--jobs",
        "1",
        "--out-list",
        str(out1),
        "--report",
        str(report),
        "--manifest-cache",
        str(cache),
    ]
    sample_main()

    sys.argv = [
        "sample_midi_subset",
        "--in",
        str(root),
        "--n",
        "8",
        "--seed",
        "7",
        "--jobs",
        "1",
        "--out-list",
        str(out2),
        "--report",
        str(report),
        "--manifest-cache",
        str(cache),
    ]
    sample_main()

    sys.argv = [
        "sample_midi_subset",
        "--in",
        str(root),
        "--n",
        "8",
        "--seed",
        "8",
        "--jobs",
        "1",
        "--out-list",
        str(out3),
        "--manifest-cache",
        str(cache),
    ]
    sample_main()

    a = out1.read_text(encoding="utf-8").splitlines()
    b = out2.read_text(encoding="utf-8").splitlines()
    c = out3.read_text(encoding="utf-8").splitlines()

    assert a == b
    assert a != c
    assert cache.exists()

    report_data = json.loads(report.read_text(encoding="utf-8"))
    assert report_data["n_selected"] == 8
    assert report_data["distribution_quality"]["max_abs_delta"] <= 0.15
