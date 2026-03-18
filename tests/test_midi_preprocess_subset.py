from __future__ import annotations

import json
import sys
from pathlib import Path

from miditoolkit import Instrument, MidiFile, Note

from src.data.midi_preprocess import main as preprocess_main


def _write_valid_midi(path: Path, *, program: int = 0) -> None:
    midi = MidiFile()
    inst = Instrument(program=program, is_drum=False, name="track")
    start = 0
    step = 140
    dur = 100
    for i in range(60):
        inst.notes.append(Note(velocity=70, pitch=60 + (i % 8), start=start, end=start + dur))
        start += step
    midi.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(path))


def test_preprocess_respects_input_list(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "clean"
    manifest = tmp_path / "manifest.jsonl"
    list_path = tmp_path / "subset.txt"

    a = raw / "a.mid"
    b = raw / "b.mid"
    _write_valid_midi(a, program=0)
    _write_valid_midi(b, program=24)

    list_path.write_text(f"{a.resolve()}\n", encoding="utf-8")

    sys.argv = [
        "midi_preprocess",
        "--in",
        str(raw),
        "--out",
        str(out),
        "--input-list",
        str(list_path),
        "--jobs",
        "1",
        "--chunksize",
        "8",
        "--write-manifest",
        str(manifest),
        "--fail-on-too-few",
        "true",
    ]
    preprocess_main()

    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert Path(record["src_path"]) == a.resolve()
    assert Path(record["out_path"]).exists()

    written_midis = list(out.glob("*.mid"))
    assert len(written_midis) == 1
