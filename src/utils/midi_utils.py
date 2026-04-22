from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable, List

def iter_midi_paths(root: Path) -> Iterable[Path]:
    """Yield all .mid/.midi files under a root directory recursively."""
    for ext in (".mid", ".midi"):
        yield from root.rglob(f"*{ext}")


def load_midi_paths_from_list(data_list_path: Path) -> list[Path]:
    if not data_list_path.is_file():
        raise ValueError(f"Expected a text file of MIDI paths, got '{data_list_path}'.")

    midi_paths: list[Path] = []
    with data_list_path.open("r", encoding="utf8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            midi_path = Path(line).expanduser()
            if not midi_path.is_absolute():
                midi_path = (data_list_path.parent / midi_path).resolve()
            if not midi_path.is_file():
                raise ValueError(f"MIDI path from list does not exist: '{midi_path}'.")
            midi_paths.append(midi_path)

    if not midi_paths:
        raise ValueError(f"No MIDI paths were found in '{data_list_path}'.")

    return midi_paths


def load_named_split_lists(
    train_list_path: Path,
    val_list_path: Path,
    test_list_path: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    return (
        load_midi_paths_from_list(train_list_path),
        load_midi_paths_from_list(val_list_path),
        load_midi_paths_from_list(test_list_path),
    )


def split_cache_dir(paths: List[Path], max_seq_len: int, split_name: str) -> Path:
    h = sha256()
    h.update(str(max_seq_len).encode("utf8"))
    h.update(split_name.encode("utf8"))
    for path in paths:
        h.update(str(path).encode("utf8"))
    return Path("cache_chunks") / h.hexdigest()[:16] / split_name


def chunk_split(
    paths: List[Path],
    tokenizer,
    save_dir: str,
    max_seq_len: int,
    avg_tokens_per_note: float | None = None,
    num_overlap_bars: int = 1,
    min_seq_len: int | None = None,
) -> List[Path]:
    """
    Returns paths to the chunked files saved in save_dir.
    Can be called repeatedly; it's cached by a hidden hash file.
    """
    from miditok.utils import split_files_for_training

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    chunk_paths = split_files_for_training(
        files_paths=paths,
        tokenizer=tokenizer,
        save_dir=Path(save_dir),
        max_seq_len=max_seq_len,
        average_num_tokens_per_note=avg_tokens_per_note,  # None -> auto-compute on first ~200 files
        num_overlap_bars=num_overlap_bars,
        min_seq_len=min_seq_len,
    )
    return chunk_paths


def build_three_datasets_from_chunks(
    tokenizer,
    train_src: List[Path],
    val_src: List[Path],
    test_src: List[Path],
    max_seq_len: int,
) -> tuple[DatasetMIDI, DatasetMIDI, DatasetMIDI, DataCollator]:
    """
    Builds three datasets (train, validation, and test) from source data chunks and returns them
    along with a data collator for tokenized sequences.
    Args:
        tokenizer: The tokenizer to be used for tokenizing the input data.
        train_src (List[Path]): List of file paths or raw data for the training dataset.
        val_src (List[Path]): List of file paths or raw data for the validation dataset.
        test_src (List[Path]): List of file paths or raw data for the test dataset.
        max_seq_len (int): Maximum sequence length for tokenized data.
    Returns:
        tuple[DatasetMIDI, DatasetMIDI, DatasetMIDI, DataCollator]: A tuple containing:
            - train_ds (DatasetMIDI): The training dataset.
            - val_ds (DatasetMIDI): The validation dataset.
            - test_ds (DatasetMIDI): The test dataset.
            - collator (DataCollator): The data collator for padding and label shifting.
    """
    from miditok.pytorch_data import DatasetMIDI, DataCollator

    train_chunks = chunk_split(
        train_src,
        tokenizer,
        str(split_cache_dir(train_src, max_seq_len, "train")),
        max_seq_len,
    )
    val_chunks = chunk_split(
        val_src,
        tokenizer,
        str(split_cache_dir(val_src, max_seq_len, "val")),
        max_seq_len,
    )
    test_chunks = chunk_split(
        test_src,
        tokenizer,
        str(split_cache_dir(test_src, max_seq_len, "test")),
        max_seq_len,
    )

    common = {
        "tokenizer": tokenizer,
        "max_seq_len": max_seq_len,
        "bos_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer["EOS_None"],
    }
    train_ds = DatasetMIDI(files_paths=train_chunks, **common)
    val_ds = DatasetMIDI(files_paths=val_chunks, **common)
    test_ds = DatasetMIDI(files_paths=test_chunks, **common)

    collator = DataCollator(
        pad_token_id=tokenizer.pad_token_id,
        copy_inputs_as_labels=True,
        shift_labels=True,
    )
    return train_ds, val_ds, test_ds, collator
