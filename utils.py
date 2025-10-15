from miditok.utils import split_files_for_training
from miditok.pytorch_data import DatasetMIDI, DataCollator

from pathlib import Path
from typing import Iterable, List

import math
from transformers import TrainerCallback


class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" in logs:
            logs["eval_perplexity"] = math.exp(
                min(20.0, logs["eval_loss"])
            )  # clip to avoid inf
        if logs and "loss" in logs:  # training PPL (approx; noisy)
            logs["train_perplexity"] = math.exp(min(20.0, logs["loss"]))


def iter_midi_paths(root: Path) -> Iterable[Path]:
    """Yield all .mid/.midi files under a root directory recursively."""
    for ext in (".mid", ".midi"):
        yield from root.rglob(f"*{ext}")


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
    train_chunks = chunk_split(train_src, tokenizer, "cache_chunks/train", max_seq_len)
    val_chunks = chunk_split(val_src, tokenizer, "cache_chunks/val", max_seq_len)
    test_chunks = chunk_split(test_src, tokenizer, "cache_chunks/test", max_seq_len)

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
        shift_labels=True,
    )
    return train_ds, val_ds, test_ds, collator
