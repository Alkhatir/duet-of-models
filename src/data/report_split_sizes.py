import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.data.tokenizer import MidiTokBuilder
from src.utils.midi_utils import (
    chunk_split,
    load_named_split_lists,
    split_cache_dir,
)


def count_split(
    paths: list[Path], tokenizer, max_seq_len: int, split_name: str
) -> dict[str, int | str]:
    chunk_dir = split_cache_dir(paths, max_seq_len, split_name)
    chunk_paths = chunk_split(
        paths=paths,
        tokenizer=tokenizer,
        save_dir=str(chunk_dir),
        max_seq_len=max_seq_len,
    )
    return {
        "midi_files": len(paths),
        "chunks": len(chunk_paths),
        "chunk_cache_dir": str(chunk_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report raw MIDI-file counts and chunked dataset sizes for train/val/test splits."
    )
    parser.add_argument(
        "--tok_cfg",
        required=True,
        help="Path to tokenizer YAML config.",
    )
    parser.add_argument(
        "--train_cfg",
        default="configs/train/base.yaml",
        help="Path to training YAML config. Used for data.block_size unless overridden.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Optional override for max sequence length / block size.",
    )
    parser.add_argument(
        "--train_list",
        required=True,
        help="Path to the file containing training MIDI paths.",
    )
    parser.add_argument(
        "--val_list",
        required=True,
        help="Path to the file containing validation MIDI paths.",
    )
    parser.add_argument(
        "--test_list",
        required=True,
        help="Path to the file containing test MIDI paths.",
    )
    args = parser.parse_args()

    train_cfg = OmegaConf.load(args.train_cfg)
    block_size = int(train_cfg.data.block_size)
    tokenizer = MidiTokBuilder.from_yaml(args.tok_cfg).to_MidiTok()

    train_paths, val_paths, test_paths = load_named_split_lists(
        Path(args.train_list),
        Path(args.val_list),
        Path(args.test_list),
    )

    report = {
        "block_size": block_size,
        "tokenizer_vocab_size": len(tokenizer),
        "train": count_split(train_paths, tokenizer, block_size, "train"),
        "val": count_split(val_paths, tokenizer, block_size, "val"),
        "test": count_split(test_paths, tokenizer, block_size, "test"),
    }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
