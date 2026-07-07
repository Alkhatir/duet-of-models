from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("transformers")
pytest.importorskip("wandb")

from src.models.Transformer.train import (
    build_datasets,
    compute_token_metrics,
    compute_torch_token_metric_counts,
)


def test_transformer_token_metrics_match_causal_lm_shift() -> None:
    labels = np.array([[10, 11, 12, 13]])
    predictions = np.array([[11, 12, 98, 99]])
    top5_predictions = np.array(
        [
            [
                [11, 1, 2, 3, 4],
                [12, 1, 2, 3, 4],
                [98, 13, 2, 3, 4],
                [99, 1, 2, 3, 4],
            ]
        ]
    )
    confidences = np.array([[0.1, 0.7, 0.8, 0.9]], dtype=np.float32)

    metrics = compute_token_metrics(
        ((predictions, top5_predictions, confidences), labels)
    )

    assert metrics["valid_tokens"] == 3
    assert metrics["token_accuracy"] == pytest.approx(2 / 3)
    assert metrics["top5_token_accuracy"] == pytest.approx(1.0)
    assert metrics["mean_token_confidence"] == pytest.approx(
        float(np.mean([0.1, 0.7, 0.8]))
    )


def test_transformer_token_metrics_ignore_shifted_padding_labels() -> None:
    labels = np.array([[10, 11, -100, 13]])
    predictions = np.array([[11, 99, 99, 99]])
    top5_predictions = np.expand_dims(predictions, axis=-1)
    confidences = np.array([[0.2, 0.3, 0.4, 0.5]], dtype=np.float32)

    metrics = compute_token_metrics(
        ((predictions, top5_predictions, confidences), labels)
    )

    assert metrics["valid_tokens"] == 2
    assert metrics["token_accuracy"] == pytest.approx(0.5)


def test_torch_train_metric_counts_match_causal_lm_shift() -> None:
    labels = torch.tensor([[10, 11, -100, 13]])
    logits = torch.zeros((1, 4, 20))
    logits[0, 0, 11] = 10.0
    logits[0, 1, 2] = 10.0
    logits[0, 1, 13] = 9.0
    logits[0, 2, 3] = 10.0
    logits[0, 2, 13] = 9.0

    counts = compute_torch_token_metric_counts(logits, labels)

    assert counts["valid_tokens"] == 2
    assert counts["correct"] == 1
    assert counts["top5_correct"] == 2


def test_transformer_build_datasets_uses_unshifted_collator_with_test_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    train_list = tmp_path / "train.txt"
    val_list = tmp_path / "val.txt"
    test_list = tmp_path / "test.txt"
    for split_list in (train_list, val_list, test_list):
        split_list.write_text("dummy.mid\n", encoding="utf-8")

    def fake_load_midi_paths_from_list(path):
        return [path.with_suffix(".mid")]

    def fake_chunk_split(paths, tokenizer, save_dir, max_seq_len):
        return [tmp_path / f"{Path(save_dir).name}.json"]

    class FakeDatasetMIDI:
        def __init__(self, files_paths, **kwargs):
            self.files_paths = files_paths
            self.kwargs = kwargs

    class FakeDataCollator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    import src.models.Transformer.train as transformer_train

    monkeypatch.setattr(
        transformer_train,
        "load_midi_paths_from_list",
        fake_load_midi_paths_from_list,
    )
    monkeypatch.setattr(transformer_train, "chunk_split", fake_chunk_split)
    monkeypatch.setattr(
        transformer_train,
        "split_cache_dir",
        lambda paths, max_seq_len, name: tmp_path / name,
    )

    import miditok.pytorch_data

    monkeypatch.setattr(miditok.pytorch_data, "DatasetMIDI", FakeDatasetMIDI)
    monkeypatch.setattr(miditok.pytorch_data, "DataCollator", FakeDataCollator)

    class FakeTokenizer:
        pad_token_id = 0

        def __getitem__(self, key):
            assert key == "EOS_None"
            return 2

    tokenizer = FakeTokenizer()
    cfg = OmegaConf.create({"data": {"block_size": 1024}})

    train_ds, val_ds, test_ds, collator = build_datasets(
        cfg,
        tokenizer=tokenizer,
        train_list_path=train_list,
        val_list_path=val_list,
        test_list_path=test_list,
    )

    assert train_ds is not None
    assert val_ds is not None
    assert test_ds is not None
    assert collator.kwargs["copy_inputs_as_labels"] is True
    assert collator.kwargs["shift_labels"] is False
