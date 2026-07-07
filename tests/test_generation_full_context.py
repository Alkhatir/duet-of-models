from __future__ import annotations

import pytest


def test_max_context_prompt_leaves_one_token_to_generate() -> None:
    omegaconf = pytest.importorskip("omegaconf")
    generate_and_score = pytest.importorskip("src.evaluation.generate_and_score")
    OmegaConf = omegaconf.OmegaConf
    compute_prompt_length = generate_and_score.compute_prompt_length
    cfg = OmegaConf.create({"prompt_mode": "max_context", "max_context_tokens": 1024})

    assert compute_prompt_length(list(range(20)), cfg) == 19


def test_max_context_prompt_uses_full_context_for_long_sequence() -> None:
    omegaconf = pytest.importorskip("omegaconf")
    generate_and_score = pytest.importorskip("src.evaluation.generate_and_score")
    OmegaConf = omegaconf.OmegaConf
    compute_prompt_length = generate_and_score.compute_prompt_length
    cfg = OmegaConf.create({"prompt_mode": "max_context", "max_context_tokens": 1024})

    assert compute_prompt_length(list(range(2000)), cfg) == 1024


def test_finite_or_none_rejects_nan() -> None:
    score_generated_midis = pytest.importorskip("src.evaluation.score_generated_midis")

    assert score_generated_midis.finite_or_none(float("nan")) is None
