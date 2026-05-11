import pytest
import torch
from omegaconf import OmegaConf

from src.models.xlstm.train import build_scheduler


def _optimizer(lr: float = 0.001) -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    return torch.optim.AdamW([param], lr=lr)


def _step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    optimizer: torch.optim.Optimizer,
    steps: int,
) -> None:
    for _ in range(steps):
        optimizer.step()
        if scheduler is not None:
            scheduler.step()


def test_no_schedule_and_no_legacy_fields_keeps_learning_rate_constant() -> None:
    cfg = OmegaConf.create({"train": {"learning_rate": 3e-4}})
    optimizer = _optimizer(lr=float(cfg.train.learning_rate))

    scheduler = build_scheduler(optimizer, cfg)
    _step_scheduler(scheduler, optimizer, 5)

    assert scheduler is None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_explicit_linear_cosine_constant_schedule() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "learning_rate": 1.0,
                "lr_schedule": [
                    {
                        "type": "linear",
                        "end_step": 2,
                        "start_lr": 0.1,
                        "end_lr": 1.0,
                    },
                    {
                        "type": "cosine",
                        "end_step": 4,
                        "start_lr": 1.0,
                        "end_lr": 0.2,
                    },
                    {"type": "constant", "end_step": None, "lr": 0.2},
                ],
            }
        }
    )
    optimizer = _optimizer(lr=float(cfg.train.learning_rate))

    scheduler = build_scheduler(optimizer, cfg)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    _step_scheduler(scheduler, optimizer, 1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.55)

    _step_scheduler(scheduler, optimizer, 1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)

    _step_scheduler(scheduler, optimizer, 1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.6)

    _step_scheduler(scheduler, optimizer, 1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)

    _step_scheduler(scheduler, optimizer, 3)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)


def test_lr_schedule_warns_and_ignores_legacy_fields() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "learning_rate": 1.0,
                "min_lr": 0.001,
                "max_lr": 9.0,
                "warmup_steps": 10,
                "decay_until_step": 20,
                "lr_schedule": [
                    {"type": "constant", "end_step": None, "lr": 0.25},
                ],
            }
        }
    )
    optimizer = _optimizer(lr=float(cfg.train.learning_rate))

    with pytest.warns(UserWarning, match="legacy LR fields are ignored"):
        scheduler = build_scheduler(optimizer, cfg)

    _step_scheduler(scheduler, optimizer, 5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.25)


def test_legacy_only_warns_and_preserves_cosine_oscillation() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "learning_rate": 1.0,
                "max_lr": 1.0,
                "min_lr": 0.0,
                "warmup_steps": 0,
                "decay_until_step": 2,
            }
        }
    )
    optimizer = _optimizer(lr=float(cfg.train.learning_rate))

    with pytest.warns(UserWarning, match="Legacy xLSTM LR fields are deprecated"):
        scheduler = build_scheduler(optimizer, cfg)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    _step_scheduler(scheduler, optimizer, 1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    _step_scheduler(scheduler, optimizer, 1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    _step_scheduler(scheduler, optimizer, 1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        ([], "non-empty"),
        ([{"type": "exponential", "end_step": 1, "lr": 0.1}], "type"),
        ([{"type": "linear", "start_lr": 0.1, "end_lr": 0.2}], "end_step"),
        (
            [{"type": "linear", "end_step": 1, "start_lr": -0.1, "end_lr": 0.2}],
            "positive",
        ),
        (
            [
                {"type": "constant", "end_step": 2, "lr": 0.1},
                {"type": "constant", "end_step": 2, "lr": 0.1},
            ],
            "strictly increasing",
        ),
        (
            [
                {"type": "constant", "end_step": None, "lr": 0.1},
                {"type": "constant", "end_step": None, "lr": 0.1},
            ],
            "only for the final phase",
        ),
    ],
)
def test_invalid_lr_schedules_raise_clear_value_errors(
    schedule: list[dict[str, object]], message: str
) -> None:
    cfg = OmegaConf.create({"train": {"learning_rate": 1.0, "lr_schedule": schedule}})
    optimizer = _optimizer(lr=float(cfg.train.learning_rate))

    with pytest.raises(ValueError, match=message):
        build_scheduler(optimizer, cfg)
