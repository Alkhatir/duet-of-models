import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from dacite import from_dict
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.tokenizer import MidiTokBuilder
from src.utils.midi_utils import (
    chunk_split,
    load_midi_paths_from_list,
    split_cache_dir,
)
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

LEGACY_LR_FIELDS = {"min_lr", "max_lr", "warmup_steps", "decay_until_step"}


def load_resolved_config(model_cfg_path: str, train_cfg_path: str) -> DictConfig:
    train_cfg = OmegaConf.load(train_cfg_path)
    model_cfg = OmegaConf.load(model_cfg_path)
    if "model" not in model_cfg:
        model_cfg = OmegaConf.create({"model": model_cfg})
    cfg = OmegaConf.merge(train_cfg, model_cfg)
    OmegaConf.resolve(cfg)
    return cfg


def apply_run_paths(cfg: DictConfig, output_dir: Path) -> DictConfig:
    cfg.output_dir = str(output_dir)
    cfg.run_name = output_dir.name
    return cfg


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for xLSTM training, but no CUDA device is available."
        )
    return torch.device(device_name)


def prepare_runtime_config(cfg: DictConfig, tokenizer_vocab_size: int) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.model.vocab_size = tokenizer_vocab_size
    cfg.model.context_length = int(cfg.data.block_size)
    return cfg


def build_dataloaders(
    cfg: DictConfig,
    tokenizer,
    train_list_path: Path,
    val_list_path: Path,
    test_list_path: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    train_paths = load_midi_paths_from_list(train_list_path)
    val_paths = load_midi_paths_from_list(val_list_path)
    test_paths = (
        load_midi_paths_from_list(test_list_path)
        if test_list_path is not None
        else None
    )

    train_chunks = chunk_split(
        train_paths,
        tokenizer,
        str(split_cache_dir(train_paths, int(cfg.data.block_size), "train")),
        int(cfg.data.block_size),
    )
    val_chunks = chunk_split(
        val_paths,
        tokenizer,
        str(split_cache_dir(val_paths, int(cfg.data.block_size), "val")),
        int(cfg.data.block_size),
    )
    test_chunks = (
        chunk_split(
            test_paths,
            tokenizer,
            str(split_cache_dir(test_paths, int(cfg.data.block_size), "test")),
            int(cfg.data.block_size),
        )
        if test_paths is not None
        else None
    )

    from miditok.pytorch_data import DataCollator, DatasetMIDI

    common = {
        "tokenizer": tokenizer,
        "max_seq_len": int(cfg.data.block_size),
        "bos_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer["EOS_None"],
    }
    train_ds = DatasetMIDI(files_paths=train_chunks, **common)
    val_ds = DatasetMIDI(files_paths=val_chunks, **common)
    test_ds = (
        DatasetMIDI(files_paths=test_chunks, **common)
        if test_chunks is not None
        else None
    )
    collator = DataCollator(
        pad_token_id=tokenizer.pad_token_id,
        copy_inputs_as_labels=True,
        shift_labels=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.per_device_train_batch_size),
        shuffle=True,
        collate_fn=collator,
    )
    eval_batch_size = int(
        cfg.train.get(
            "per_device_eval_batch_size", cfg.train.per_device_train_batch_size
        )
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader


def build_model(cfg: DictConfig, device: torch.device) -> xLSTMLMModel:
    model = xLSTMLMModel(
        from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model, resolve=True))
    ).to(device=device)
    model.reset_parameters()
    return model


def build_optimizer(model: xLSTMLMModel, cfg: DictConfig) -> optim.Optimizer:
    optim_groups = model._create_weight_decay_optim_groups()

    return optim.AdamW(
        (
            {
                "weight_decay": float(cfg.train.get("weight_decay", 0.0)),
                "params": optim_groups[0],
            },
            {"weight_decay": 0.0, "params": optim_groups[1]},
        ),
        lr=float(cfg.train.learning_rate),
    )


def _positive_float(value: Any, field_name: str, phase_index: int) -> float:
    try:
        lr = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"train.lr_schedule[{phase_index}].{field_name} must be a positive number."
        ) from exc
    if lr <= 0:
        raise ValueError(
            f"train.lr_schedule[{phase_index}].{field_name} must be positive."
        )
    return lr


def validate_lr_schedule(
    schedule_cfg: Any,
) -> list[dict[str, float | int | str | None]]:
    schedule = OmegaConf.to_container(schedule_cfg, resolve=True)
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("train.lr_schedule must be a non-empty list of phases.")

    phases: list[dict[str, float | int | str | None]] = []
    previous_end = 0
    for index, phase in enumerate(schedule):
        if not isinstance(phase, dict):
            raise ValueError(f"train.lr_schedule[{index}] must be a mapping.")

        phase_type = str(phase.get("type", "")).lower()
        if phase_type not in {"linear", "cosine", "constant"}:
            raise ValueError(
                f"train.lr_schedule[{index}].type must be one of: "
                "linear, cosine, constant."
            )

        raw_end_step = phase.get("end_step")
        is_final = index == len(schedule) - 1
        if raw_end_step is None:
            end_step = None
            if not is_final:
                raise ValueError(
                    f"train.lr_schedule[{index}].end_step may be null only for "
                    "the final phase."
                )
            if phase_type != "constant":
                raise ValueError(
                    f"train.lr_schedule[{index}].end_step is required for "
                    f"{phase_type} phases."
                )
        else:
            try:
                end_step = int(raw_end_step)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"train.lr_schedule[{index}].end_step must be an integer or null."
                ) from exc
            if end_step <= previous_end:
                raise ValueError(
                    "train.lr_schedule end_step values must be strictly increasing."
                )

        if phase_type == "constant":
            if "lr" not in phase:
                raise ValueError(f"train.lr_schedule[{index}].lr is required.")
            if end_step is None and not is_final:
                raise ValueError(
                    f"train.lr_schedule[{index}].end_step is required for "
                    "non-final constant phases."
                )
            lr = _positive_float(phase["lr"], "lr", index)
            start_lr = lr
            end_lr = lr
        else:
            if "start_lr" not in phase:
                raise ValueError(f"train.lr_schedule[{index}].start_lr is required.")
            if "end_lr" not in phase:
                raise ValueError(f"train.lr_schedule[{index}].end_lr is required.")
            start_lr = _positive_float(phase["start_lr"], "start_lr", index)
            end_lr = _positive_float(phase["end_lr"], "end_lr", index)
            if phase_type == "cosine" and start_lr < end_lr:
                raise ValueError(
                    f"train.lr_schedule[{index}] cosine phases require "
                    "start_lr >= end_lr."
                )

        phases.append(
            {
                "type": phase_type,
                "start_step": previous_end,
                "end_step": end_step,
                "start_lr": start_lr,
                "end_lr": end_lr,
            }
        )
        if end_step is not None:
            previous_end = end_step

    return phases


def _has_legacy_lr_fields(cfg: DictConfig) -> bool:
    return any(field in cfg.train for field in LEGACY_LR_FIELDS)


def _set_scheduler_base_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        param_group["initial_lr"] = lr


def _build_linear_scheduler(
    optimizer: optim.Optimizer,
    start_lr: float,
    end_lr: float,
    steps: int,
) -> optim.lr_scheduler.LinearLR:
    base_lr = max(start_lr, end_lr)
    _set_scheduler_base_lr(optimizer, base_lr)
    return optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_lr / base_lr,
        end_factor=end_lr / base_lr,
        total_iters=steps,
    )


def _build_constant_scheduler(
    optimizer: optim.Optimizer,
    lr: float,
    steps: int = 1_000_000_000,
) -> optim.lr_scheduler.ConstantLR:
    _set_scheduler_base_lr(optimizer, lr)
    return optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=steps,
    )


def _build_explicit_scheduler(
    optimizer: optim.Optimizer,
    phases: list[dict[str, float | int | str | None]],
) -> optim.lr_scheduler.LRScheduler:
    schedulers: list[optim.lr_scheduler.LRScheduler] = []
    milestones: list[int] = []
    first_scheduler_base_lr: float | None = None

    for index, phase in enumerate(phases):
        phase_type = str(phase["type"])
        start_step = int(phase["start_step"])
        end_step = phase["end_step"]
        start_lr = float(phase["start_lr"])
        end_lr = float(phase["end_lr"])
        steps = (
            int(end_step) - start_step
            if end_step is not None
            else 1_000_000_000
        )

        if phase_type == "linear":
            scheduler_base_lr = max(start_lr, end_lr)
            scheduler = _build_linear_scheduler(optimizer, start_lr, end_lr, steps)
        elif phase_type == "cosine":
            scheduler_base_lr = start_lr
            _set_scheduler_base_lr(optimizer, start_lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=end_lr,
            )
        else:
            scheduler_base_lr = start_lr
            scheduler = _build_constant_scheduler(optimizer, start_lr, steps)

        if first_scheduler_base_lr is None:
            first_scheduler_base_lr = scheduler_base_lr
        schedulers.append(scheduler)
        if end_step is not None and index < len(phases) - 1:
            milestones.append(int(end_step))

    final_phase = phases[-1]
    final_end_step = final_phase["end_step"]
    if final_end_step is not None and str(final_phase["type"]) != "constant":
        milestones.append(int(final_end_step))
        schedulers.append(
            _build_constant_scheduler(optimizer, float(final_phase["end_lr"]))
        )

    if len(schedulers) == 1:
        return schedulers[0]

    _set_scheduler_base_lr(optimizer, float(first_scheduler_base_lr))
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones,
    )


def build_scheduler(optimizer: optim.Optimizer, cfg: DictConfig):
    if "lr_schedule" in cfg.train:
        legacy_fields = sorted(
            field for field in LEGACY_LR_FIELDS if field in cfg.train
        )
        if legacy_fields:
            warnings.warn(
                "train.lr_schedule is set, so legacy LR fields are ignored: "
                + ", ".join(legacy_fields),
                UserWarning,
                stacklevel=2,
            )
        return _build_explicit_scheduler(
            optimizer,
            validate_lr_schedule(cfg.train.lr_schedule),
        )

    if not _has_legacy_lr_fields(cfg):
        return None

    warnings.warn(
        "Legacy xLSTM LR fields are deprecated and preserve the current warmup/cosine "
        "behavior temporarily. Migrate to train.lr_schedule for explicit LR behavior.",
        UserWarning,
        stacklevel=2,
    )

    max_lr = float(cfg.train.get("max_lr", cfg.train.learning_rate))
    min_lr = float(cfg.train.get("min_lr", 0.0))
    warmup_steps = int(cfg.train.get("warmup_steps", 0))
    decay_until_step = int(cfg.train.get("decay_until_step", 0))

    for param_group in optimizer.param_groups:
        param_group["lr"] = max_lr

    schedulers: list[optim.lr_scheduler.LRScheduler] = []
    milestones: list[int] = []

    if warmup_steps > 0:
        start_factor = min_lr / max_lr if max_lr > 0 else 1.0
        start_factor = max(start_factor, 1e-8)
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        schedulers.append(warmup_scheduler)

    cosine_steps = decay_until_step - warmup_steps
    if cosine_steps > 0:
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=min_lr,
        )
        schedulers.append(cosine_scheduler)
        if warmup_steps > 0:
            milestones.append(warmup_steps)

    if not schedulers:
        return None
    if len(schedulers) == 1:
        return schedulers[0]
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones,
    )


def make_autocast_context(cfg: DictConfig, device: torch.device):
    enabled = bool(cfg.train.get("enable_mixed_precision", device.type == "cuda"))
    precision_name = str(cfg.train.get("amp_precision", "bfloat16"))
    precision = torch_dtype_map.get(precision_name, torch.bfloat16)
    return torch.autocast(
        device_type=device.type,
        dtype=precision,
        enabled=enabled and device.type != "cpu",
    )


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )


def compute_batch_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    valid_mask = labels != -100
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return {
            "token_accuracy": 0.0,
            "top5_token_accuracy": 0.0,
            "mean_token_confidence": 0.0,
            "valid_tokens": 0.0,
        }

    masked_logits = logits[valid_mask]
    masked_labels = labels[valid_mask]
    probs = torch.softmax(masked_logits, dim=-1)
    predictions = masked_logits.argmax(dim=-1)
    topk = min(5, masked_logits.size(-1))
    topk_predictions = masked_logits.topk(topk, dim=-1).indices

    token_accuracy = (predictions == masked_labels).float().mean().item()
    top5_accuracy = (
        (topk_predictions == masked_labels.unsqueeze(-1))
        .any(dim=-1)
        .float()
        .mean()
        .item()
    )
    mean_confidence = probs.gather(1, predictions.unsqueeze(-1)).mean().item()

    return {
        "token_accuracy": float(token_accuracy),
        "top5_token_accuracy": float(top5_accuracy),
        "mean_token_confidence": float(mean_confidence),
        "valid_tokens": float(valid_count),
    }


def should_log_to_wandb(cfg: DictConfig) -> bool:
    report_to = cfg.train.get("report_to", ["none"])
    if isinstance(report_to, str):
        report_targets = [report_to]
    else:
        report_targets = list(report_to)
    return "wandb" in {str(target).lower() for target in report_targets}


def init_wandb_run(
    cfg: DictConfig,
    train_list_path: Path,
    val_list_path: Path,
    test_list_path: Path | None,
    tok_cfg: str,
):
    if not should_log_to_wandb(cfg):
        return None

    import wandb

    run_config = OmegaConf.to_container(cfg, resolve=True)
    run_config["train_list"] = str(train_list_path)
    run_config["val_list"] = str(val_list_path)
    run_config["test_list"] = str(test_list_path) if test_list_path else None
    run_config["tokenizer_config_path"] = tok_cfg
    run = wandb.init(
        project=str(cfg.get("wandb_project", "duet-of-models-xlstm")),
        name=cfg.get("run_name", None),
        config=run_config,
        reinit="finish_previous",
    )
    run.define_metric("step")
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="step")
    run.define_metric("val/*", step_metric="step")
    run.define_metric("test/*", step_metric="step")
    run.define_metric("best/*", step_metric="step")
    run.define_metric("final/*", step_metric="step")
    run.define_metric("progress/*", step_metric="step")
    run.summary["output_dir"] = str(cfg.output_dir)
    run.summary["run_name"] = str(cfg.get("run_name", ""))
    run.summary["train_list"] = str(train_list_path)
    run.summary["val_list"] = str(val_list_path)
    run.summary["test_list"] = str(test_list_path) if test_list_path else None
    run.summary["tokenizer_config_path"] = tok_cfg
    return run


def evaluate(
    model: xLSTMLMModel, loader: DataLoader, cfg: DictConfig, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_valid_tokens = 0.0
    total_token_accuracy = 0.0
    total_top5_accuracy = 0.0
    total_mean_confidence = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with make_autocast_context(cfg, device):
                logits = model(input_ids)
                loss = compute_loss(logits, labels)
            batch_metrics = compute_batch_metrics(logits, labels)
            valid_tokens = batch_metrics["valid_tokens"]
            total_loss += float(loss.item()) * valid_tokens
            total_valid_tokens += valid_tokens
            total_token_accuracy += batch_metrics["token_accuracy"] * valid_tokens
            total_top5_accuracy += batch_metrics["top5_token_accuracy"] * valid_tokens
            total_mean_confidence += (
                batch_metrics["mean_token_confidence"] * valid_tokens
            )

    denom = max(total_valid_tokens, 1.0)
    avg_loss = total_loss / denom
    return {
        "loss": avg_loss,
        "perplexity": math.exp(min(20.0, avg_loss)),
        "token_accuracy": total_token_accuracy / denom,
        "top5_token_accuracy": total_top5_accuracy / denom,
        "mean_token_confidence": total_mean_confidence / denom,
        "valid_tokens": total_valid_tokens,
    }


def save_artifacts(
    model: xLSTMLMModel,
    optimizer: optim.Optimizer,
    cfg: DictConfig,
    output_dir: Path,
    step: int,
    metrics: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    OmegaConf.save(config=cfg, f=str(output_dir / "config.resolved.yaml"))
    if metrics is not None:
        write_metrics(output_dir, metrics)


def write_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)


def load_model_checkpoint(
    model: xLSTMLMModel,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def train(
    cfg: DictConfig,
    train_list_path: Path,
    val_list_path: Path,
    tok_cfg: str,
    test_list_path: Path | None = None,
) -> dict[str, Any]:
    seed_value = int(cfg.get("seed", 42))
    torch.manual_seed(seed_value)

    tokenizer = MidiTokBuilder.from_yaml(tok_cfg).to_MidiTok()
    cfg = prepare_runtime_config(cfg, tokenizer_vocab_size=len(tokenizer))
    device = resolve_device(str(cfg.train.get("device", "cuda")))

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        tokenizer,
        train_list_path,
        val_list_path,
        test_list_path,
    )
    model = build_model(cfg, device)
    weight_precision = str(cfg.train.get("weight_precision", "float32"))
    model = model.to(dtype=torch_dtype_map[weight_precision])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    wandb_run = init_wandb_run(
        cfg,
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        test_list_path=test_list_path,
        tok_cfg=tok_cfg,
    )

    output_dir = Path(str(cfg.output_dir))
    logging_steps = max(1, int(cfg.train.get("logging_steps", 20)))
    eval_steps = max(1, int(cfg.train.get("eval_steps", 100)))
    save_steps = max(1, int(cfg.train.get("save_steps", eval_steps)))
    max_grad_norm = float(cfg.train.get("max_grad_norm", 0.0))
    epochs = int(cfg.train.get("num_train_epochs", 1))
    steps_per_epoch = len(train_loader)
    total_train_steps = steps_per_epoch * epochs
    val_steps = len(val_loader)
    test_steps = len(test_loader) if test_loader is not None else 0

    if wandb_run is not None:
        wandb_run.summary["model/num_parameters"] = sum(
            param.numel() for param in model.parameters()
        )
        wandb_run.summary["model/trainable_parameters"] = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        wandb_run.summary["data/train_batches_per_epoch"] = steps_per_epoch
        wandb_run.summary["data/val_batches"] = val_steps
        wandb_run.summary["data/test_batches"] = test_steps
        wandb_run.summary["data/train_batch_size"] = int(
            cfg.train.per_device_train_batch_size
        )
        wandb_run.summary["data/eval_batch_size"] = int(
            cfg.train.get(
                "per_device_eval_batch_size", cfg.train.per_device_train_batch_size
            )
        )
        wandb_run.summary["data/block_size"] = int(cfg.data.block_size)
        wandb_run.summary["train/total_steps"] = total_train_steps

    global_step = 0
    running_loss = 0.0
    best_val_loss = float("inf")
    best_val_metrics: dict[str, float] | None = None
    train_bar = tqdm(range(epochs), desc="Epochs")
    for epoch in train_bar:
        batch_bar = tqdm(train_loader, desc=f"Train {epoch + 1}/{epochs}", leave=False)
        for batch in batch_bar:
            model.train()
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with make_autocast_context(cfg, device):
                logits = model(input_ids)
                loss = compute_loss(logits, labels)

            loss.backward()
            grad_norm = None
            if max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            running_loss += float(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            epoch_progress = global_step / max(steps_per_epoch, 1)

            if global_step % logging_steps == 0:
                avg_train_loss = running_loss / logging_steps
                train_log = {
                    "step": global_step,
                    "epoch": epoch_progress,
                    "train/loss": avg_train_loss,
                    "train/perplexity": math.exp(min(20.0, avg_train_loss)),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/batch_loss": float(loss.item()),
                    "train/valid_tokens": float((labels != -100).sum().item()),
                    "progress/epoch": epoch_progress,
                    "progress/percent": (
                        100.0 * global_step / max(total_train_steps, 1)
                    ),
                }
                if grad_norm is not None:
                    train_log["train/grad_norm"] = float(grad_norm)
                """ print(
                    f"step={global_step} train_loss={train_log['train/loss']:.4f} "
                    f"train_perplexity={train_log['train/perplexity']:.4f} "
                    f"lr={train_log['train/lr']:.6g}"
                ) """
                if wandb_run is not None:
                    wandb_run.log(train_log, step=global_step)
                running_loss = 0.0

            if global_step % eval_steps == 0:
                val_metrics = evaluate(model, val_loader, cfg, device)
                val_log = {
                    "step": global_step,
                    "epoch": epoch_progress,
                    "val/loss": val_metrics["loss"],
                    "val/perplexity": val_metrics["perplexity"],
                    "val/token_accuracy": val_metrics["token_accuracy"],
                    "val/top5_token_accuracy": val_metrics["top5_token_accuracy"],
                    "val/mean_token_confidence": val_metrics["mean_token_confidence"],
                    "val/valid_tokens": val_metrics["valid_tokens"],
                }
                """ print(
                    f"step={global_step} val_loss={val_log['val/loss']:.4f} "
                    f"val_perplexity={val_log['val/perplexity']:.4f} "
                    f"val_token_accuracy={val_log['val/token_accuracy']:.4f} "
                    f"val_top5={val_log['val/top5_token_accuracy']:.4f}"
                ) """
                if wandb_run is not None:
                    wandb_run.log(val_log, step=global_step)
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_val_metrics = val_metrics
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "step": global_step,
                                "epoch": epoch_progress,
                                "best/val_loss": val_metrics["loss"],
                                "best/val_perplexity": val_metrics["perplexity"],
                                "best/val_token_accuracy": val_metrics[
                                    "token_accuracy"
                                ],
                                "best/val_top5_token_accuracy": val_metrics[
                                    "top5_token_accuracy"
                                ],
                            },
                            step=global_step,
                        )
                        wandb_run.summary["best/global_step"] = global_step
                        wandb_run.summary["best/val_loss"] = val_metrics["loss"]
                        wandb_run.summary["best/val_perplexity"] = val_metrics[
                            "perplexity"
                        ]
                    save_artifacts(
                        model,
                        optimizer,
                        cfg,
                        output_dir / "best",
                        global_step,
                        {
                            "global_step": global_step,
                            "val_loss": val_metrics["loss"],
                            "val_perplexity": val_metrics["perplexity"],
                            "val_token_accuracy": val_metrics["token_accuracy"],
                            "val_top5_token_accuracy": val_metrics[
                                "top5_token_accuracy"
                            ],
                            "val_mean_token_confidence": val_metrics[
                                "mean_token_confidence"
                            ],
                        },
                    )

            if global_step % save_steps == 0:
                save_artifacts(
                    model,
                    optimizer,
                    cfg,
                    output_dir / f"step-{global_step}",
                    global_step,
                )

    val_metrics = evaluate(model, val_loader, cfg, device)
    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        best_val_metrics = val_metrics
        save_artifacts(
            model,
            optimizer,
            cfg,
            output_dir / "best",
            global_step,
            {
                "global_step": global_step,
                "val_loss": val_metrics["loss"],
                "val_perplexity": val_metrics["perplexity"],
                "val_token_accuracy": val_metrics["token_accuracy"],
                "val_top5_token_accuracy": val_metrics["top5_token_accuracy"],
                "val_mean_token_confidence": val_metrics["mean_token_confidence"],
            },
        )

    final_metrics = {
        "global_step": global_step,
        "final_val_loss": val_metrics["loss"],
        "final_val_perplexity": val_metrics["perplexity"],
        "final_val_token_accuracy": val_metrics["token_accuracy"],
        "final_val_top5_token_accuracy": val_metrics["top5_token_accuracy"],
        "final_val_mean_token_confidence": val_metrics["mean_token_confidence"],
        "best_val_loss": best_val_loss,
        "best_val_perplexity": (
            best_val_metrics["perplexity"]
            if best_val_metrics is not None
            else val_metrics["perplexity"]
        ),
        "best_val_token_accuracy": (
            best_val_metrics["token_accuracy"]
            if best_val_metrics is not None
            else val_metrics["token_accuracy"]
        ),
        "best_val_top5_token_accuracy": (
            best_val_metrics["top5_token_accuracy"]
            if best_val_metrics is not None
            else val_metrics["top5_token_accuracy"]
        ),
        "best_val_mean_token_confidence": (
            best_val_metrics["mean_token_confidence"]
            if best_val_metrics is not None
            else val_metrics["mean_token_confidence"]
        ),
        "test_evaluated": False,
        "test_checkpoint_path": None,
    }

    best_checkpoint_path = output_dir / "best" / "checkpoint.pt"
    save_artifacts(model, optimizer, cfg, output_dir, global_step, final_metrics)

    test_metrics: dict[str, float] | None = None
    if test_loader is not None:
        load_model_checkpoint(model, best_checkpoint_path, device)
        test_metrics = evaluate(model, test_loader, cfg, device)

    if test_metrics is not None:
        final_metrics["test_evaluated"] = True
        final_metrics["test_checkpoint_path"] = str(best_checkpoint_path)
        final_metrics.update(
            {
                "test_loss": test_metrics["loss"],
                "test_perplexity": test_metrics["perplexity"],
                "test_token_accuracy": test_metrics["token_accuracy"],
                "test_top5_token_accuracy": test_metrics["top5_token_accuracy"],
                "test_mean_token_confidence": test_metrics["mean_token_confidence"],
                "test_valid_tokens": test_metrics["valid_tokens"],
            }
        )
        write_metrics(output_dir, final_metrics)
    if wandb_run is not None:
        final_log = {
            "step": global_step,
            "epoch": float(epochs),
            "final/val_loss": val_metrics["loss"],
            "final/val_perplexity": val_metrics["perplexity"],
            "final/val_token_accuracy": val_metrics["token_accuracy"],
            "final/val_top5_token_accuracy": val_metrics["top5_token_accuracy"],
            "best/val_loss": final_metrics["best_val_loss"],
            "best/val_perplexity": final_metrics["best_val_perplexity"],
        }
        if test_metrics is not None:
            final_log.update(
                {
                    "test/loss": test_metrics["loss"],
                    "test/perplexity": test_metrics["perplexity"],
                    "test/token_accuracy": test_metrics["token_accuracy"],
                    "test/top5_token_accuracy": test_metrics["top5_token_accuracy"],
                    "test/mean_token_confidence": test_metrics[
                        "mean_token_confidence"
                    ],
                    "test/valid_tokens": test_metrics["valid_tokens"],
                }
            )
        wandb_run.log(final_log, step=global_step)
        wandb_run.summary["final/global_step"] = global_step
        wandb_run.summary["final/val_loss"] = val_metrics["loss"]
        wandb_run.summary["final/val_perplexity"] = val_metrics["perplexity"]
        wandb_run.summary["final/val_token_accuracy"] = val_metrics["token_accuracy"]
        wandb_run.summary["final/checkpoint_path"] = str(output_dir / "checkpoint.pt")
        wandb_run.summary["best/checkpoint_path"] = str(
            output_dir / "best" / "checkpoint.pt"
        )
        if test_metrics is not None:
            wandb_run.summary["test/loss"] = test_metrics["loss"]
            wandb_run.summary["test/perplexity"] = test_metrics["perplexity"]
            wandb_run.summary["test/token_accuracy"] = test_metrics["token_accuracy"]
            wandb_run.summary["test/checkpoint_path"] = str(best_checkpoint_path)
        wandb_run.summary["metrics_path"] = str(output_dir / "metrics.json")
        wandb_run.finish()
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to xLSTM model YAML config.")
    parser.add_argument(
        "--train_cfg",
        default="configs/train/base.yaml",
        help="Path to shared training YAML config.",
    )
    parser.add_argument(
        "--tok_cfg",
        required=True,
        help="Path to tokenizer YAML config.",
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
        default=None,
        help="Optional path to the file containing test MIDI paths.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where checkpoints, resolved config, and metrics are written.",
    )
    args = parser.parse_args()

    cfg = load_resolved_config(args.cfg, args.train_cfg)
    cfg = apply_run_paths(cfg, Path(args.output_dir))
    metrics = train(
        cfg,
        train_list_path=Path(args.train_list),
        val_list_path=Path(args.val_list),
        test_list_path=Path(args.test_list) if args.test_list else None,
        tok_cfg=args.tok_cfg,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
