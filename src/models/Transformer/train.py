from __future__ import annotations

import argparse
import dataclasses
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    LlamaConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.data.tokenizer import MidiTokBuilder
from src.utils.midi_utils import (
    chunk_split,
    load_midi_paths_from_list,
    split_cache_dir,
)

LEGACY_TRAINER_SCHEDULER_FIELDS = {"warmup_steps", "lr_scheduler_type"}


class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" in logs:
            logs["eval_perplexity"] = math.exp(min(20.0, logs["eval_loss"]))
        if logs and "loss" in logs:
            logs["train_perplexity"] = math.exp(min(20.0, logs["loss"]))


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
    cfg.run_name = str(cfg.get("run_name", output_dir.name))
    return cfg


def prepare_runtime_config(cfg: DictConfig, tokenizer_vocab_size: int) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.model.vocab_size = tokenizer_vocab_size
    cfg.model.max_position_embeddings = int(cfg.data.block_size)
    return cfg


def _model_value(model_cfg: DictConfig, name: str, default: Any | None = None) -> Any:
    if name in model_cfg:
        return model_cfg[name]
    if default is not None:
        return default
    raise ValueError(f"model.{name} is required.")


def build_model(cfg: DictConfig) -> torch.nn.Module:
    model_cfg = cfg.model
    architecture = str(model_cfg.get("architecture", "llama")).lower()
    if architecture in {"llama", "llama2", "llama-2"}:
        config = LlamaConfig(
            vocab_size=int(_model_value(model_cfg, "vocab_size")),
            hidden_size=int(_model_value(model_cfg, "hidden_size")),
            intermediate_size=int(_model_value(model_cfg, "intermediate_size")),
            num_attention_heads=int(_model_value(model_cfg, "num_attention_heads")),
            num_hidden_layers=int(_model_value(model_cfg, "num_hidden_layers")),
            max_position_embeddings=int(
                _model_value(model_cfg, "max_position_embeddings")
            ),
            rope_theta=float(model_cfg.get("rope_theta", 1e6)),
            bos_token_id=model_cfg.get("bos_token_id", None),
            eos_token_id=model_cfg.get("eos_token_id", None),
            pad_token_id=model_cfg.get("pad_token_id", None),
        )
    elif architecture == "gpt2":
        config = GPT2Config(
            vocab_size=int(_model_value(model_cfg, "vocab_size")),
            n_embd=int(_model_value(model_cfg, "hidden_size")),
            n_layer=int(_model_value(model_cfg, "num_hidden_layers")),
            n_head=int(_model_value(model_cfg, "num_attention_heads")),
            n_positions=int(_model_value(model_cfg, "max_position_embeddings")),
            n_ctx=int(_model_value(model_cfg, "max_position_embeddings")),
            bos_token_id=model_cfg.get("bos_token_id", None),
            eos_token_id=model_cfg.get("eos_token_id", None),
            pad_token_id=model_cfg.get("pad_token_id", None),
        )
    else:
        raise ValueError(
            f"Unknown transformer architecture '{architecture}'. "
            "Expected one of: gpt2, llama."
        )
    return AutoModelForCausalLM.from_config(config)


def build_datasets(
    cfg: DictConfig,
    tokenizer,
    train_list_path: Path,
    val_list_path: Path,
    test_list_path: Path | None = None,
):
    from miditok.pytorch_data import DataCollator, DatasetMIDI

    max_seq_len = int(cfg.data.block_size)
    train_midis = load_midi_paths_from_list(train_list_path)
    val_midis = load_midi_paths_from_list(val_list_path)
    test_midis = (
        load_midi_paths_from_list(test_list_path)
        if test_list_path is not None
        else None
    )
    train_chunks = chunk_split(
        train_midis,
        tokenizer,
        str(split_cache_dir(train_midis, max_seq_len, "train")),
        max_seq_len,
    )
    val_chunks = chunk_split(
        val_midis,
        tokenizer,
        str(split_cache_dir(val_midis, max_seq_len, "val")),
        max_seq_len,
    )
    test_chunks = (
        chunk_split(
            test_midis,
            tokenizer,
            str(split_cache_dir(test_midis, max_seq_len, "test")),
            max_seq_len,
        )
        if test_midis is not None
        else None
    )
    common = {
        "tokenizer": tokenizer,
        "max_seq_len": max_seq_len,
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
        shift_labels=False,
    )
    return train_ds, val_ds, test_ds, collator


def _normalise_training_arg_names(raw: dict[str, Any]) -> dict[str, Any]:
    fields = {field.name for field in dataclasses.fields(TrainingArguments)}
    normalised = dict(raw)
    if "eval_strategy" in fields and "evaluation_strategy" in normalised:
        normalised["eval_strategy"] = normalised.pop("evaluation_strategy")
    if "evaluation_strategy" in fields and "eval_strategy" in normalised:
        normalised["evaluation_strategy"] = normalised.pop("eval_strategy")
    return normalised


def filter_training_args(raw: dict[str, Any], output_dir: str) -> TrainingArguments:
    fields = {field.name for field in dataclasses.fields(TrainingArguments)}
    raw = _normalise_training_arg_names(raw)
    kwargs = {
        key: value
        for key, value in raw.items()
        if key in fields and key not in {"lr_schedule"}
    }
    kwargs["output_dir"] = output_dir
    kwargs.setdefault("report_to", ["none"])
    kwargs.setdefault("run_name", Path(output_dir).name)
    kwargs.setdefault("remove_unused_columns", False)
    eval_strategy = kwargs.get(
        "eval_strategy", kwargs.get("evaluation_strategy", "no")
    )
    if str(eval_strategy).lower() != "no":
        kwargs.setdefault("save_strategy", eval_strategy)
        kwargs.setdefault("load_best_model_at_end", True)
        kwargs.setdefault("metric_for_best_model", "eval_loss")
        kwargs.setdefault("greater_is_better", False)
    return TrainingArguments(**kwargs)


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

    run_config = OmegaConf.to_container(cfg, resolve=True)
    run_config["train_list"] = str(train_list_path)
    run_config["val_list"] = str(val_list_path)
    run_config["test_list"] = str(test_list_path) if test_list_path else None
    run_config["tokenizer_config_path"] = tok_cfg
    return wandb.init(
        project=str(cfg.get("wandb_project", "duet-of-models-transformer")),
        name=cfg.get("run_name", None),
        config=run_config,
        reinit="finish_previous",
        settings=wandb.Settings(start_method="fork"),
    )


def _add_perplexity(metrics: dict[str, float], loss_key: str) -> dict[str, float]:
    if loss_key in metrics:
        metrics[f"{loss_key.removesuffix('_loss')}_perplexity"] = math.exp(
            min(20.0, metrics[loss_key])
        )
    return metrics


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


def build_optimizer(model: torch.nn.Module, cfg: DictConfig) -> optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower() or "ln_" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": float(cfg.train.get("weight_decay", 0.0)),
            },
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(cfg.train.learning_rate),
    )


def build_scheduler(optimizer: optim.Optimizer, cfg: DictConfig):
    if "lr_schedule" not in cfg.train:
        return None

    ignored_fields = sorted(
        field for field in LEGACY_TRAINER_SCHEDULER_FIELDS if field in cfg.train
    )
    if ignored_fields:
        warnings.warn(
            "train.lr_schedule is set, so Trainer scheduler fields are ignored: "
            + ", ".join(ignored_fields),
            UserWarning,
            stacklevel=2,
        )
    return _build_explicit_scheduler(
        optimizer,
        validate_lr_schedule(cfg.train.lr_schedule),
    )


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = torch.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    topk = min(5, logits.size(-1))
    top5_predictions = logits.topk(topk, dim=-1).indices
    return predictions, top5_predictions, confidences


def compute_token_metrics(eval_pred) -> dict[str, float]:
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        token_predictions, top5_predictions, confidences = predictions
    else:
        token_predictions = predictions
        top5_predictions = np.expand_dims(predictions, axis=-1)
        confidences = np.ones_like(predictions, dtype=np.float32)

    token_predictions = np.asarray(token_predictions)
    top5_predictions = np.asarray(top5_predictions)
    confidences = np.asarray(confidences)
    labels = np.asarray(labels)

    if labels.shape[-1] < 2:
        return {
            "token_accuracy": 0.0,
            "top5_token_accuracy": 0.0,
            "mean_token_confidence": 0.0,
            "valid_tokens": 0.0,
        }

    # Hugging Face causal LM loss compares logits[:, :-1] to labels[:, 1:].
    token_predictions = token_predictions[..., :-1]
    top5_predictions = top5_predictions[..., :-1, :]
    confidences = confidences[..., :-1]
    labels = labels[..., 1:]

    valid_mask = labels != -100
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return {
            "token_accuracy": 0.0,
            "top5_token_accuracy": 0.0,
            "mean_token_confidence": 0.0,
            "valid_tokens": 0.0,
        }

    valid_predictions = token_predictions[valid_mask]
    valid_labels = labels[valid_mask]
    valid_top5 = top5_predictions[valid_mask]
    valid_confidences = confidences[valid_mask]

    return {
        "token_accuracy": float(np.mean(valid_predictions == valid_labels)),
        "top5_token_accuracy": float(
            np.mean(np.any(valid_top5 == np.expand_dims(valid_labels, axis=-1), axis=-1))
        ),
        "mean_token_confidence": float(np.mean(valid_confidences)),
        "valid_tokens": float(valid_count),
    }


def train(
    cfg: DictConfig,
    train_list_path: Path,
    val_list_path: Path,
    tok_cfg: str,
    test_list_path: Path | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(cfg.get("seed", 42)))

    tokenizer = MidiTokBuilder.from_yaml(tok_cfg).to_MidiTok()
    cfg = prepare_runtime_config(cfg, tokenizer_vocab_size=len(tokenizer))
    cfg.model.pad_token_id = tokenizer.pad_token_id
    cfg.model.eos_token_id = tokenizer["EOS_None"]

    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(output_dir / "config.resolved.yaml"))

    model = build_model(cfg)
    train_ds, val_ds, test_ds, collator = build_datasets(
        cfg,
        tokenizer=tokenizer,
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        test_list_path=test_list_path,
    )
    training_args = filter_training_args(
        OmegaConf.to_container(cfg.train, resolve=True),
        str(output_dir),
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    wandb_run = init_wandb_run(
        cfg,
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        test_list_path=test_list_path,
        tok_cfg=tok_cfg,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_token_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[PerplexityCallback()],
    )
    train_result = trainer.train()
    best_checkpoint_path = trainer.state.best_model_checkpoint
    trainer.save_model(str(output_dir))

    final_metrics: dict[str, Any] = {
        key: float(value)
        for key, value in train_result.metrics.items()
        if isinstance(value, int | float)
    }
    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    final_metrics.update(
        {
            key: float(value)
            for key, value in _add_perplexity(val_metrics, "val_loss").items()
            if isinstance(value, int | float)
        }
    )

    if test_ds is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
        final_metrics.update(
            {
                key: float(value)
                for key, value in _add_perplexity(test_metrics, "test_loss").items()
                if isinstance(value, int | float)
            }
        )
        final_metrics["test_evaluated"] = True
        final_metrics["test_checkpoint_path"] = str(
            best_checkpoint_path or output_dir
        )
    else:
        final_metrics["test_evaluated"] = False

    with (output_dir / "metrics.json").open("w", encoding="utf8") as fh:
        json.dump(final_metrics, fh, indent=2, sort_keys=True)

    if wandb_run is not None:
        wandb_run.summary["output_dir"] = str(output_dir)
        wandb_run.summary["metrics_path"] = str(output_dir / "metrics.json")
        wandb_run.finish()

    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to transformer model YAML.")
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
        default=None,
        help="Directory where model files, resolved config, and metrics are written.",
    )
    args = parser.parse_args()

    cfg = load_resolved_config(args.cfg, args.train_cfg)
    if args.output_dir is not None:
        cfg = apply_run_paths(cfg, Path(args.output_dir))
    elif "output_dir" in cfg:
        cfg = apply_run_paths(cfg, Path(str(cfg.output_dir)))
    else:
        raise ValueError("--output_dir is required when cfg has no output_dir.")

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
