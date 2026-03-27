import argparse
import json
import math
from pathlib import Path
from random import shuffle, seed as set_seed
from typing import Any

import torch
import torch.optim as optim
from dacite import from_dict
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.tokenizer import MidiTokBuilder
from src.utils.midi_utils import build_three_datasets_from_chunks
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_resolved_config(model_cfg_path: str, train_cfg_path: str) -> DictConfig:
    train_cfg = OmegaConf.load(train_cfg_path)
    model_cfg = OmegaConf.load(model_cfg_path)
    if "model" not in model_cfg:
        model_cfg = OmegaConf.create({"model": model_cfg})
    cfg = OmegaConf.merge(train_cfg, model_cfg)
    OmegaConf.resolve(cfg)
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


def build_dataloaders(
    cfg: DictConfig, tokenizer, data_list_path: Path
) -> tuple[DataLoader, DataLoader, DataLoader]:
    midi_paths = load_midi_paths_from_list(data_list_path)

    seed_value = int(cfg.get("seed", 42))
    set_seed(seed_value)
    shuffle(midi_paths)
    total = len(midi_paths)

    train_ds, val_ds, test_ds, collator = build_three_datasets_from_chunks(
        tokenizer=tokenizer,
        train_src=midi_paths[: int(total * 0.8)],
        val_src=midi_paths[int(total * 0.8) : int(total * 0.9)],
        test_src=midi_paths[int(total * 0.9) :],
        max_seq_len=int(cfg.data.block_size),
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
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
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


def build_scheduler(optimizer: optim.Optimizer, cfg: DictConfig):
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


def evaluate(
    model: xLSTMLMModel, loader: DataLoader, cfg: DictConfig, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with make_autocast_context(cfg, device):
                logits = model(input_ids)
                loss = compute_loss(logits, labels)
            total_loss += float(loss.item())
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    return {
        "loss": avg_loss,
        "perplexity": math.exp(min(20.0, avg_loss)),
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
        with (output_dir / "metrics.json").open("w", encoding="utf8") as fh:
            json.dump(metrics, fh, indent=2, sort_keys=True)


def train(cfg: DictConfig, data_list_path: Path, tok_cfg: str) -> dict[str, float]:
    seed_value = int(cfg.get("seed", 42))
    torch.manual_seed(seed_value)

    tokenizer = MidiTokBuilder.from_yaml(tok_cfg).to_MidiTok()
    cfg = prepare_runtime_config(cfg, tokenizer_vocab_size=len(tokenizer))
    device = resolve_device(str(cfg.train.get("device", "cuda")))

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, tokenizer, data_list_path
    )
    model = build_model(cfg, device)
    weight_precision = str(cfg.train.get("weight_precision", "float32"))
    model = model.to(dtype=torch_dtype_map[weight_precision])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    output_dir = Path(str(cfg.output_dir))
    logging_steps = max(1, int(cfg.train.get("logging_steps", 20)))
    eval_steps = max(1, int(cfg.train.get("eval_steps", 100)))
    save_steps = max(1, int(cfg.train.get("save_steps", eval_steps)))
    max_grad_norm = float(cfg.train.get("max_grad_norm", 0.0))
    epochs = int(cfg.train.get("num_train_epochs", 1))

    global_step = 0
    running_loss = 0.0
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
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            running_loss += float(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % logging_steps == 0:
                avg_train_loss = running_loss / logging_steps
                print(
                    f"step={global_step} train_loss={avg_train_loss:.4f} "
                    f"train_perplexity={math.exp(min(20.0, avg_train_loss)):.4f}"
                )
                running_loss = 0.0

            if global_step % eval_steps == 0:
                val_metrics = evaluate(model, val_loader, cfg, device)
                print(
                    f"step={global_step} val_loss={val_metrics['loss']:.4f} "
                    f"val_perplexity={val_metrics['perplexity']:.4f}"
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
    test_metrics = evaluate(model, test_loader, cfg, device)
    final_metrics = {
        "global_step": global_step,
        "val_loss": val_metrics["loss"],
        "val_perplexity": val_metrics["perplexity"],
        "test_loss": test_metrics["loss"],
        "test_perplexity": test_metrics["perplexity"],
    }
    save_artifacts(model, optimizer, cfg, output_dir, global_step, final_metrics)
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
        "--data_list",
        required=True,
        help="Path to the file containing MIDI files paths.",
    )
    args = parser.parse_args()

    cfg = load_resolved_config(args.cfg, args.train_cfg)
    metrics = train(cfg, data_list_path=Path(args.data_list), tok_cfg=args.tok_cfg)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
