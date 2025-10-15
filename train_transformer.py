import dataclasses
import argparse
from pathlib import Path
from typing import Any, Dict
from random import shuffle, seed as set_seed

import torch
from omegaconf import OmegaConf

import wandb

from transformers import (
    Trainer,
    TrainingArguments,
    LlamaConfig,
    GPT2Config,
    AutoModelForCausalLM,
)

from .tokenization import MidiTokBuilder
from .utils import build_three_datasets_from_chunks, iter_midi_paths, PerplexityCallback

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


def build_model(cfg: Dict[str, Any]):
    m = cfg["model"]
    arch = m.get("architecture", "llama").lower()
    if (
        arch == "llama"
    ):  # in case I used "LLaMA" or "llama2" which I think I never gonna do unless I understand what theta parameter does haha :D
        conf = LlamaConfig(
            vocab_size=m["vocab_size"],
            hidden_size=m["hidden_size"],
            intermediate_size=m["intermediate_size"],
            num_attention_heads=m["num_attention_heads"],
            num_hidden_layers=m["num_hidden_layers"],
            max_position_embeddings=m["max_position_embeddings"],
            rope_theta=m.get("rope_theta", 1e6),
        )
        return AutoModelForCausalLM.from_config(conf)
    elif arch == "gpt2":
        conf = GPT2Config(
            vocab_size=m["vocab_size"],
            n_embd=m["hidden_size"],
            n_layer=m["num_hidden_layers"],
            n_head=m["num_attention_heads"],
            n_positions=m["max_position_embeddings"],
        )
        return AutoModelForCausalLM.from_config(conf)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def filter_training_args(raw: Dict[str, Any], output_dir: str) -> TrainingArguments:
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    kwargs = {k: v for k, v in raw.items() if k in fields}
    kwargs.setdefault("output_dir", output_dir)
    return TrainingArguments(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument(
        "--tok_cfg", default=None, help="Path to tokenizer YAML config "
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Path to data directory where MIDI files are located",
    )
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.cfg), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError("Configuration file must resolve to a dictionary.")
    seed_value = cfg.get("seed", 42)  # Default to 42 if "seed" is not found
    torch.manual_seed(int(seed_value))

    model = build_model(cfg.get("model", {}))

    tokenizer = MidiTokBuilder.from_yaml(args.tok_cfg).to_MidiTok()

    if args.data_dir is None:
        raise ValueError("You must provide a data directory with --data_dir")

    all_midis = list(iter_midi_paths(Path(args.data_dir)))
    set_seed(seed_value)
    shuffle(all_midis)
    n = len(all_midis)

    train_ds, val_ds, test_ds, collator = build_three_datasets_from_chunks(
        tokenizer=tokenizer,
        train_src=all_midis[: int(n * 0.8)],
        val_src=all_midis[int(n * 0.8) : int(n * 0.9)],
        test_src=all_midis[int(n * 0.9) :],
        max_seq_len=cfg["data"]["block_size"],
    )

    training_args = filter_training_args(cfg["train"], cfg["output_dir"])

    # TODO: Initialize W&B logging and pass config (except vocab_file path) to it
    wandb.init(
        project=cfg.get("wandb_project", "duet-of-models"),
        config={
            **cfg,
            "model": {
                k: v for k, v in cfg.get("model", {}).items() if k != "vocab_file"
            },
        },
        name=cfg.get("run_name", None),
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[PerplexityCallback()],
    )
    trainer.train()

    # Save final model & config for easy reuse
    trainer.save_model(cfg["output_dir"])

    test_metrics = trainer.evaluate(test_ds)
    print("Test metrics:", test_metrics)
    test_metrics["test_perplexity"] = math.exp(min(20.0, test_metrics["eval_loss"]))
    wandb.log(test_metrics)  # shows up as a final step in the same run


if __name__ == "__main__":
    main()
