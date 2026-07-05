from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any, Callable

import torch
from miditok import TokSequence
from miditok.pytorch_data import DatasetMIDI
from omegaconf import DictConfig, OmegaConf

from src.data.tokenizer import MidiTokBuilder
from src.utils.midi_utils import chunk_split, load_midi_paths_from_list


@dataclass
class GenerationSample:
    sample_id: int
    source_path: Path
    chunk_path: Path
    prompt_ids: list[int]
    reference_ids: list[int]
    generated_ids: list[int]
    prompt_length: int
    reference_length: int
    generated_length: int


class BaseModelAdapter:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    @property
    def max_context_length(self) -> int:
        raise NotImplementedError

    def next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or checkpoint.pt path.")
    parser.add_argument("--model_cfg", required=True, help="Model config path used as adapter fallback.")
    parser.add_argument("--tok_cfg", required=True, help="Tokenizer YAML config.")
    parser.add_argument("--data_list", required=True, help="Text file containing MIDI paths.")
    parser.add_argument("--eval_cfg", default="configs/eval/generation_shared.yaml")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging regardless of config.")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    return parser.parse_args()


def load_eval_config(eval_cfg_path: str) -> DictConfig:
    cfg = OmegaConf.load(eval_cfg_path)
    metrics_cfg_path = str(cfg.get("metrics_config", "configs/eval/music_metrics.yaml"))
    cfg.metrics = OmegaConf.load(metrics_cfg_path)
    OmegaConf.resolve(cfg)
    return cfg


def resolve_device(cfg: DictConfig) -> torch.device:
    device_name = str(cfg.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested, but no CUDA device is available.")
    return torch.device(device_name)


def cache_key(paths: list[Path], tok_cfg: str, max_seq_len: int, split_name: str) -> str:
    h = sha256()
    h.update(tok_cfg.encode("utf8"))
    h.update(str(max_seq_len).encode("utf8"))
    h.update(split_name.encode("utf8"))
    for path in paths:
        h.update(str(path).encode("utf8"))
    return h.hexdigest()[:16]


def build_eval_chunks(
    tokenizer,
    source_paths: list[Path],
    tok_cfg: str,
    max_seq_len: int,
    split_name: str,
) -> list[Path]:
    save_dir = Path("cache_chunks") / "eval" / cache_key(source_paths, tok_cfg, max_seq_len, split_name) / split_name
    return chunk_split(source_paths, tokenizer, str(save_dir), max_seq_len)


def build_token_dataset(tokenizer, chunk_paths: list[Path], max_seq_len: int) -> DatasetMIDI:
    return DatasetMIDI(
        files_paths=chunk_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["EOS_None"],
    )


def select_eval_samples(
    data_list_path: Path,
    tokenizer,
    tok_cfg: str,
    eval_cfg: DictConfig,
    max_seq_len: int,
) -> list[tuple[Path, Path, list[int]]]:
    midi_paths = load_midi_paths_from_list(data_list_path)
    split_name = str(eval_cfg.get("split", "test"))
    chunk_paths = build_eval_chunks(tokenizer, midi_paths, tok_cfg, max_seq_len, split_name)
    chunk_paths = sorted(chunk_paths)
    Random(int(eval_cfg.seed)).shuffle(chunk_paths)
    selected_chunks = chunk_paths[: int(eval_cfg.num_samples)]

    dataset = build_token_dataset(tokenizer, selected_chunks, max_seq_len)
    samples: list[tuple[Path, Path, list[int]]] = []
    for idx, chunk_path in enumerate(selected_chunks):
        token_ids = dataset[idx]["input_ids"].tolist()
        samples.append((chunk_path, chunk_path, token_ids))
    return samples


def sample_next_token(logits: torch.Tensor, cfg: DictConfig) -> int:
    sampling_cfg = cfg.sampling
    forbidden_token_ids = sampling_cfg.get("forbidden_token_ids", [])
    if forbidden_token_ids:
        logits = logits.clone()
        valid_forbidden_ids = [
            int(token_id)
            for token_id in forbidden_token_ids
            if 0 <= int(token_id) < logits.size(-1)
        ]
        if valid_forbidden_ids:
            logits[valid_forbidden_ids] = float("-inf")

    if bool(sampling_cfg.get("greedy", False)):
        return int(torch.argmax(logits, dim=-1).item())

    temperature = float(sampling_cfg.get("temperature", 1.0))
    if temperature <= 0:
        raise ValueError("temperature must be > 0 when greedy=false")
    logits = logits / temperature

    top_k = int(sampling_cfg.get("top_k", 0))
    if top_k > 0:
        top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = top_values[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

    top_p = float(sampling_cfg.get("top_p", 1.0))
    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_continuation(
    adapter: BaseModelAdapter,
    prompt_ids: list[int],
    max_new_tokens: int,
    cfg: DictConfig,
    eos_token_id: int | None,
) -> list[int]:
    generated = prompt_ids.copy()
    continuation: list[int] = []
    min_new_tokens = int(cfg.sampling.get("min_new_tokens", 0))
    for _ in range(max_new_tokens):
        window = generated[-adapter.max_context_length :]
        input_ids = torch.tensor(window, dtype=torch.long, device=adapter.device).unsqueeze(0)
        logits = adapter.next_token_logits(input_ids)
        next_token = sample_next_token(logits[0], cfg)
        generated.append(next_token)
        continuation.append(next_token)
        if (
            eos_token_id is not None
            and next_token == eos_token_id
            and len(continuation) >= min_new_tokens
        ):
            break
    return continuation


def compute_prompt_length(token_ids: list[int], cfg: DictConfig) -> int:
    min_prompt = int(cfg.min_prompt_tokens)
    max_prompt = int(cfg.max_prompt_tokens)
    ratio_prompt = int(len(token_ids) * float(cfg.prompt_ratio))
    prompt_length = max(min_prompt, ratio_prompt)
    prompt_length = min(prompt_length, max_prompt, len(token_ids) - 1)
    return prompt_length


def decode_ids_to_midi(tokenizer, token_ids: list[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score = tokenizer.decode(TokSequence(ids=token_ids))
    if not output_path.exists():
        score.dump_midi(output_path)


def maybe_init_wandb(cfg: DictConfig, args: argparse.Namespace, out_dir: Path):
    enabled = bool(cfg.wandb.get("enabled", False)) or bool(args.wandb)
    if not enabled:
        return None
    import wandb

    project = args.wandb_project or str(cfg.wandb.get("project", "duet-of-models"))
    run_name = args.wandb_run_name or cfg.wandb.get("run_name", None)
    run = wandb.init(
        project=project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    run.summary["output_dir"] = str(out_dir)
    return run


def compute_repetition_4gram(token_ids: list[int]) -> float:
    if len(token_ids) < 4:
        return 0.0
    ngrams: dict[tuple[int, ...], int] = {}
    total = len(token_ids) - 3
    repeated = 0
    for i in range(total):
        gram = tuple(token_ids[i : i + 4])
        ngrams[gram] = ngrams.get(gram, 0) + 1
    for count in ngrams.values():
        if count > 1:
            repeated += count
    return repeated / max(total, 1)


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_totals: dict[str, float] = {}
    numeric_counts: dict[str, int] = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                numeric_totals[key] = numeric_totals.get(key, 0.0) + float(value)
                numeric_counts[key] = numeric_counts.get(key, 0) + 1
    aggregated = {
        key: numeric_totals[key] / numeric_counts[key] for key in sorted(numeric_totals)
    }
    aggregated["num_samples"] = len(records)
    aggregated["decode_success_count"] = sum(1 for record in records if record.get("decode_success"))
    aggregated["decode_failure_count"] = len(records) - aggregated["decode_success_count"]
    aggregated["decode_success_rate"] = aggregated["decode_success_count"] / max(len(records), 1)
    return aggregated


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def evaluate_sample(
    sample: GenerationSample,
    tokenizer,
    out_dir: Path,
    metrics_cfg: DictConfig,
) -> dict[str, Any]:
    import pretty_midi

    from src.evaluation.music_metrics import midi_roundtrip_metrics_onset_chroma

    sample_dir = out_dir / "samples" / f"{sample.sample_id:04d}"
    prompt_path = sample_dir / "prompt.mid"
    reference_path = sample_dir / "reference.mid"
    generated_path = sample_dir / "generated.mid"
    result: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "source_path": str(sample.source_path),
        "chunk_path": str(sample.chunk_path),
        "prompt_length": sample.prompt_length,
        "reference_length": sample.reference_length,
        "generated_length": sample.generated_length,
        "repetition_4gram": compute_repetition_4gram(sample.generated_ids[sample.prompt_length :]),
        "decode_success": False,
    }
    try:
        decode_ids_to_midi(tokenizer, sample.prompt_ids, prompt_path)
        decode_ids_to_midi(tokenizer, sample.reference_ids, reference_path)
        decode_ids_to_midi(tokenizer, sample.generated_ids, generated_path)
        prompt_end_s = pretty_midi.PrettyMIDI(str(prompt_path)).get_end_time()
        metrics = midi_roundtrip_metrics_onset_chroma(
            original_mid=str(reference_path),
            reconstructed_mid=str(generated_path),
            onset_tol=float(metrics_cfg.onset_tolerance_sec),
            include_drums=bool(metrics_cfg.include_drums),
            fs_chroma=int(metrics_cfg.chroma_fs),
            calculate_transpose_invariant_chroma=bool(metrics_cfg.transpose_invariant),
            start_s=float(prompt_end_s),
            max_len_s=None,
        )
        result.update(
            {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "onset_mae_sec": metrics["onset_mae_sec"],
                "dur_mae_sec": metrics["dur_mae_sec"],
                "chroma_dtw": metrics["chroma_dtw"],
                "decode_success": True,
                "prompt_midi": str(prompt_path),
                "reference_midi": str(reference_path),
                "generated_midi": str(generated_path),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    save_json(
        sample_dir / "sample.json",
        {
            **result,
            "prompt_ids": sample.prompt_ids,
            "reference_ids": sample.reference_ids,
            "generated_ids": sample.generated_ids,
        },
    )
    return result


AdapterFactory = Callable[[Path, Path, int, torch.device], BaseModelAdapter]


def run_generation_evaluation(
    *,
    model_type: str,
    adapter_factory: AdapterFactory,
    args: argparse.Namespace | None = None,
) -> None:
    args = parse_args() if args is None else args
    eval_cfg = load_eval_config(args.eval_cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "eval_config.resolved.json", OmegaConf.to_container(eval_cfg, resolve=True))

    tokenizer = MidiTokBuilder.from_yaml(args.tok_cfg).to_MidiTok()
    device = resolve_device(eval_cfg)
    checkpoint_path = Path(args.checkpoint)
    model_cfg_path = Path(args.model_cfg)
    adapter = adapter_factory(
        checkpoint_path=checkpoint_path,
        model_cfg_path=model_cfg_path,
        tokenizer_vocab_size=len(tokenizer),
        device=device,
    )
    wandb_run = maybe_init_wandb(eval_cfg, args, out_dir)

    samples = select_eval_samples(
        data_list_path=Path(args.data_list),
        tokenizer=tokenizer,
        tok_cfg=args.tok_cfg,
        eval_cfg=eval_cfg,
        max_seq_len=adapter.max_context_length,
    )
    eos_token_id = tokenizer["EOS_None"] if "EOS_None" in tokenizer else None

    records: list[dict[str, Any]] = []
    for sample_id, (source_path, chunk_path, token_ids) in enumerate(samples):
        if len(token_ids) < max(int(eval_cfg.min_prompt_tokens) + 1, 2):
            continue
        prompt_length = compute_prompt_length(token_ids, eval_cfg)
        generation_limit = min(
            len(token_ids) - prompt_length,
            int(eval_cfg.max_generation_tokens),
        )
        prompt_ids = token_ids[:prompt_length]
        reference_full_ids = token_ids[: prompt_length + generation_limit]
        continuation_ids = generate_continuation(
            adapter=adapter,
            prompt_ids=prompt_ids,
            max_new_tokens=generation_limit,
            cfg=eval_cfg,
            eos_token_id=eos_token_id,
        )
        generated_full_ids = prompt_ids + continuation_ids
        sample = GenerationSample(
            sample_id=sample_id,
            source_path=source_path,
            chunk_path=chunk_path,
            prompt_ids=prompt_ids,
            reference_ids=reference_full_ids,
            generated_ids=generated_full_ids,
            prompt_length=len(prompt_ids),
            reference_length=len(reference_full_ids) - len(prompt_ids),
            generated_length=len(continuation_ids),
        )
        records.append(
            evaluate_sample(
                sample=sample,
                tokenizer=tokenizer,
                out_dir=out_dir,
                metrics_cfg=eval_cfg.metrics,
            )
        )

    aggregate = aggregate_metrics(records)
    aggregate["model_type"] = model_type
    aggregate["checkpoint"] = str(checkpoint_path)
    save_json(out_dir / "aggregate_metrics.json", aggregate)
    save_json(out_dir / "per_sample_metrics.json", records)

    if wandb_run is not None:
        wandb_run.log({f"eval/{key}": value for key, value in aggregate.items() if isinstance(value, (int, float))})
        wandb_run.summary["aggregate_metrics_path"] = str(out_dir / "aggregate_metrics.json")
        wandb_run.finish()

    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(
        "Use a model-specific evaluator: "
        "python -m src.models.Transformer.generate_and_score or "
        "python -m src.models.xlstm.generate_and_score"
    )
