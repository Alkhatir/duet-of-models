from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", required=True, help="Directory containing generated samples.")
    parser.add_argument("--eval_cfg", default="configs/eval/generation_shared.yaml")
    parser.add_argument("--model_type", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging regardless of config.")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return data


def load_eval_config(eval_cfg_path: str) -> dict[str, Any]:
    cfg = load_yaml(Path(eval_cfg_path))
    metrics_cfg_path = str(cfg.get("metrics_config", "configs/eval/music_metrics.yaml"))
    cfg["metrics"] = load_yaml(Path(metrics_cfg_path))
    return cfg


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def maybe_init_wandb(cfg: dict[str, Any], args: argparse.Namespace, samples_dir: Path):
    wandb_cfg = cfg.get("wandb", {})
    enabled = bool(wandb_cfg.get("enabled", False)) or bool(args.wandb)
    if not enabled:
        return None
    import wandb

    project = args.wandb_project or str(wandb_cfg.get("project", "duet-of-models"))
    run_name = args.wandb_run_name or wandb_cfg.get("run_name", None)
    run = wandb.init(
        project=project,
        name=run_name,
        config=cfg,
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    run.summary["samples_dir"] = str(samples_dir)
    return run


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
    aggregate = {
        key: numeric_totals[key] / numeric_counts[key] for key in sorted(numeric_totals)
    }
    aggregate["num_samples"] = len(records)
    aggregate["decode_success_count"] = sum(1 for record in records if record.get("decode_success"))
    aggregate["decode_failure_count"] = len(records) - aggregate["decode_success_count"]
    aggregate["decode_success_rate"] = aggregate["decode_success_count"] / max(len(records), 1)
    aggregate["score_success_count"] = sum(1 for record in records if record.get("score_success"))
    aggregate["score_failure_count"] = len(records) - aggregate["score_success_count"]
    aggregate["score_success_rate"] = aggregate["score_success_count"] / max(len(records), 1)
    return aggregate


def score_sample(sample_json_path: Path, metrics_cfg: dict[str, Any]) -> dict[str, Any]:
    import pretty_midi

    from src.evaluation.music_metrics import midi_roundtrip_metrics_onset_chroma

    sample = load_json(sample_json_path)
    result: dict[str, Any] = {
        **{key: value for key, value in sample.items() if not key.endswith("_ids")},
        "score_success": False,
    }
    if not bool(sample.get("decode_success", False)):
        result["error"] = sample.get("error", "Decode failed during generation.")
        return result

    prompt_path = Path(str(sample["prompt_midi"]))
    reference_path = Path(str(sample["reference_midi"]))
    generated_path = Path(str(sample["generated_midi"]))
    try:
        prompt_end_s = pretty_midi.PrettyMIDI(str(prompt_path)).get_end_time()
        metrics = midi_roundtrip_metrics_onset_chroma(
            original_mid=str(reference_path),
            reconstructed_mid=str(generated_path),
            onset_tol=float(metrics_cfg["onset_tolerance_sec"]),
            include_drums=bool(metrics_cfg["include_drums"]),
            fs_chroma=int(metrics_cfg["chroma_fs"]),
            calculate_transpose_invariant_chroma=bool(metrics_cfg["transpose_invariant"]),
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
                "score_success": True,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    eval_cfg = load_eval_config(args.eval_cfg)
    generation_config_path = samples_dir / "generation_config.json"
    generation_config = load_json(generation_config_path) if generation_config_path.is_file() else {}

    records = [
        score_sample(sample_json_path, eval_cfg["metrics"])
        for sample_json_path in sorted((samples_dir / "samples").glob("*/sample.json"))
    ]
    aggregate = aggregate_metrics(records)
    aggregate["model_type"] = args.model_type or generation_config.get("model_type")
    aggregate["checkpoint"] = args.checkpoint or generation_config.get("checkpoint")
    save_json(samples_dir / "aggregate_metrics.json", aggregate)
    save_json(samples_dir / "per_sample_metrics.json", records)

    wandb_run = maybe_init_wandb(eval_cfg, args, samples_dir)
    if wandb_run is not None:
        wandb_run.log({f"eval/{key}": value for key, value in aggregate.items() if isinstance(value, (int, float))})
        wandb_run.summary["aggregate_metrics_path"] = str(samples_dir / "aggregate_metrics.json")
        wandb_run.finish()

    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
