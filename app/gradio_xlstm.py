from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from miditok import TokSequence
from omegaconf import OmegaConf

from src.data.tokenizer import MidiTokBuilder
from src.evaluation.generate_and_score import (
    decode_ids_to_midi,
    generate_continuation,
)
from src.models.xlstm.generate_and_score import XLSTMAdapter


DEFAULT_CHECKPOINT = "experiments/xlstm-small-batch8-tok11/best/checkpoint.pt"
DEFAULT_MODEL_CFG = "configs/model/xlstm/small.yaml"
DEFAULT_TOKENIZER_CFG = "configs/data/11.yaml"
DEFAULT_OUTPUT_DIR = "experiments/gradio_xlstm_generations"
DEFAULT_SOUNDFONT = ""
MIDI_PLAYER_HEAD = """
<style>
  .midi-web-player {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    background: #ffffff;
    overflow: hidden;
  }
  .midi-web-player__title {
    padding: 8px 10px;
    color: #0f172a;
    font: 13px sans-serif;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
  }
  .midi-web-player midi-player {
    display: block;
    width: 100%;
  }
  .midi-web-player midi-visualizer {
    display: block;
    width: 100%;
    max-height: 420px;
    overflow: auto;
  }
  .midi-web-player-empty {
    padding: 12px;
    color: #475569;
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
  }
</style>
"""
MIDI_PLAYER_SCRIPT_URL = (
    "https://cdn.jsdelivr.net/combine/"
    "npm/tone@14.7.58,"
    "npm/@magenta/music@1.23.1/es6/core.js,"
    "npm/html-midi-player@1.5.0"
)


@dataclass(frozen=True)
class LoadedXLSTM:
    checkpoint_path: Path
    model_cfg_path: Path
    tokenizer_cfg_path: Path
    device_name: str
    tokenizer: Any
    adapter: XLSTMAdapter


_MODEL_CACHE: LoadedXLSTM | None = None
STEP_CHECKPOINT_RE = re.compile(r"^step-(\d+)$")


@dataclass(frozen=True)
class RunSelection:
    label: str
    path: Path


@dataclass(frozen=True)
class CheckpointSelection:
    label: str
    path: Path
    sort_key: tuple[int, int]


def discover_checkpoints(root: Path = Path("experiments")) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path) for path in root.glob("**/checkpoint.pt"))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _is_checkpoint_variant_dir(path: Path) -> bool:
    return path.name == "best" or STEP_CHECKPOINT_RE.match(path.name) is not None


def _run_dir_for_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_dir = checkpoint_path.parent
    if _is_checkpoint_variant_dir(checkpoint_dir):
        return checkpoint_dir.parent
    return checkpoint_dir


def discover_runs(root: Path = Path("experiments")) -> list[RunSelection]:
    if not root.exists():
        return []
    runs = {
        _run_dir_for_checkpoint(path)
        for path in root.glob("**/checkpoint.pt")
        if path.is_file()
    }
    return [
        RunSelection(label=_display_path(path), path=path)
        for path in sorted(runs, key=lambda item: _display_path(item))
    ]


def _resolve_run_path(run_label: str | None, root: Path = Path("experiments")) -> Path | None:
    if not run_label:
        return None
    path = Path(run_label).expanduser()
    if path.is_dir():
        return path
    root_path = root / run_label
    if root_path.is_dir():
        return root_path
    for run in discover_runs(root):
        if run.label == run_label:
            return run.path
    return None


def _checkpoint_label(path: Path, run_dir: Path) -> str:
    try:
        checkpoint_dir = path.parent.relative_to(run_dir)
    except ValueError:
        return _display_path(path)
    if str(checkpoint_dir) == ".":
        return "last (final/checkpoint.pt)"
    if checkpoint_dir.name == "best":
        return "best/checkpoint.pt"
    match = STEP_CHECKPOINT_RE.match(checkpoint_dir.name)
    if match:
        return f"{checkpoint_dir.name} (step {match.group(1)})"
    return f"{checkpoint_dir}/checkpoint.pt"


def list_run_checkpoints(run_dir: Path | None) -> list[CheckpointSelection]:
    if run_dir is None or not run_dir.is_dir():
        return []

    selections: list[CheckpointSelection] = []
    for checkpoint_path in run_dir.glob("**/checkpoint.pt"):
        checkpoint_dir = checkpoint_path.parent
        if checkpoint_dir == run_dir:
            selections.append(
                CheckpointSelection(
                    label=_checkpoint_label(checkpoint_path, run_dir),
                    path=checkpoint_path,
                    sort_key=(2, 0),
                )
            )
            continue
        if checkpoint_dir.parent != run_dir:
            continue
        if checkpoint_dir.name == "best":
            selections.append(
                CheckpointSelection(
                    label=_checkpoint_label(checkpoint_path, run_dir),
                    path=checkpoint_path,
                    sort_key=(0, 0),
                )
            )
            continue
        match = STEP_CHECKPOINT_RE.match(checkpoint_dir.name)
        if match:
            selections.append(
                CheckpointSelection(
                    label=_checkpoint_label(checkpoint_path, run_dir),
                    path=checkpoint_path,
                    sort_key=(1, int(match.group(1))),
                )
            )

    return sorted(selections, key=lambda item: item.sort_key)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _resolve_tokenizer_cfg(run_dir: Path | None, fallback: str) -> str:
    if run_dir is None:
        return fallback
    for summary_name in ("summary.json", "wandb-summary.json"):
        summary = _load_json(run_dir / summary_name)
        tok_cfg = summary.get("tokenizer_config_path")
        if tok_cfg:
            return str(tok_cfg)
    return fallback


def _resolve_model_cfg(run_dir: Path | None, fallback: str) -> str:
    if run_dir is None:
        return fallback
    resolved = run_dir / "config.resolved.yaml"
    if resolved.is_file():
        return _display_path(resolved)
    best_resolved = run_dir / "best" / "config.resolved.yaml"
    if best_resolved.is_file():
        return _display_path(best_resolved)
    return fallback


def _summarize_run_config(run_dir: Path | None, model_cfg_path: str, tokenizer_cfg_path: str) -> str:
    if run_dir is None:
        return "No run selected."

    summary = _load_json(run_dir / "summary.json") or _load_json(run_dir / "wandb-summary.json")
    cfg = OmegaConf.load(model_cfg_path) if Path(model_cfg_path).is_file() else OmegaConf.create({})
    model_cfg = cfg.get("model", cfg)
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})

    lines = [
        f"Run: {_display_path(run_dir)}",
        f"Model config: {model_cfg_path}",
        f"Tokenizer config: {tokenizer_cfg_path}",
    ]
    if model_cfg:
        fields = {
            "embedding_dim": model_cfg.get("embedding_dim"),
            "num_blocks": model_cfg.get("num_blocks"),
            "context_length": model_cfg.get("context_length"),
            "vocab_size": model_cfg.get("vocab_size"),
        }
        details = ", ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
        if details:
            lines.append(f"Model: {details}")
    if train_cfg:
        learning_rate = train_cfg.get("learning_rate")
        batch_size = train_cfg.get("per_device_train_batch_size")
        if learning_rate is not None or batch_size is not None:
            lines.append(f"Training: batch_size={batch_size}, learning_rate={learning_rate}")
    block_size = data_cfg.get("block_size")
    if block_size is not None:
        lines.append(f"Data: block_size={block_size}")
    if summary:
        best_step = summary.get("best/global_step")
        best_loss = summary.get("best/val_loss")
        final_step = summary.get("final/global_step") or summary.get("step")
        metric_bits = []
        if best_step is not None:
            metric_bits.append(f"best_step={best_step}")
        if best_loss is not None:
            metric_bits.append(f"best_val_loss={best_loss:.4f}" if isinstance(best_loss, float) else f"best_val_loss={best_loss}")
        if final_step is not None:
            metric_bits.append(f"final_step={final_step}")
        if metric_bits:
            lines.append(f"Metrics: {', '.join(metric_bits)}")
    return "\n".join(lines)


def _checkpoint_path_from_label(run_dir: Path | None, checkpoint_label: str | None) -> str:
    if run_dir is None:
        return ""
    for selection in list_run_checkpoints(run_dir):
        if selection.label == checkpoint_label:
            return _display_path(selection.path)
    return ""


def _best_checkpoint_path(run_dir: Path | None) -> str:
    if run_dir is None:
        return ""
    best = run_dir / "best" / "checkpoint.pt"
    if best.is_file():
        return _display_path(best)
    for selection in list_run_checkpoints(run_dir):
        if selection.sort_key[0] == 2:
            return _display_path(selection.path)
    checkpoints = list_run_checkpoints(run_dir)
    return _display_path(checkpoints[0].path) if checkpoints else ""


def update_run_selection(
    run_label: str | None,
    use_best_checkpoint: bool,
    model_cfg_fallback: str,
    tokenizer_cfg_fallback: str,
) -> tuple[Any, str, str, str, str]:
    import gradio as gr

    run_dir = _resolve_run_path(run_label)
    checkpoints = list_run_checkpoints(run_dir)
    checkpoint_choices = [selection.label for selection in checkpoints if selection.sort_key[0] != 0]
    default_checkpoint = checkpoint_choices[-1] if checkpoint_choices else None
    model_cfg_path = _resolve_model_cfg(run_dir, model_cfg_fallback)
    tokenizer_cfg_path = _resolve_tokenizer_cfg(run_dir, tokenizer_cfg_fallback)
    resolved_checkpoint = (
        _best_checkpoint_path(run_dir)
        if use_best_checkpoint
        else _checkpoint_path_from_label(run_dir, default_checkpoint)
    )
    return (
        gr.update(
            choices=checkpoint_choices,
            value=default_checkpoint,
            interactive=not use_best_checkpoint and bool(checkpoint_choices),
        ),
        resolved_checkpoint,
        model_cfg_path,
        tokenizer_cfg_path,
        _summarize_run_config(run_dir, model_cfg_path, tokenizer_cfg_path),
    )


def update_checkpoint_selection(
    run_label: str | None,
    use_best_checkpoint: bool,
    checkpoint_label: str | None,
) -> tuple[Any, str]:
    import gradio as gr

    run_dir = _resolve_run_path(run_label)
    checkpoints = list_run_checkpoints(run_dir)
    checkpoint_choices = [selection.label for selection in checkpoints if selection.sort_key[0] != 0]
    selected = checkpoint_label if checkpoint_label in checkpoint_choices else (checkpoint_choices[-1] if checkpoint_choices else None)
    if use_best_checkpoint:
        resolved_checkpoint = _best_checkpoint_path(run_dir)
    else:
        resolved_checkpoint = _checkpoint_path_from_label(run_dir, selected)
    return (
        gr.update(value=selected, interactive=not use_best_checkpoint and bool(checkpoint_choices)),
        resolved_checkpoint,
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but no CUDA device is available.")
    return torch.device(device_name)


def load_xlstm(
    checkpoint_path: str,
    model_cfg_path: str,
    tokenizer_cfg_path: str,
    device_name: str,
) -> LoadedXLSTM:
    global _MODEL_CACHE

    checkpoint = Path(checkpoint_path).expanduser()
    model_cfg = Path(model_cfg_path).expanduser()
    tokenizer_cfg = Path(tokenizer_cfg_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not model_cfg.is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_cfg}")
    if not tokenizer_cfg.is_file():
        raise FileNotFoundError(f"Tokenizer config does not exist: {tokenizer_cfg}")

    cache_hit = (
        _MODEL_CACHE is not None
        and _MODEL_CACHE.checkpoint_path == checkpoint
        and _MODEL_CACHE.model_cfg_path == model_cfg
        and _MODEL_CACHE.tokenizer_cfg_path == tokenizer_cfg
        and _MODEL_CACHE.device_name == device_name
    )
    if cache_hit:
        return _MODEL_CACHE

    tokenizer = MidiTokBuilder.from_yaml(str(tokenizer_cfg)).to_MidiTok()
    device = resolve_device(device_name)
    adapter = XLSTMAdapter(
        checkpoint_path=checkpoint,
        model_cfg_path=model_cfg,
        tokenizer_vocab_size=len(tokenizer),
        device=device,
    )
    _MODEL_CACHE = LoadedXLSTM(
        checkpoint_path=checkpoint,
        model_cfg_path=model_cfg,
        tokenizer_cfg_path=tokenizer_cfg,
        device_name=device_name,
        tokenizer=tokenizer,
        adapter=adapter,
    )
    return _MODEL_CACHE


def _as_token_ids(encoded: TokSequence | list[TokSequence]) -> list[int]:
    if isinstance(encoded, list):
        if not encoded:
            raise ValueError("The uploaded MIDI produced no token sequences.")
        ids = encoded[0].ids
    else:
        ids = encoded.ids
    if not ids:
        raise ValueError("The uploaded MIDI produced no token ids.")
    return [int(token_id) for token_id in ids]


def encode_prompt(tokenizer, prompt_midi: str | None, prompt_tokens: int) -> list[int]:
    if prompt_midi:
        prompt_path = Path(prompt_midi).expanduser()
        if not prompt_path.is_file() and not prompt_path.is_absolute():
            prompt_path = Path.cwd() / prompt_path
        token_ids = _as_token_ids(tokenizer.encode(prompt_path))
        return token_ids[: max(1, min(prompt_tokens, len(token_ids)))]
    if "BOS_None" in tokenizer:
        return [int(tokenizer["BOS_None"])]
    return [int(tokenizer.pad_token_id)]


def build_sampling_config(
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    forbidden_token_ids: list[int] | None = None,
    min_new_tokens: int = 0,
) -> Any:
    return OmegaConf.create(
        {
            "sampling": {
                "greedy": bool(greedy),
                "temperature": float(temperature),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "forbidden_token_ids": forbidden_token_ids or [],
                "min_new_tokens": int(min_new_tokens),
            }
        }
    )


def special_token_ids_to_forbid(tokenizer) -> list[int]:
    forbidden_ids = []
    for token_name in ("PAD_None", "BOS_None", "MASK_None"):
        if token_name in tokenizer:
            forbidden_ids.append(int(tokenizer[token_name]))
    return forbidden_ids


def _token_text(tokenizer, token_id: int) -> str:
    try:
        return str(tokenizer[int(token_id)])
    except Exception:
        return f"<unknown:{token_id}>"


def _token_category(token_text: str) -> str:
    prefix = token_text.split("_", 1)[0]
    if prefix in {"PAD", "BOS", "EOS", "MASK"}:
        return "Special"
    known_prefixes = {
        "Bar",
        "Position",
        "Pitch",
        "PitchDrum",
        "Velocity",
        "Duration",
        "Rest",
        "Tempo",
        "TimeSig",
        "Program",
        "Chord",
        "Pedal",
        "PitchBend",
    }
    return prefix if prefix in known_prefixes else "Other"


def _count_token_categories(tokenizer, token_ids: list[int]) -> tuple[dict[str, int], dict[str, int]]:
    categories: dict[str, int] = {}
    tokens: dict[str, int] = {}
    for token_id in token_ids:
        token_text = _token_text(tokenizer, int(token_id))
        category = _token_category(token_text)
        categories[category] = categories.get(category, 0) + 1
        tokens[token_text] = tokens.get(token_text, 0) + 1
    return categories, tokens


def _decoded_midi_stats(midi_path: Path) -> dict[str, int | float | None]:
    from miditoolkit import MidiFile

    midi = MidiFile(str(midi_path))
    note_count = 0
    drum_note_count = 0
    track_count = 0
    pitches = []
    max_tick = 0
    for instrument in midi.instruments:
        if instrument.notes:
            track_count += 1
        for note in instrument.notes:
            note_count += 1
            drum_note_count += int(bool(instrument.is_drum))
            pitches.append(int(note.pitch))
            max_tick = max(max_tick, int(note.end))
    ticks_per_beat = int(getattr(midi, "ticks_per_beat", 0) or 0)
    duration_beats = max_tick / ticks_per_beat if ticks_per_beat else None
    return {
        "note_count": note_count,
        "tracks_with_notes": track_count,
        "drum_note_count": drum_note_count,
        "duration_ticks": max_tick,
        "duration_beats": round(duration_beats, 2) if duration_beats is not None else None,
        "min_pitch": min(pitches) if pitches else None,
        "max_pitch": max(pitches) if pitches else None,
    }


def generation_statistics(
    tokenizer,
    prompt_ids: list[int],
    continuation_ids: list[int],
    output_path: Path,
    *,
    special_tokens_blocked: bool,
    forbidden_token_ids: list[int],
    eos_token_id: int | None,
    sampling: dict[str, Any],
) -> str:
    prompt_categories, _ = _count_token_categories(tokenizer, prompt_ids)
    continuation_categories, continuation_tokens = _count_token_categories(
        tokenizer,
        continuation_ids,
    )
    total_generated = len(continuation_ids)
    special_count = continuation_categories.get("Special", 0)
    pitch_count = continuation_categories.get("Pitch", 0) + continuation_categories.get("PitchDrum", 0)
    duration_count = continuation_categories.get("Duration", 0)
    timing_count = (
        continuation_categories.get("Bar", 0)
        + continuation_categories.get("Position", 0)
        + continuation_categories.get("Rest", 0)
    )
    top_tokens = sorted(
        continuation_tokens.items(),
        key=lambda item: (-item[1], item[0]),
    )[:12]
    eos_generated = (
        eos_token_id is not None
        and bool(continuation_ids)
        and int(continuation_ids[-1]) == int(eos_token_id)
    )
    midi_stats = _decoded_midi_stats(output_path)
    note_count = int(midi_stats["note_count"] or 0)
    tokens_per_note = round(total_generated / note_count, 2) if note_count else None
    timing_per_pitch = round(timing_count / pitch_count, 2) if pitch_count else None

    payload = {
        "sampling": sampling,
        "special_token_blocking": {
            "enabled": special_tokens_blocked,
            "forbidden_token_ids": forbidden_token_ids,
            "forbidden_tokens": [
                _token_text(tokenizer, token_id) for token_id in forbidden_token_ids
            ],
        },
        "lengths": {
            "prompt_tokens_used": len(prompt_ids),
            "generated_tokens": total_generated,
            "stopped_on_eos": eos_generated,
        },
        "generated_token_categories": continuation_categories,
        "prompt_token_categories": prompt_categories,
        "quality_indicators": {
            "special_token_rate": round(special_count / max(total_generated, 1), 4),
            "pitch_token_rate": round(pitch_count / max(total_generated, 1), 4),
            "duration_token_rate": round(duration_count / max(total_generated, 1), 4),
            "timing_token_rate": round(timing_count / max(total_generated, 1), 4),
            "timing_tokens_per_pitch_token": timing_per_pitch,
            "generated_tokens_per_decoded_note": tokens_per_note,
            "unique_generated_tokens": len(continuation_tokens),
        },
        "decoded_midi": midi_stats,
        "most_common_generated_tokens": dict(top_tokens),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _resolve_midi_path(midi_path: str | None) -> Path | None:
    if not midi_path:
        return None
    path = Path(midi_path).expanduser()
    if not path.is_file() and not path.is_absolute():
        path = Path.cwd() / path
    return path if path.is_file() else None


def midi_player_html(midi_path: str | None, title: str = "MIDI") -> str:
    resolved_path = _resolve_midi_path(midi_path)
    if resolved_path is None:
        return "<div class='midi-web-player-empty'>No MIDI selected.</div>"

    player_id = "midi-player-" + hashlib.sha1(str(resolved_path.resolve()).encode("utf8")).hexdigest()[:12]
    visualizer_id = f"{player_id}-visualizer"
    iframe_id = f"{player_id}-frame"
    midi_b64 = base64.b64encode(resolved_path.read_bytes()).decode("ascii")
    label = html.escape(f"{title}: {resolved_path.name}")
    srcdoc = html.escape(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="{MIDI_PLAYER_SCRIPT_URL}"></script>
  <style>
    body {{ margin: 0; font-family: sans-serif; background: #fff; color: #0f172a; }}
    .title {{ padding: 8px 10px; font-size: 13px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; }}
    midi-player {{ display: block; width: 100%; }}
    midi-visualizer {{ display: block; width: 100%; min-height: 320px; max-height: 420px; overflow: auto; }}
    .status {{ padding: 10px; color: #475569; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="title">{label}</div>
  <midi-player id="{player_id}" sound-font visualizer="#{visualizer_id}"></midi-player>
  <midi-visualizer id="{visualizer_id}" type="piano-roll"></midi-visualizer>
  <div id="status" class="status">Loading MIDI player...</div>
  <script>
    const midiBase64 = "{midi_b64}";
    function base64ToBytes(value) {{
      const binary = atob(value);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {{
        bytes[index] = binary.charCodeAt(index);
      }}
      return bytes;
    }}
    customElements.whenDefined("midi-player").then(() => {{
      const blob = new Blob([base64ToBytes(midiBase64)], {{ type: "audio/midi" }});
      const url = URL.createObjectURL(blob);
      document.getElementById("{player_id}").src = url;
      document.getElementById("{visualizer_id}").src = url;
      document.getElementById("status").textContent = "";
    }}).catch((error) => {{
      document.getElementById("status").textContent = `Could not load MIDI player: ${{error}}`;
    }});
  </script>
</body>
</html>""",
        quote=True,
    )
    return (
        "<div class='midi-web-player'>"
        f"<iframe id='{iframe_id}' srcdoc=\"{srcdoc}\" "
        "style='display:block;width:100%;height:520px;border:0;' "
        "allow='autoplay; clipboard-read; clipboard-write'></iframe>"
        "</div>"
    )


def preview_midi(midi_path: str | None) -> str:
    return midi_player_html(midi_path, title="Prompt")


def generate_music(
    checkpoint_path: str,
    model_cfg_path: str,
    tokenizer_cfg_path: str,
    prompt_midi: str | None,
    prompt_tokens: int,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    block_special_tokens: bool,
    seed: int,
    device_name: str,
    output_dir: str,
) -> tuple[str, str, str, str]:
    loaded = load_xlstm(
        checkpoint_path=checkpoint_path,
        model_cfg_path=model_cfg_path,
        tokenizer_cfg_path=tokenizer_cfg_path,
        device_name=device_name,
    )

    torch.manual_seed(int(seed))
    if loaded.adapter.device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    prompt_ids = encode_prompt(
        loaded.tokenizer,
        prompt_midi=prompt_midi,
        prompt_tokens=int(prompt_tokens),
    )
    prompt_ids = prompt_ids[-loaded.adapter.max_context_length :]
    eos_token_id = loaded.tokenizer["EOS_None"] if "EOS_None" in loaded.tokenizer else None
    forbidden_token_ids = (
        special_token_ids_to_forbid(loaded.tokenizer) if block_special_tokens else []
    )
    min_new_tokens = min(64, int(max_new_tokens)) if block_special_tokens else 0
    sampling_cfg = build_sampling_config(
        greedy=greedy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        forbidden_token_ids=forbidden_token_ids,
        min_new_tokens=min_new_tokens,
    )
    continuation_ids = generate_continuation(
        adapter=loaded.adapter,
        prompt_ids=prompt_ids,
        max_new_tokens=int(max_new_tokens),
        cfg=sampling_cfg,
        eos_token_id=eos_token_id,
    )
    generated_ids = prompt_ids + continuation_ids

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = out_dir / f"xlstm-generated-{timestamp}.mid"
    try:
        decode_ids_to_midi(loaded.tokenizer, generated_ids, output_path)
    except Exception as exc:
        raise RuntimeError(
            "Generated tokens could not be decoded to MIDI. "
            "Use a longer generation length or provide a MIDI prompt."
        ) from exc

    status = (
        f"Generated {len(continuation_ids)} new tokens from "
        f"{len(prompt_ids)} prompt tokens on {loaded.adapter.device}."
    )
    stats = generation_statistics(
        loaded.tokenizer,
        prompt_ids,
        continuation_ids,
        output_path,
        special_tokens_blocked=bool(block_special_tokens),
        forbidden_token_ids=forbidden_token_ids,
        eos_token_id=eos_token_id,
        sampling={
            "greedy": bool(greedy),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "min_new_tokens": min_new_tokens,
        },
    )
    return (
        str(output_path),
        status,
        midi_player_html(str(output_path), title="Generated"),
        stats,
    )


def create_demo(args: argparse.Namespace):
    import gradio as gr

    run_choices = discover_runs()
    default_run_path = _run_dir_for_checkpoint(Path(args.checkpoint)) if args.checkpoint else None
    if default_run_path is None or not default_run_path.is_dir():
        default_run_path = _run_dir_for_checkpoint(Path(DEFAULT_CHECKPOINT))
    default_run = (
        _display_path(default_run_path)
        if default_run_path is not None and default_run_path.is_dir()
        else (run_choices[0].label if run_choices else "")
    )
    default_run_dir = _resolve_run_path(default_run)
    default_non_best_checkpoints = [
        selection.label
        for selection in list_run_checkpoints(default_run_dir)
        if selection.sort_key[0] != 0
    ]
    default_checkpoint_choice = (
        default_non_best_checkpoints[-1] if default_non_best_checkpoints else None
    )
    default_checkpoint = _best_checkpoint_path(default_run_dir)
    default_model_cfg = _resolve_model_cfg(default_run_dir, args.model_cfg)
    default_tokenizer_cfg = _resolve_tokenizer_cfg(default_run_dir, args.tok_cfg)
    default_summary = _summarize_run_config(
        default_run_dir,
        default_model_cfg,
        default_tokenizer_cfg,
    )

    with gr.Blocks(title="xLSTM MIDI Generator") as demo:
        gr.Markdown("# xLSTM MIDI Generator")
        with gr.Row():
            with gr.Column(scale=1):
                run = gr.Dropdown(
                    choices=[selection.label for selection in run_choices],
                    value=default_run,
                    allow_custom_value=True,
                    label="Run",
                )
                use_best_checkpoint = gr.Checkbox(
                    value=True,
                    label="Use best checkpoint",
                )
                checkpoint_choice = gr.Dropdown(
                    choices=default_non_best_checkpoints,
                    value=default_checkpoint_choice,
                    interactive=False,
                    label="Step or final checkpoint",
                )
                checkpoint = gr.Textbox(
                    value=default_checkpoint,
                    label="Resolved checkpoint",
                    interactive=False,
                )
                model_cfg = gr.Textbox(
                    value=default_model_cfg,
                    label="Resolved model config",
                    interactive=False,
                )
                tokenizer_cfg = gr.Textbox(
                    value=default_tokenizer_cfg,
                    label="Resolved tokenizer config",
                    interactive=False,
                )
                config_summary = gr.Textbox(
                    value=default_summary,
                    label="Selected run details",
                    lines=8,
                    interactive=False,
                )
                device = gr.Radio(
                    choices=["auto", "cuda", "cpu"],
                    value=args.device,
                    label="Device",
                )
                output_dir = gr.Textbox(value=args.output_dir, label="Output directory")
            with gr.Column(scale=1):
                prompt = gr.File(
                    label="Prompt MIDI",
                    file_types=[".mid", ".midi"],
                    type="filepath",
                )
                prompt_tokens = gr.Slider(
                    minimum=1,
                    maximum=1024,
                    value=128,
                    step=1,
                    label="Prompt tokens",
                )
                max_new_tokens = gr.Slider(
                    minimum=1,
                    maximum=2048,
                    value=512,
                    step=1,
                    label="New tokens",
                )
                seed = gr.Number(value=42, precision=0, label="Seed")
                preview_prompt = gr.Button("Preview Prompt")
                prompt_preview = gr.HTML(label="Prompt MIDI player")

        with gr.Row():
            greedy = gr.Checkbox(value=False, label="Greedy decoding")
            block_special_tokens = gr.Checkbox(
                value=True,
                label="Block special tokens",
            )
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
            top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k")
            top_p = gr.Slider(0.05, 1.0, value=0.95, step=0.01, label="Top-p")

        generate = gr.Button("Generate", variant="primary")
        generated_file = gr.File(label="Generated MIDI")
        generated_preview = gr.HTML(label="Generated MIDI player")
        status = gr.Textbox(label="Status")
        generation_stats = gr.Textbox(
            label="Generation statistics",
            lines=24,
            interactive=False,
        )

        run.change(
            fn=update_run_selection,
            inputs=[run, use_best_checkpoint, gr.State(args.model_cfg), gr.State(args.tok_cfg)],
            outputs=[checkpoint_choice, checkpoint, model_cfg, tokenizer_cfg, config_summary],
        )
        use_best_checkpoint.change(
            fn=update_checkpoint_selection,
            inputs=[run, use_best_checkpoint, checkpoint_choice],
            outputs=[checkpoint_choice, checkpoint],
        )
        checkpoint_choice.change(
            fn=update_checkpoint_selection,
            inputs=[run, use_best_checkpoint, checkpoint_choice],
            outputs=[checkpoint_choice, checkpoint],
        )
        preview_prompt.click(
            fn=preview_midi,
            inputs=[prompt],
            outputs=[prompt_preview],
        )

        generate.click(
            fn=generate_music,
            inputs=[
                checkpoint,
                model_cfg,
                tokenizer_cfg,
                prompt,
                prompt_tokens,
                max_new_tokens,
                greedy,
                temperature,
                top_k,
                top_p,
                block_special_tokens,
                seed,
                device,
                output_dir,
            ],
            outputs=[generated_file, status, generated_preview, generation_stats],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio UI for xLSTM MIDI generation.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model_cfg", default=DEFAULT_MODEL_CFG)
    parser.add_argument("--tok_cfg", default=DEFAULT_TOKENIZER_CFG)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--soundfont",
        default=DEFAULT_SOUNDFONT,
        help="Deprecated no-op kept for compatibility; MIDI playback runs in the browser.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = create_demo(args)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        head=MIDI_PLAYER_HEAD,
        allowed_paths=[str(Path.cwd().resolve()), "/tmp"],
    )


if __name__ == "__main__":
    main()
