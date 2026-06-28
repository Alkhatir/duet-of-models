from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from miditok import TokSequence
from omegaconf import OmegaConf

from src.data.tokenizer import MidiTokBuilder
from src.evaluation.generate_and_score import (
    XLSTMAdapter,
    decode_ids_to_midi,
    generate_continuation,
)


DEFAULT_CHECKPOINT = "experiments/xlstm-small-batch8-tok11/best/checkpoint.pt"
DEFAULT_MODEL_CFG = "configs/model/xlstm/small.yaml"
DEFAULT_TOKENIZER_CFG = "configs/data/11.yaml"
DEFAULT_OUTPUT_DIR = "experiments/gradio_xlstm_generations"


@dataclass(frozen=True)
class LoadedXLSTM:
    checkpoint_path: Path
    model_cfg_path: Path
    tokenizer_cfg_path: Path
    device_name: str
    tokenizer: Any
    adapter: XLSTMAdapter


_MODEL_CACHE: LoadedXLSTM | None = None


def discover_checkpoints(root: Path = Path("experiments")) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path) for path in root.glob("**/checkpoint.pt"))


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
) -> Any:
    return OmegaConf.create(
        {
            "sampling": {
                "greedy": bool(greedy),
                "temperature": float(temperature),
                "top_k": int(top_k),
                "top_p": float(top_p),
            }
        }
    )


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
    seed: int,
    device_name: str,
    output_dir: str,
) -> tuple[str, str]:
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
    sampling_cfg = build_sampling_config(
        greedy=greedy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
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
    return str(output_path), status


def create_demo(args: argparse.Namespace):
    import gradio as gr

    checkpoint_choices = discover_checkpoints()
    default_checkpoint = (
        args.checkpoint
        or (DEFAULT_CHECKPOINT if Path(DEFAULT_CHECKPOINT).is_file() else None)
        or (checkpoint_choices[0] if checkpoint_choices else "")
    )

    with gr.Blocks(title="xLSTM MIDI Generator") as demo:
        gr.Markdown("# xLSTM MIDI Generator")
        with gr.Row():
            with gr.Column(scale=1):
                checkpoint = gr.Dropdown(
                    choices=checkpoint_choices,
                    value=default_checkpoint,
                    allow_custom_value=True,
                    label="Checkpoint",
                )
                model_cfg = gr.Textbox(value=args.model_cfg, label="Model config")
                tokenizer_cfg = gr.Textbox(value=args.tok_cfg, label="Tokenizer config")
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

        with gr.Row():
            greedy = gr.Checkbox(value=False, label="Greedy decoding")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
            top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k")
            top_p = gr.Slider(0.05, 1.0, value=0.95, step=0.01, label="Top-p")

        generate = gr.Button("Generate", variant="primary")
        generated_file = gr.File(label="Generated MIDI")
        status = gr.Textbox(label="Status")

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
                seed,
                device,
                output_dir,
            ],
            outputs=[generated_file, status],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio UI for xLSTM MIDI generation.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model_cfg", default=DEFAULT_MODEL_CFG)
    parser.add_argument("--tok_cfg", default=DEFAULT_TOKENIZER_CFG)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = create_demo(args)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
