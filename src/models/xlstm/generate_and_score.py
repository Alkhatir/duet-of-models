from __future__ import annotations

from pathlib import Path

import torch
from dacite import from_dict
from omegaconf import OmegaConf
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

from src.evaluation.generate_and_score import (
    BaseModelAdapter,
    run_generation_evaluation,
)


class XLSTMAdapter(BaseModelAdapter):
    def __init__(
        self,
        checkpoint_path: Path,
        model_cfg_path: Path,
        tokenizer_vocab_size: int,
        device: torch.device,
    ) -> None:
        super().__init__(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "config" in checkpoint and "model" in checkpoint["config"]:
            cfg = OmegaConf.create(checkpoint["config"])
            model_cfg = cfg.model
        else:
            raw_cfg = OmegaConf.load(model_cfg_path)
            if "model" in raw_cfg:
                model_cfg = raw_cfg.model
            else:
                model_cfg = raw_cfg
        model_cfg = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        converted_slstm_backend = False
        if device.type == "cpu" and "slstm_block" in model_cfg:
            slstm_cfg = model_cfg.slstm_block.get("slstm", None)
            if slstm_cfg is not None and "backend" in slstm_cfg:
                converted_slstm_backend = str(slstm_cfg.backend) != "vanilla"
                slstm_cfg.backend = "vanilla"
        model_cfg.vocab_size = tokenizer_vocab_size
        model = xLSTMLMModel(
            from_dict(xLSTMLMModelConfig, OmegaConf.to_container(model_cfg, resolve=True))
        ).to(device)
        state_dict = checkpoint["model_state_dict"]
        if converted_slstm_backend:
            state_dict = {
                key: value.transpose(1, 2).contiguous()
                if key.endswith("._recurrent_kernel_") and value.ndim == 3
                else value
                for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        self._max_context_length = int(model_cfg.context_length)

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    def next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(input_ids)
        return logits[:, -1, :]


def build_adapter(
    checkpoint_path: Path,
    model_cfg_path: Path,
    tokenizer_vocab_size: int,
    device: torch.device,
) -> BaseModelAdapter:
    return XLSTMAdapter(
        checkpoint_path=checkpoint_path,
        model_cfg_path=model_cfg_path,
        tokenizer_vocab_size=tokenizer_vocab_size,
        device=device,
    )


def main() -> None:
    run_generation_evaluation(
        model_type="xlstm",
        adapter_factory=build_adapter,
    )


if __name__ == "__main__":
    main()
