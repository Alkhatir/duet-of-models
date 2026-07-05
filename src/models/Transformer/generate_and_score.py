from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from src.evaluation.generate_and_score import (
    BaseModelAdapter,
    run_generation_evaluation,
)


class TransformerAdapter(BaseModelAdapter):
    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        super().__init__(device)
        self.model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path)).to(device)
        self.model.eval()
        config = self.model.config
        self._max_context_length = int(
            getattr(config, "max_position_embeddings", None)
            or getattr(config, "n_positions", None)
            or getattr(config, "n_ctx", 1024)
        )

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    def next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        return outputs.logits[:, -1, :]


def build_adapter(
    checkpoint_path: Path,
    model_cfg_path: Path,
    tokenizer_vocab_size: int,
    device: torch.device,
) -> BaseModelAdapter:
    del model_cfg_path, tokenizer_vocab_size
    return TransformerAdapter(checkpoint_path=checkpoint_path, device=device)


def main() -> None:
    run_generation_evaluation(
        model_type="transformer",
        adapter_factory=build_adapter,
    )


if __name__ == "__main__":
    main()
