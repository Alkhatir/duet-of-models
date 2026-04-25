# xLSTM Model Configs

These configs define a simple xLSTM scaling ladder for symbolic MIDI modeling.

The rule is:

- keep the general recipe fixed
- scale mainly with `embedding_dim` and `num_blocks`
- increase `num_heads` only when width makes it natural
- increase the number of `sLSTM` blocks gradually instead of changing the architecture wholesale

In this repo, `vocab_size` and `context_length` are set at runtime by the training code, so they are intentionally omitted here.

Available configs:

- `basic.yaml`: compatibility baseline, same architecture as `tiny` with CUDA sLSTM backend
- `basic_cpu.yaml`: compatibility baseline, same architecture as `tiny` with vanilla sLSTM backend
- `small.yaml`: first upgrade over the baseline
- `base.yaml`: balanced default for serious experiments
- `medium.yaml`: larger run for checking whether the smaller models are still underfitting

Recommended order:

1. `basic.yaml` or `basic_cpu.yaml`
2. `small.yaml`
3. `base.yaml`
4. `medium.yaml`

Approximate parameter counts with this repo's current tokenizer setup are:

- `basic`: ~0.64M
- `small`: ~1.70M
- `base`: ~3.77M
- `medium`: ~9.39M

Use `medium` only after you have evidence that `base` is still underfitting relative to your token budget and hardware.
