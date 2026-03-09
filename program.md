# AutoResearch Program — RTX 4060 Laptop (8GB VRAM)

## Hardware

- GPU: NVIDIA RTX 4060 Laptop, 8GB VRAM
- RAM: 16GB
- Training budget: exactly 5 minutes wall-clock time
- VRAM is the binding constraint — do not increase model size, depth, or sequence length

## Hard Constraints (do not change these)

- `DEPTH = 4` — model is already reduced to fit the GPU
- `MAX_SEQ_LEN = 512` — sequence length reduced for memory
- `TOTAL_BATCH_SIZE = 2**14` — tuned for ~3GB training footprint
- `WINDOW_PATTERN = "L"` — banded attention is off (slow on consumer cards)
- Do not add new pip dependencies

## Metric

`val_bpb` (validation bits per byte) — lower is better. Vocabulary-size-independent, so architectural changes are fairly compared. Never remove the logging for this metric.

## What to Try

Everything here targets improving sample efficiency in the 5-minute window:

1. **Optimizer** — peak learning rate, warmup steps, decay shape, betas, epsilon, weight decay
2. **Regularization** — dropout rates, gradient clipping threshold
3. **Architecture** — attention head count, FFN multiplier, norm placement, residual scaling
4. **Training dynamics** — gradient accumulation steps
5. **Activations** — GELU vs SiLU vs ReLU variants

## What Has Been Tried

(filled in by experiment history)

## Constraints for Agent

- Single GPU, no distributed training
- The Muon + AdamW combo is already wired up — tune its parameters, don't replace it
- One variable at a time — small targeted changes are easier to attribute
- OOM = crash = revert — if unsure, go smaller
