# AutoResearch — Consumer GPU Setup Guide

You have an RTX 4060 (or similar 8GB card). This guide gets you from zero to a running overnight research loop in under 30 minutes.

The original autoresearch was written for datacenter GPUs. These settings make it work on yours.

---

## Prerequisites

### 1. uv — fast Python package manager

```bash
# Linux / WSL2 / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Ollama — local model runner

```bash
# Linux / WSL2
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download installer from https://ollama.com/download/windows
```

### 3. Verify your GPU is seen

```bash
nvidia-smi  # should show your card and CUDA version
```

---

## Step 1 — Clone and Install

```bash
git clone https://github.com/karpathy/autoresearch
cd autoresearch
uv sync
```

---

## Step 2 — Tune for Consumer VRAM

Two files need edits before anything runs. These are the numbers that make an 8GB card viable.

### `prepare.py`

```python
MAX_SEQ_LEN = 512       # default is ~2048 — this halves memory usage
EVAL_TOKENS = 262144    # 2^18 — fast enough eval without sacrificing signal
```

### `train.py`

```python
DEPTH = 4               # default is 8 — this makes the model fit
TOTAL_BATCH_SIZE = 2**14   # ~16K tokens — tuned for 8GB
WINDOW_PATTERN = "L"    # disables banded attention — it's slow on consumer cards
DEVICE_BATCH_SIZE = 4   # per-GPU batch; lower to 2 if you hit OOM
```

Why these numbers: `DEPTH=4` cuts parameters by roughly 4x. `WINDOW_PATTERN="L"` removes a fancy sliding-window attention pattern that the Ampere/Ada architectures handle poorly at small batch sizes. The result is a model that genuinely improves in 5 minutes instead of barely moving.

---

## Step 3 — Prepare Data (one time, ~2 min)

```bash
uv run prepare.py
```

Downloads training data and trains a BPE tokenizer. Run once. Never again.

---

## Step 4 — Verify a Single Run

```bash
uv run train.py
```

Should take ~5 minutes and end with `val_bpb=X.XXXX`. That number is your baseline. The agent will spend the night trying to lower it.

**OOM?** Lower `DEVICE_BATCH_SIZE` from 4 to 2 in `train.py`.

**CUDA not found?**

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Step 5 — Pull a Model

Ollama needs ~5GB VRAM. Training needs ~3GB. On 8GB they share the card, barely.

```bash
# Recommended — best code quality for the VRAM cost
ollama pull qwen2.5-coder:7b

# Alternatives
ollama pull deepseek-coder:6.7b   # comparable quality, slightly lighter
ollama pull llama3.2:3b            # use this if VRAM is too tight
```

| Model | VRAM | Code Quality |
|---|---|---|
| `qwen2.5-coder:7b` | ~5GB | Excellent — recommended |
| `deepseek-coder:6.7b` | ~4.5GB | Excellent |
| `codellama:7b` | ~4GB | Good |
| `llama3.2:3b` | ~2GB | OK |

---

## Step 6 — Run the Agent

```bash
# Terminal 1 — keep alive the whole time
ollama serve

# Terminal 2
cd autoresearch

# Quick test (5 experiments, ~30 min)
python ollama_agent.py --model qwen2.5-coder:7b --experiments 5

# Overnight run (50 experiments, ~5-6 hours)
python ollama_agent.py --model qwen2.5-coder:7b --experiments 50
```

What you'll see:

```
[22:01:05] AutoResearch Ollama Agent | model=qwen2.5-coder:7b | experiments=50
[22:01:08] Model 'qwen2.5-coder:7b' is ready.
[22:01:08] Running baseline experiment to establish starting val_bpb...
[22:06:12] Training finished in 5.1min -- val_bpb = 1.4823
============================================================
  Experiment 1/50 | Best val_bpb = 1.4823
============================================================
[22:06:14] Querying qwen2.5-coder:7b...
[22:06:19] Proposal: Warmup steps are too short for this batch size...
[22:11:22] IMPROVEMENT. val_bpb=1.4701 (delta=+0.0122) -- keeping change.
```

---

## Step 7 — Check Results in the Morning

```bash
# What improved overnight
cat agent_log.jsonl | python -c "
import json, sys
exps = [json.loads(l) for l in sys.stdin]
kept = [e for e in exps if e.get('kept')]
print(f'Experiments run:  {len(exps)}')
print(f'Improvements kept: {len(kept)}')
if kept:
    best = min(e['best_bpb'] for e in kept if 'best_bpb' in e)
    print(f'Best val_bpb:      {best:.4f}')
"
```

The best `train.py` is always saved to `train.py.best`. The agent restores it automatically at the end of the run.

---

## Troubleshooting

**OOM during training**

```python
# train.py
DEVICE_BATCH_SIZE = 2   # was 4
```

**Both Ollama and training crash (full VRAM conflict)**

Force Ollama to run on CPU for inference:

```bash
OLLAMA_NUM_GPU=0 ollama serve
```

Inference will be slower but training gets all the VRAM. Or just switch to `llama3.2:3b`.

**Agent keeps producing malformed code**

The agent auto-reverts and skips malformed experiments. If it's happening repeatedly, your context might be too long or the model is confused. Try restarting the agent — it will re-establish a baseline and continue from scratch.

**`val_bpb` not printing**

Check that `train.py` still has the `val_bpb` logging line. The agent is explicitly instructed not to remove it, but if something went wrong, compare against `train.py.best` or the baseline.

**Windows-specific issues**

Community fork with Windows fixes: https://github.com/jsegov/autoresearch-win-rtx

---

## Overnight Workflow

```bash
# Terminal 1
ollama serve

# Terminal 2 — log everything
cd autoresearch
python ollama_agent.py \
  --model qwen2.5-coder:7b \
  --experiments 50 \
  2>&1 | tee run_$(date +%Y%m%d).log
```

---

## File Reference

| File | What it is | Edit? |
|---|---|---|
| `train.py` | Model + training loop | Yes — set DEPTH, TOTAL_BATCH_SIZE, etc. Agent edits this too. |
| `prepare.py` | Data + tokenizer prep | Yes — set MAX_SEQ_LEN, EVAL_TOKENS. Run once. |
| `program.md` | Research policy for agent | Optional — controls agent strategy |
| `ollama_agent.py` | The loop | Yes — change `--model`, `--experiments` |
| `train.py.best` | Best found so far | No — auto-managed |
| `agent_log.jsonl` | Full experiment history | No — auto-managed |
