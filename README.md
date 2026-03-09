# autoresearch

![teaser](progress.png)

Neural networks are just matrix multiplications stacked on top of each other, and somehow they work. The research process that discovers better ways to stack them is itself a loop — propose, evaluate, keep or discard. This repo automates that loop.

You don't need a datacenter. You don't need API credits. You need a GPU, a free evening, and the willingness to let a machine run experiments while you sleep.

autoresearch edits `train.py`, trains for exactly 5 minutes, reads `val_bpb`, keeps what improved, reverts what didn't. Repeat 50 times overnight. Wake up to a better model.

This fork runs the entire loop on a consumer NVIDIA GPU. RTX 3060, 4060, 4070 — laptop or desktop, 8GB VRAM. The research agent is a local Ollama model. No cloud. No fees. Completely offline.

---

## The Loop

```python
while True:
    propose change to train.py      # local Ollama model reads history, proposes one edit
    train for exactly 5 minutes     # uv run train.py
    if val_bpb improved:
        keep it                     # save to train.py.best
    else:
        revert                      # restore previous version
    log result
```

That is the entire algorithm. The power is in the repetition. Each experiment is 5 minutes. Overnight you get 50 of them. The agent builds up a history of what worked and what didn't, and its proposals get progressively more targeted.

The metric is `val_bpb` — validation bits per byte. Lower is better. It is vocabulary-size-independent, which means you can fairly compare a model with 3 attention heads against one with 6, or GELU against SiLU, without the number being contaminated by tokenizer choices.

---

## This Fork: Consumer GPUs

The original autoresearch assumed datacenter hardware. On an H100 you have 80GB of VRAM and memory is rarely the constraint. On a laptop RTX 4060 you have 8GB, and Ollama already uses 5 of them.

This fork makes it work anyway. Four changes to the defaults:

- `DEPTH = 4` — cuts parameters by roughly 4x. The model fits in ~3GB with room to breathe.
- `MAX_SEQ_LEN = 512` — enough context for meaningful language modeling, not enough to OOM.
- `TOTAL_BATCH_SIZE = 2**14` — ~16K tokens per step, tuned for the 3GB training footprint.
- `WINDOW_PATTERN = "L"` — disables banded attention. Ampere and Ada handle sliding-window attention poorly at small batch sizes.

Ollama takes ~5GB. Training takes ~3GB. On 8GB they coexist. Barely, but reliably.

If you have any modern NVIDIA laptop or desktop GPU, this works. See `GUIDE.md` for the exact setup steps.

---

## Files

| File | Purpose |
|---|---|
| `train.py` | Model, optimizer, training loop — mutated by the agent each experiment |
| `prepare.py` | One-time data download and BPE tokenizer training |
| `program.md` | Research policy: constraints, metric, what to try — fed to the agent each iteration |
| `ollama_agent.py` | The autonomous loop: proposes, runs, evaluates, keeps or reverts |
| `GUIDE.md` | Complete setup guide for RTX 4060 8GB + Ollama |

---

## Setup and Running

You need three things installed before anything else: Python 3.10+, Ollama, and uv.

**Step 1 — Install Ollama**

```bash
# Linux / WSL
curl -fsSL https://ollama.com/install.sh | sh

# Windows (PowerShell, run as Administrator)
winget install Ollama.Ollama

# macOS
brew install ollama
```

After installing, verify it works:

```bash
ollama --version
```

**Step 2 — Install uv**

`uv` is a fast Python package manager. It replaces pip/venv for this project.

```bash
# Linux / macOS / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installing, then verify:

```bash
uv --version
```

**Step 3 — Clone and install**

```bash
git clone https://github.com/nileshsarkarRA/autoresearcher.git
cd autoresearcher/autoresearch

uv sync
```

**Step 4 — Download data and train tokenizer (one-time, ~2 min)**

```bash
uv run prepare.py
```

This downloads the training corpus and builds a BPE tokenizer. You only do this once.

**Step 5 — Pull the coding model**

```bash
ollama pull qwen2.5-coder:7b
```

~4.5GB download. This is the model that will propose code changes.

**Step 6 — Verify a single training run**

```bash
# Terminal 1 — start Ollama server (keep this running)
ollama serve

# Terminal 2 — run one training experiment manually
uv run train.py
```

The run takes about 5 minutes. At the end you should see something like:

```
val_bpb=1.4823
```

That number is your baseline. Lower is better. If you see it, everything is working.

**Step 7 — Run the agent overnight**

```bash
# Terminal 1 — already running ollama serve

# Terminal 2
cd autoresearch
python ollama_agent.py --model qwen2.5-coder:7b --experiments 50
```

Walk away. Come back in 5 hours. The agent will have run 50 experiments — each one proposing a change, training for 5 minutes, measuring `val_bpb`, and deciding whether to keep or revert.

What you'll see:

```
============================================================
  AutoResearch Ollama Agent | model=qwen2.5-coder:7b | experiments=50
============================================================
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

**Step 8 — Check results in the morning**

```bash
cat agent_log.jsonl | python -c "
import json, sys
exps = [json.loads(l) for l in sys.stdin]
kept = [e for e in exps if e.get('kept')]
print(f'Experiments run:   {len(exps)}')
print(f'Improvements kept: {len(kept)}')
if kept:
    best = min(e['best_bpb'] for e in kept if 'best_bpb' in e)
    print(f'Best val_bpb:      {best:.4f}')
"
```

The best `train.py` is automatically saved to `train.py.best` and restored at the end of the run. Your original is preserved in `train.py.baseline`.

---

## Agent Options

```bash
# Run more experiments for a longer overnight session
python ollama_agent.py --experiments 100

# Use a smaller model if VRAM is tight
python ollama_agent.py --model llama3.2:3b --experiments 50

# Use a remote Ollama instance
python ollama_agent.py --ollama-url http://192.168.1.10:11434 --experiments 50
```

---

## Model Choices for 8GB VRAM

| Model | VRAM | Code Quality |
|---|---|---|
| `qwen2.5-coder:7b` | ~5GB | Excellent — recommended |
| `deepseek-coder:6.7b` | ~4.5GB | Excellent |
| `codellama:7b` | ~4GB | Good |
| `llama3.2:3b` | ~2GB | OK — use if VRAM is tight |

---

## Notable Forks

- https://github.com/miolini/autoresearch-macos
- https://github.com/trevin-creator/autoresearch-mlx
- https://github.com/jsegov/autoresearch-win-rtx

---

## License

MIT
