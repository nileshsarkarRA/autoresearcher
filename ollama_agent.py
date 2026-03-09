#!/usr/bin/env python3
"""
ollama_agent.py — Drop-in replacement for Claude Code in Karpathy's autoresearcher.
Uses a local Ollama model to autonomously iterate on train.py overnight.

Usage:
    python ollama_agent.py
    python ollama_agent.py --model qwen2.5-coder:7b --experiments 50
    python ollama_agent.py --model deepseek-coder:6.7b --experiments 20 --ollama-url http://localhost:11434

Requirements:
    pip install requests
    ollama pull qwen2.5-coder:7b
    ollama serve   (in a separate terminal)
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# -- Config -------------------------------------------------------------------

DEFAULT_MODEL    = "qwen2.5-coder:7b"
TRAIN_FILE       = Path("train.py")
PROGRAM_FILE     = Path("program.md")
BACKUP_FILE      = Path("train.py.best")
BASELINE_FILE    = Path("train.py.baseline")
LOG_FILE         = Path("agent_log.jsonl")
OLLAMA_URL       = "http://localhost:11434"
TRAIN_TIMEOUT    = 2100   # 35 min hard timeout (30 min budget + startup overhead)

# -- Logging ------------------------------------------------------------------

COLORS = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
    "reset":  "\033[0m",
}

def log(msg: str, color: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    c  = COLORS.get(color, "")
    r  = COLORS["reset"] if color else ""
    print(f"{c}[{ts}] {msg}{r}", flush=True)


def banner(text: str):
    print(f"\n{COLORS['bold']}{'='*60}{COLORS['reset']}", flush=True)
    print(f"{COLORS['bold']}  {text}{COLORS['reset']}", flush=True)
    print(f"{COLORS['bold']}{'='*60}{COLORS['reset']}\n", flush=True)


# -- Ollama API ---------------------------------------------------------------

def check_ollama(base_url: str, model: str):
    """Ensure Ollama is running and the chosen model is available."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        available = [m["name"] for m in r.json().get("models", [])]
        short = {m.split(":")[0] for m in available}
        if model not in available and model.split(":")[0] not in short:
            log(f"Model '{model}' not found locally. Pulling now (this may take a few minutes)...", "yellow")
            subprocess.run(["ollama", "pull", model], check=True)
            log(f"Model '{model}' pulled.", "green")
        else:
            log(f"Model '{model}' is ready.", "green")
    except requests.exceptions.ConnectionError:
        log("Cannot connect to Ollama. Start it with: ollama serve", "red")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        log(f"Failed to pull model: {e}", "red")
        sys.exit(1)


def ollama_chat(base_url: str, model: str, messages: list[dict], temperature: float = 0.7) -> str:
    """Send messages to Ollama and return the assistant's response text."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 8192,
            "num_predict": 4096,
        },
    }
    resp = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# -- Code Extraction ----------------------------------------------------------

def extract_python_block(text: str) -> str | None:
    """Extract the first complete Python code block from model output."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if len(code) > 100:
            return code

    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "import" in code and len(code) > 100:
            return code

    if text.count("def ") >= 2 and text.count("import") >= 2:
        return text.strip()

    return None


def is_valid_python(code: str) -> bool:
    """Quick syntax check on proposed code."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError as e:
        log(f"Syntax error in proposed code: {e}", "red")
        return False


# -- Training -----------------------------------------------------------------

def parse_val_bpb(output: str) -> float | None:
    """Extract val_bpb from training output."""
    matches = re.findall(r"val[_\s]bpb[\s:=]+([0-9]+\.[0-9]+)", output, re.IGNORECASE)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    return None


def run_training() -> tuple[float | None, str]:
    """Run `uv run train.py` and return (val_bpb or None, stdout+stderr)."""
    log("Starting training run (5-min budget + startup time)...", "cyan")
    start = time.time()
    try:
        result = subprocess.run(
            ["uv", "run", "train.py"],
            capture_output=True,
            text=True,
            timeout=TRAIN_TIMEOUT,
        )
        elapsed = time.time() - start
        output = result.stdout + result.stderr
        bpb = parse_val_bpb(output)
        if bpb:
            log(f"Training finished in {elapsed/60:.1f}min — val_bpb = {bpb:.4f}", "cyan")
        else:
            log(f"Training finished in {elapsed/60:.1f}min but no val_bpb found in output", "yellow")
            tail = "\n".join(output.splitlines()[-20:])
            log(f"Output tail:\n{tail}", "dim")
        return bpb, output
    except subprocess.TimeoutExpired:
        log(f"Training exceeded {TRAIN_TIMEOUT/60:.0f}min hard timeout.", "red")
        return None, "TIMEOUT"
    except FileNotFoundError:
        log("'uv' not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh", "red")
        return None, "UV_NOT_FOUND"
    except Exception as e:
        log(f"Training error: {e}", "red")
        return None, str(e)


# -- Result Logging -----------------------------------------------------------

def save_to_log(entry: dict):
    entry["timestamp"] = datetime.now().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def print_summary(history: list[dict], best_bpb: float, baseline_bpb: float):
    """Print a summary table of all experiments."""
    banner("SESSION SUMMARY")
    total    = len(history)
    kept     = sum(1 for h in history if h.get("kept"))
    crashed  = sum(1 for h in history if h.get("bpb") is None)
    improved = baseline_bpb - best_bpb

    log(f"Experiments run:   {total}", "bold")
    log(f"Improvements kept: {kept}", "green")
    log(f"Reverted:          {total - kept - crashed}", "yellow")
    log(f"Crashed/invalid:   {crashed}", "red")
    log(f"Baseline val_bpb:  {baseline_bpb:.4f}", "bold")
    log(f"Best val_bpb:      {best_bpb:.4f}", "green")
    log(f"Total improvement: {improved:+.4f} ({improved/baseline_bpb*100:+.1f}%)",
        "green" if improved > 0 else "yellow")
    print()


# -- Prompts ------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert ML research agent performing autonomous hyperparameter and architecture search.

Your goal: minimize val_bpb (validation bits per byte) on a language model training setup.
Lower val_bpb = better model. This runs on an RTX 4060 laptop with 8GB VRAM.

Your rules:
1. Make ONE focused, targeted change per experiment. Small changes are easier to reason about.
2. Prioritize changes with a clear theoretical basis.
3. Never increase DEPTH, MAX_SEQ_LEN, or model size — the GPU cannot handle it.
4. Never remove the val_bpb logging — the harness needs it to evaluate experiments.
5. Your response must contain EXACTLY ONE complete ```python ... ``` block with the FULL train.py file.
6. Before the code block, write 2-3 sentences explaining your hypothesis for this change.
7. Never add new dependencies (no new imports that are not already in pyproject.toml).

Good ideas to try (in rough priority order):
- Learning rate schedule tuning (peak lr, warmup steps, decay shape)
- Optimizer parameter tuning (betas, epsilon, weight decay)
- Gradient clipping threshold
- Layer norm placement (pre-norm vs post-norm)
- Residual scaling
- Attention head count / key-value dimensions
- Embedding dropout or attention dropout
- Activation function (GELU vs SiLU etc.)
"""


def build_user_prompt(program_md: str, train_py: str, history: list[dict]) -> str:
    history_block = ""
    if history:
        history_block = "\n\n## Experiment History (most recent last)\n"
        for h in history[-8:]:
            kept_str = "KEPT" if h.get("kept") else "REVERTED"
            bpb_str  = f"{h['bpb']:.4f}" if h.get("bpb") else "CRASHED"
            best_str = f"(best now: {h['best_bpb']:.4f})" if h.get("best_bpb") else ""
            desc     = h.get("description", "")[:120]
            history_block += f"- Exp {h['exp']}: val_bpb={bpb_str} {kept_str} {best_str} — {desc}\n"

    return f"""## Research Program

{program_md}

## Current train.py

```python
{train_py}
```
{history_block}

Based on the history above, propose your next experiment. Remember:
- ONE change only
- State your hypothesis first (2-3 sentences)
- Then provide the complete modified train.py in a ```python ... ``` block
"""


# -- Main Agent Loop ----------------------------------------------------------

def run_agent(model: str, base_url: str, n_experiments: int):
    banner(f"AutoResearcher Ollama Agent | model={model} | experiments={n_experiments}")
    check_ollama(base_url, model)

    for fname in [TRAIN_FILE, PROGRAM_FILE]:
        if not fname.exists():
            log(f"{fname} not found. Are you in the autoresearcher directory?", "red")
            sys.exit(1)

    shutil.copy(TRAIN_FILE, BASELINE_FILE)

    log("Running baseline experiment to establish starting val_bpb...", "cyan")
    baseline_bpb, baseline_output = run_training()

    if baseline_bpb is None:
        log("Baseline run failed. Fix train.py before running the agent.", "red")
        print(baseline_output[-2000:])
        sys.exit(1)

    log(f"Baseline val_bpb = {baseline_bpb:.4f}", "green")
    best_bpb = baseline_bpb
    shutil.copy(TRAIN_FILE, BACKUP_FILE)
    save_to_log({"exp": 0, "type": "baseline", "bpb": baseline_bpb})

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    history:  list[dict] = []

    for exp_num in range(1, n_experiments + 1):
        banner(f"Experiment {exp_num}/{n_experiments} | Best val_bpb = {best_bpb:.4f}")

        program_md = PROGRAM_FILE.read_text()
        train_py   = TRAIN_FILE.read_text()

        log(f"Querying {model}...", "cyan")
        user_msg = build_user_prompt(program_md, train_py, history)
        messages.append({"role": "user", "content": user_msg})

        try:
            response = ollama_chat(base_url, model, messages, temperature=0.7)
        except requests.exceptions.Timeout:
            log("Ollama timed out generating response. Skipping experiment.", "red")
            messages.pop()
            continue
        except Exception as e:
            log(f"Ollama API error: {e}", "red")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": response})

        desc_lines = [
            l.strip() for l in response.split("\n")
            if l.strip() and not l.startswith("```") and not l.startswith("#")
        ]
        description = " ".join(desc_lines[:3])[:250]
        log(f"Proposal: {description}", "yellow")

        new_code = extract_python_block(response)

        if not new_code:
            log("No valid Python code block found. Asking model to retry...", "red")
            retry_msg = (
                "Your response did not contain a valid ```python ... ``` code block. "
                "Please respond again with ONLY a 2-sentence hypothesis followed by "
                "the complete modified train.py inside a ```python ... ``` block."
            )
            messages.append({"role": "user", "content": retry_msg})
            try:
                response = ollama_chat(base_url, model, messages, temperature=0.5)
                messages.append({"role": "assistant", "content": response})
                new_code = extract_python_block(response)
            except Exception:
                pass

        if not new_code:
            log("Still no code block after retry. Skipping experiment.", "red")
            history.append({"exp": exp_num, "bpb": None, "best_bpb": best_bpb,
                            "kept": False, "description": "NO CODE BLOCK"})
            save_to_log(history[-1])
            continue

        if not is_valid_python(new_code):
            log("Code has syntax errors. Reverting and skipping.", "red")
            feedback = "Your code had Python syntax errors and was rejected. Try a simpler change."
            messages.append({"role": "user", "content": feedback})
            history.append({"exp": exp_num, "bpb": None, "best_bpb": best_bpb,
                            "kept": False, "description": "SYNTAX ERROR"})
            save_to_log(history[-1])
            continue

        TRAIN_FILE.write_text(new_code)

        bpb, output = run_training()

        if bpb is not None:
            delta = best_bpb - bpb
            if bpb < best_bpb:
                best_bpb = bpb
                shutil.copy(TRAIN_FILE, BACKUP_FILE)
                kept = True
                log(f"IMPROVEMENT. val_bpb={bpb:.4f} (delta={delta:+.4f}) — keeping change.", "green")
                feedback = (
                    f"Result: val_bpb={bpb:.4f} — IMPROVEMENT of {delta:.4f}. "
                    f"Change KEPT. New best: {best_bpb:.4f}. "
                    "Build on this or try a different direction."
                )
            else:
                kept = False
                shutil.copy(BACKUP_FILE, TRAIN_FILE)
                log(f"No improvement. val_bpb={bpb:.4f} vs best={best_bpb:.4f} — reverting.", "red")
                feedback = (
                    f"Result: val_bpb={bpb:.4f} — WORSE than best ({best_bpb:.4f}). "
                    "Change REVERTED. Try a different approach next time."
                )
        else:
            kept = False
            shutil.copy(BACKUP_FILE, TRAIN_FILE)
            tail = output[-600:] if len(output) > 600 else output
            log("Training crashed or no val_bpb produced — reverting.", "red")
            feedback = (
                f"Training CRASHED or produced no val_bpb metric. Change REVERTED. "
                f"Error output tail:\n{tail}\n"
                "Make sure val_bpb is still being logged and no new dependencies were added."
            )

        messages.append({"role": "user", "content": feedback})

        entry = {
            "exp":         exp_num,
            "bpb":         bpb,
            "best_bpb":    best_bpb,
            "kept":        kept,
            "description": description,
        }
        history.append(entry)
        save_to_log(entry)

        # Keep system prompt + last 16 messages (8 turns) to avoid context blowout
        if len(messages) > 18:
            messages = messages[:1] + messages[-16:]

    print_summary(history, best_bpb, baseline_bpb)
    log(f"Best train.py saved to: {BACKUP_FILE}", "green")
    log(f"Experiment log saved to: {LOG_FILE}", "green")
    shutil.copy(BACKUP_FILE, TRAIN_FILE)
    log("train.py restored to best version.", "green")


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ollama-powered AutoResearcher agent for nileshsarkarRA/autoresearcher"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--experiments", default=20, type=int,
        help="Number of experiments to run (default: 20, use 50 for overnight)"
    )
    parser.add_argument(
        "--ollama-url", default=OLLAMA_URL,
        help=f"Ollama server URL (default: {OLLAMA_URL})"
    )
    args = parser.parse_args()

    run_agent(
        model         = args.model,
        base_url      = args.ollama_url,
        n_experiments = args.experiments,
    )
