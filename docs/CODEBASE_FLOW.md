# AutoResearcher Codebase Flow Explanation

## 🎯 Overview
When you run `./autoresearcher`, it executes a complete **4-step training pipeline** optimized for NVIDIA A100 80GB GPUs. Here's what happens, step by step, with all the files involved:

---

## 📋 Step-by-Step Execution Flow

### **STEP 0: Initial Setup & Configuration**
**File: `autoresearcher` (main bash script)**

```bash
./autoresearcher
```

When you run the script:

1. **Prints Banner** - Shows the ASCII art welcome screen
2. **Parses Arguments** - Checks for command-line flags like `--minutes`, `--hours`, `--dataset`, etc.
3. **Interactive Prompts** (if not automated):
   - Asks how long to train (5, 10, 20, 30, 60 min, or custom)
   - Asks which dataset (ClimbMix, ArXiv, Wikipedia, Code, StackExchange)
   - Asks for number of workers and shards
   - Asks if you want DeepSeek integration

**Key Variables Set:**
- `TRAINING_MINUTES`: Total training duration
- `DATASET`: Which dataset to download
- `NUM_SHARDS`: How many data shards to download (default: 10)
- `WORKERS`: Parallel workers for downloading (default: 42)
- `SETUP_DEEPSEEK`: Whether to enable DeepSeek code optimization

**Creates:**
- `logs/` directory for training logs
- `logs/training_YYYYMMDD_HHMMSS.log` file for timestamped logs

---

### **STEP 1: Hardware Verification**
**File: `setup_a100.py`**

```bash
python setup_a100.py
```

The script checks:
- ✅ Python 3.12+ installed
- ✅ PyTorch with CUDA support
- ✅ GPU available (checks NVIDIA A100, gets total memory)
- ✅ Compute capability (should be 9.0 for A100)
- ✅ CPU cores (checks for 42 cores recommended)
- ✅ System memory (256GB+ recommended)
- ✅ Disk space available

**Output:**
- Warnings if hardware doesn't meet recommendations
- Errors if critical components missing
- Success message: ✓ Hardware verification passed!

**Why this matters:** The codebase is heavily optimized for A100 hardware. This check ensures you have the right setup.

---

### **STEP 2: Data Preparation & Download**
**File: `prepare.py`**

```bash
python prepare.py --dataset climbmix --num-shards 10 --download-workers 42
```

This is where the actual data comes from:

#### **2a: Define Dataset**
The script loads a dataset configuration from `DATASETS` dict:

```python
DATASETS = {
    "climbmix": {
        "base_url": "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
        "max_shard": 6542,
    },
    "arxiv": { "base_url": "https://...", ... },
    "wiki": { ... },
    "code": { ... },
}
```

Each dataset is a collection of **parquet files** (shards) hosted on HuggingFace.

#### **2b: Download Data in Parallel**
Using `multiprocessing.Pool` with 42 workers:
- Downloads each shard (data chunk) in parallel
- Shards are saved to `~/.cache/autoresearch/data/`
- Each shard is a Parquet file containing text data

**What gets downloaded:**
- Example: `shard_00001.parquet`, `shard_00002.parquet`, ... `shard_00010.parquet`
- Each file is raw text data from your chosen source

#### **2c: Train BPE Tokenizer**
```python
# Constants from prepare.py
VOCAB_SIZE = 8192
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|..."""
```

The script trains a **Byte-Pair Encoding (BPE) tokenizer** that converts:
- Raw text → Token IDs (integers)
- Example: "Hello world" → [1502, 3829]

The tokenizer is saved to `~/.cache/autoresearch/tokenizer/`.

**Why important:** The LLM can only work with numbers, not raw text!

#### **2d: Create Data Loaders**
```python
def make_dataloader(dataset, batch_size, seq_len, num_workers):
    # Creates batches of tokenized data for training
    # Each batch: (seq_len=2048 tokens per sample)
    return dataloader
```

**Output files created:**
- `~/.cache/autoresearch/data/shard_*.parquet` (raw data)
- `~/.cache/autoresearch/tokenizer/` (BPE tokenizer)
- Data is now ready to feed into the model!

---

### **STEP 3: DeepSeek Setup (Optional)**
**File: `ollama_deepseek.py`**

If you said "yes" to DeepSeek integration:

```bash
# Check if Ollama is running
pgrep ollama || ollama serve

# Pull the DeepSeek model
ollama pull deepseek-coder:6.7b-base-q4_0
```

This downloads a **6.7 billion parameter language model** (~4.5 GB) that can:
- Generate code suggestions
- Optimize the training process
- Provide feedback every `IMPROVE_INTERVAL_MINUTES` minutes

**What Ollama does:**
- Runs as a background server on `http://localhost:11434`
- Accepts text prompts → Returns generated text
- Used to generate research ideas or code improvements

**Why optional:** DeepSeek is heavy (4.5GB), takes time to load, and isn't required for training.

---

### **STEP 4: Training Loop**
**File: `train.py`**

```bash
export TIME_BUDGET_SECONDS=600  # 10 minutes
uv run train.py
```

This is the **core ML training** - the main event!

#### **4a: Model Definition**

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048      # Context length (2K tokens = ~8KB text)
    vocab_size: int = 32768       # Number of possible tokens
    n_layer: int = 12              # 12 transformer layers
    n_head: int = 6                # 6 attention heads
    n_kv_head: int = 6             # Key-Value heads
    n_embd: int = 768              # Embedding dimension (hidden size)
    window_pattern: str = "SSSL"   # Attention window pattern
```

The model architecture:
```
Text Input (2048 tokens)
    ↓
Token Embedding (768-dim vectors)
    ↓
12 Transformer Blocks:
    - Causal Self-Attention (reads all previous tokens)
    - MLP (feed-forward network)
    - Residual connections & normalization
    ↓
Prediction Head (output vocab_size logits)
    ↓
Next Token Probability Distribution
```

#### **4b: Flash Attention 3 (FA3)**
```python
from kernels import get_kernel
fa3 = get_kernel(repo).flash_attn_interface
```

Uses **ultra-optimized CUDA kernels** for:
- Fast attention computation on A100
- Reduced memory usage
- 10-100x faster than standard PyTorch attention

This is why it's optimized for A100!

#### **4c: Training Loop**
```python
for step in range(num_steps):
    # 1. Get batch of data from dataloader
    x, y = next(train_loader)  # x: input tokens, y: target tokens
    
    # 2. Forward pass through model
    logits = model(x)          # Shape: [batch, seq_len, vocab_size]
    
    # 3. Calculate loss (cross-entropy)
    loss = F.cross_entropy(logits.flatten(0,-2), y.flatten())
    
    # 4. Backward pass (compute gradients)
    loss.backward()
    
    # 5. Optimizer step (update weights)
    optimizer.step()
    optimizer.zero_grad()
    
    # 6. Log metrics every K steps
    if step % log_interval == 0:
        print(f"step {step} | loss: {loss:.3f} | mfu: {mfu}% | tok/sec: {throughput}")
```

#### **4d: Key Metrics Logged**

```
step 00245 | loss: 3.456 | lrm: 1.00 | dt: 1000ms | tok/sec: 135,454 | mfu: 2.2% | epoch: 1 | remaining: 500s
```

- **loss**: How wrong the predictions are (lower = better). Starts ~4.0, should improve to ~2.5
- **mfu**: Model FLOPs Utilization (% of A100's peak compute used). Should be 15-30%
- **tok/sec**: Tokens per second (throughput). Higher = faster training
- **dt**: Time per step in milliseconds
- **lrm**: Learning rate multiplier (cosine annealing schedule)
- **remaining**: Wall-clock time left until `TIME_BUDGET` runs out

#### **4e: Automatic Stopping**
```bash
# Main script monitors training
while kill -0 $TRAIN_PID; do
    ELAPSED=$(($(date +%s) - TRAINING_START))
    if [ $ELAPSED -gt $TRAINING_LIMIT_SECONDS ]; then
        kill $TRAIN_PID  # Stop training at time limit
        break
    fi
    sleep 1
done
```

The bash script monitors the training process and **kills it exactly when the time limit is reached**.

#### **4f: Live Terminal Dashboard**
Every 5 seconds, the script displays:
```
╔════════════════════════════════════════════════════════════╗
║          🚀 TRAINING IN PROGRESS - LIVE METRICS 🚀         ║
╚════════════════════════════════════════════════════════════╝

⏱️  Training Progress:
  Elapsed: 2m 30s | Remaining: 7m 30s
  Progress: [##########░░░░░░░░░░] 25%

📊 Latest Training Metrics:
  step 00245 | loss: 3.456 | mfu: 2.2% | tok/sec: 135,454

📈 Detailed Metrics:
  Loss (BPB):        3.456
  MFU (%):           2.2
  LR Multiplier:     1.00
  Tokens/sec:        135,454
  Epoch:             1
```

**All logs are tee'd (piped) to:** `logs/training_YYYYMMDD_HHMMSS.log`

---

## 🎨 Post-Training: Graph Generation

**File: `autoresearcher` (lines 300-480)**

After training finishes, the script runs a **Python sub-script embedded in the bash file**:

```python
# Parse training log file
# Extract: step numbers, losses, MFU values, throughput values

# Create 4-subplot figure:
# 1. Loss Curve (top-left) - should show downward trend
# 2. Model FLOPs Utilization (top-right) - GPU efficiency
# 3. Throughput (bottom-left) - tokens/sec over time
# 4. Summary Statistics (bottom-right) - final metrics

# Save to:
#   assets/training_metrics_YYYYMMDD_HHMMSS.png (timestamped)
#   assets/training_metrics_latest.png (always latest)
```

**Final Output Files:**
```
📁 logs/
  └─ training_20260311_120000.log    ← Full training log with all metrics
  
📁 assets/
  ├─ training_metrics_latest.png     ← ⭐ Beautiful 4-subplot graph
  ├─ training_metrics_20260311_*.png ← Historical graphs
  └─ README.md                        ← Graph documentation
```

---

## 📊 Data Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ User runs: ./autoresearcher                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: setup_a100.py                                       │
│ ✓ Check Python 3.12+, PyTorch, CUDA, GPU memory, CPU cores │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: prepare.py                                          │
│ • Download data shards from HuggingFace (parallel workers)  │
│ • Train BPE tokenizer (converts text → token IDs)           │
│ • Create data loaders for training                          │
│ Output: ~/.cache/autoresearch/data/ & tokenizer/            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: ollama_deepseek.py (OPTIONAL)                       │
│ • Start Ollama server                                       │
│ • Download 6.7B DeepSeek model                              │
│ • Ready for code generation feedback                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: train.py                                            │
│ • Load 12-layer GPT model (768-dim, 6 heads)                │
│ • Use Flash Attention 3 for speed                           │
│ • Train on downloaded data for TIME_BUDGET seconds          │
│ • Log loss, MFU, throughput every step                      │
│ • Auto-stop at time limit                                   │
│ Output: logs/training_*.log                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ POST-TRAINING: Graph Generation                             │
│ • Parse training log                                        │
│ • Generate 4-subplot metrics graph                          │
│ • Save to assets/training_metrics_*.png                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 🎉 DONE! View results:                                      │
│   • assets/training_metrics_latest.png (main output)        │
│   • logs/training_*.log (full training log)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Files Reference

| File | Purpose | What It Does |
|------|---------|-------------|
| `autoresearcher` | Main orchestrator | Coordinates all 4 steps, interactive prompts, logging, graph generation |
| `setup_a100.py` | Hardware verification | Checks GPU/CPU/memory meets A100 requirements |
| `prepare.py` | Data preparation | Downloads data shards, trains tokenizer, creates dataloaders |
| `train.py` | Training loop | GPT model definition, Flash Attention 3, training loop, metrics logging |
| `ollama_deepseek.py` | AI integration | Interface to Ollama/DeepSeek for code generation (optional) |

---

## 💾 Cached Data Locations

```
~/.cache/autoresearch/
├── data/
│   ├── shard_00001.parquet    ← Raw dataset shards
│   ├── shard_00002.parquet
│   └── ... (up to NUM_SHARDS)
└── tokenizer/
    └── (BPE tokenizer files)   ← Converts text to token IDs
```

These are **downloaded once and reused** across runs (saves time & bandwidth!).

---

## 🚀 Example Execution Trace

```bash
$ ./autoresearcher --minutes 10 --dataset arxiv

✓ Hardware verification passed! (2 seconds)
  ✓ Python 3.12.1
  ✓ PyTorch 2.1.0 + CUDA 12.1
  ✓ GPU: NVIDIA A100 (80GB)
  ✓ Compute Capability: 9.0

✓ Data preparation completed! (45 seconds)
  ✓ Downloaded 10 ArXiv shards (4.2 GB)
  ✓ Trained BPE tokenizer (vocab size: 8192)
  ✓ Created 10 data batches (batch size: 128)

🚀 Training started! (600 seconds = 10 minutes)
  step 00000 | loss: 4.123 | mfu: 2.1% | tok/sec: 132,456 | epoch: 1
  step 00010 | loss: 3.987 | mfu: 2.3% | tok/sec: 135,123 | epoch: 1
  step 00020 | loss: 3.856 | mfu: 2.4% | tok/sec: 136,789 | epoch: 1
  ...
  ⏱ TIME LIMIT REACHED - Stopping training
  
✓ Training completed! (600 seconds = 10 minutes)
  Final Loss: 3.456 (↓ 16% improvement)
  
✓ Graph generated: assets/training_metrics_latest.png
  📈 4 subplots: Loss, MFU, Throughput, Summary

🎉 Pipeline complete! Check assets/training_metrics_latest.png
```

---

## 🎓 What Gets Optimized on A100

1. **Flash Attention 3**: Ultra-fast attention using A100 tensor cores
2. **fp8 / bf16 precision**: Mixed precision training (if enabled)
3. **Gradient accumulation**: Efficient memory usage
4. **Expandable segments**: Dynamic CUDA memory management
5. **Parallel data loading**: 42 workers downloading while GPU trains

These optimizations are **why A100 runs 5-10x faster** than consumer GPUs!

---

## 📝 Summary

When you run `./autoresearcher`:

1. **Configures** - Asks about duration, dataset, workers
2. **Verifies** - Checks A100 hardware
3. **Downloads** - Fetches data shards in parallel
4. **Tokenizes** - Converts text to token IDs (once)
5. **Sets up DeepSeek** - Optional AI feedback (if enabled)
6. **Trains** - Runs GPT model for your specified time
7. **Monitors** - Shows live metrics every 5 seconds
8. **Auto-stops** - Kills training exactly at time limit
9. **Visualizes** - Generates beautiful metric graphs
10. **Saves** - Stores logs and graphs for later analysis

All orchestrated by a single bash script calling Python scripts! 🎉
