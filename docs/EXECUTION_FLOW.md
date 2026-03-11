# AutoResearcher: Visual Execution Summary

## [SCENE] The Complete Story - Start to Finish

### [INFO] You Are Here
```
User: ./autoresearcher --minutes 10
         │
         └─→ [SCENE] SCENE 1: Configuration & Prompts
         └─→ [SCENE] SCENE 2: Hardware Verification
         └─→ [SCENE] SCENE 3: Data Download
         └─→ [SCENE] SCENE 4: Tokenization
         └─→ [SCENE] SCENE 5: DeepSeek Setup (optional)
         └─→ [SCENE] SCENE 6: Training Loop
         └─→ [SCENE] SCENE 7: Graph Generation
         └─→ [END] THE END: Results Ready!
```

---

## [SCENE] SCENE 1: Configuration & Prompts (5 seconds)

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║          [STARTUP] AutoResearcher - A100 80GB Complete Pipeline [STARTUP]       ║
║                                                                   ║
║  Optimized for: NVIDIA A100 80GB | Intel Xeon 42-core           ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

(if interactive mode, user sees prompts)

▶ How long to train? [1-6] or minutes: 2
▶ Which dataset? [1-5]: 1
▶ Number of shards? [default 10]: 10
▶ Number of workers? [default 42]: 42
▶ Enable DeepSeek? [y/n]: n

═══════════════════════════════════════════════════════════════════
📋 Configuration Summary

[TIME] TIME: 10 minutes
[DATA] DATA: ClimbMix dataset, 10 shards, 42 workers
[CONFIG] AI: DeepSeek disabled

Start? [y]: y
═══════════════════════════════════════════════════════════════════
```

**Result:** [OK] All variables set, ready to proceed

---

## [SCENE] SCENE 2: Hardware Verification (2-5 seconds)

```bash
$ python setup_a100.py
```

```
[SETUP] Step 1/4: Hardware Verification
────────────────────────────────────

Checking Python version...
[OK] Python 3.12.1

Checking PyTorch & CUDA...
[OK] PyTorch 2.1.0
[OK] CUDA 12.1
[OK] Device: NVIDIA A100-SXM4-80GB
[OK] Compute Capability: 9.0 (Hopper GPU - OPTIMAL!)
[OK] GPU Memory: 80.0 GB

Checking CPU...
[OK] CPU Cores: 42 (Intel Xeon Platinum)

Checking System Memory...
[OK] System Memory: 256.0 GB (available: 200.0 GB)

────────────────────────────────────
[OK] Hardware verification passed!
```

**File Involved:** `setup_a100.py`

**What Happened:**
- Imported psutil, torch, sys
- Checked all hardware parameters
- Verified A100 specific details
- Green light to proceed!

---

## [SCENE] SCENE 3: Data Download (30-120 seconds)

```bash
$ python prepare.py --dataset climbmix --num-shards 10 --download-workers 42
```

```
[DOWNLOAD] Step 2/4: Data Preparation (climbmix)
──────────────────────────────────────────────────

Downloading 10 shards with 42 parallel workers...

[Worker 01] Downloading shard 00001.parquet ...  [OK] 412 MB
[Worker 02] Downloading shard 00002.parquet ...  [OK] 405 MB
[Worker 03] Downloading shard 00003.parquet ...  [OK] 418 MB
[Worker 04] Downloading shard 00004.parquet ...  [OK] 410 MB
...
[Worker 42] Downloading shard 00010.parquet ...  [OK] 408 MB

Total downloaded: 4.1 GB in 45 seconds
Download speed: 91 MB/s (parallel efficiency!)

──────────────────────────────────────────────

Training BPE Tokenizer...
  • Vocabulary size: 8192 tokens
  • Tokens trained: 50,000,000
  • Vocabulary coverage: 99.8%

✓ Tokenizer training complete!

──────────────────────────────────────────────

Creating data loaders...
  • Sequence length: 2048 tokens
  • Batch size: 128
  • Total batches: 2,048
  • Tokens per epoch: ~268 million

[OK] Data preparation completed!
```

**File Involved:** `prepare.py`

**Where Data Lives Now:**
```
~/.cache/autoresearch/
├── data/
│   ├── shard_00001.parquet    ← 412 MB
│   ├── shard_00002.parquet    ← 405 MB
│   ├── shard_00003.parquet    ← 418 MB
│   ... (10 total)
│
└── tokenizer/
    ├── tokenizer.model        ← BPE model
    └── tokenizer.vocab        ← 8192 vocab
```

**Key Thing:** Data is now cached! Next run (same dataset) will skip download. [FAST]

---

## [SCENE] SCENE 4: Tokenization (Built into Prepare.py)

```
Understanding the Token Conversion:

Raw Text from Parquet:
"In machine learning, we train neural networks..."

↓ (BPE Tokenizer: 8192 vocabulary)

Token IDs:
[456, 2341, 1092, 847, 312, 1876, 4521, 3012, ...]
  ↑     ↑      ↑      ↑    ↑    ↑     ↑     ↑
 "In" "machine" "learning" "," "we" "train" "neural" ...

Ready for Model!
Each token is just an integer 0-8191
Model will convert these to 768-dimensional vectors
```

---

## [SCENE] SCENE 5: DeepSeek Setup (Optional, 20-60 seconds)

```
(This only runs if you chose --deepseek)

[CONFIG] Step 3/4: DeepSeek Coder Integration Setup
─────────────────────────────────────────────

Checking Ollama...
[OK] Ollama found at /usr/local/bin/ollama
[OK] Starting Ollama server...
[OK] Ollama ready at http://localhost:11434

Checking DeepSeek model...
[ERROR] deepseek-coder:6.7b-base-q4_0 not found locally

Downloading model (first time only)...
```

It shows a progress bar:

```
Downloading deepseek-coder:6.7b-base-q4_0...
████████████████████░░░░░░░░░░░░░░░░░░ 45% [3.2 GB / 7.1 GB]

(takes 2-5 minutes depending on internet speed)

[OK] Model downloaded successfully!
[OK] DeepSeek ready for inference

───────────────────────────────────────────

DeepSeek will provide code feedback every 5 minutes during training.
```

**Files Involved:** `ollama_deepseek.py`, Ollama server

**After Setup:** DeepSeek runs independently on port 11434. Ready! [STARTUP]

---

## [SCENE] SCENE 6: Training Loop (600 seconds = 10 minutes)

```bash
$ export TIME_BUDGET_SECONDS=600
$ uv run train.py
```

### What Happens Behind the Scenes

```
[00:00] Loading model...
  • Creating 12-layer GPT network
  • 35 million parameters
  • Flash Attention 3 kernels ready
  • Model on GPU: 78 GB used
  
[00:05] Starting training...
  • Loading batch 0 (128 samples, 2048 tokens each)
  • Forward pass: 128 sequences through 12 transformer blocks
  • Backward pass: Computing gradients
  • Optimizer: Updating all parameters
  • Logging: step 0, loss, MFU, throughput

[00:06] Batch 1...
[00:07] Batch 2...
...
[10:00] Time limit reached!
  • Final step: 245 completed
  • Training stopped automatically
```

### What YOU See Every 5 Seconds

```
╔════════════════════════════════════════════════════════════════╗
║          [STARTUP] TRAINING IN PROGRESS - LIVE METRICS [STARTUP]             ║
╚════════════════════════════════════════════════════════════════╝

[TIME] Training Progress:
  Elapsed: 2m 30s | Remaining: 7m 30s
  Progress: [##########░░░░░░░░░░░░░░░░░░░░░░] 25%

[DATA] Latest Training Metrics:
  step 00245 | loss: 3.456 | lrm: 1.00 | dt: 1000ms | tok/sec: 135,454 | mfu: 2.2% | epoch: 1 | remaining: 500s

[METRICS] Detailed Metrics:
  Loss (BPB):        3.456     ← Lower is better! Started at 4.12
  MFU (%):           2.2       ← GPU efficiency
  LR Multiplier:     1.00      ← Schedule control
  Tokens/sec:        135,454   ← Fast!
  Epoch:             1         ← Data coverage
```

### What's Being Saved

Every step is appended to:
```
logs/training_20260311_120000.log

[2026-03-11 12:00:05] step 00000 | loss: 4.123 | lrm: 1.00 | dt: 1000ms | tok/sec: 132,456 | mfu: 2.1% | epoch: 1 | remaining: 600s
[2026-03-11 12:00:06] step 00001 | loss: 4.087 | lrm: 1.00 | dt: 980ms | tok/sec: 133,215 | mfu: 2.1% | epoch: 1 | remaining: 599s
[2026-03-11 12:00:07] step 00002 | loss: 4.056 | lrm: 0.99 | dt: 1020ms | tok/sec: 134,890 | mfu: 2.2% | epoch: 1 | remaining: 598s
...
[2026-03-11 12:10:05] step 00245 | loss: 3.456 | lrm: 0.05 | dt: 985ms | tok/sec: 135,454 | mfu: 2.2% | epoch: 1 | remaining: 0s
[2026-03-11 12:10:05] ========== TRAINING COMPLETE ==========
[2026-03-11 12:10:05] Total training time: 10m 0s
```

**Files Involved:** `train.py` (reads from prepare.py, writes to logs/)

---

## [SCENE] SCENE 7: Graph Generation (2-5 seconds)

```
Generating training metrics graph...

Step 7a: Parse log file
  [OK] Read 246 step entries
  [OK] Extracted loss values: [4.123, 4.087, 4.056, ..., 3.456]
  [OK] Extracted MFU values: [2.1%, 2.1%, 2.2%, ..., 2.2%]
  [OK] Extracted throughput: [132k, 133k, 134k, ..., 135k]

Step 7b: Create visualization
  [OK] Subplot 1 (top-left): Loss Curve
    └─ Trends: 4.12 → 3.45 (↓ 16% improvement!)
  
  [OK] Subplot 2 (top-right): MFU (%)
    └─ Peak: 2.4%, Average: 2.1%
  
  [OK] Subplot 3 (bottom-left): Throughput
    └─ Average: 134.6k tokens/sec
  
  [OK] Subplot 4 (bottom-right): Summary Stats
    └─ Total steps: 245
    └─ Loss improvement: 16%
    └─ Training time: 10m 0s

Step 7c: Save files
  [OK] Saved: assets/training_metrics_20260311_120000.png (1.2 MB)
  [OK] Updated: assets/training_metrics_latest.png
  ✓ Graph generation complete!
```

**What the Graph Looks Like:**

```
╔─────────────────────────────────────────────────────────────╗
║  A100 Training Metrics - 245 Steps                          ║
╠──────────────────────────┬──────────────────────────────────╣
║ Loss Curve              │ Model FLOPs Utilization          ║
║ ↑ 4.2                  │ ↑ 3%                              ║
║ │        \              │ │       ___                       ║
║ │         \             │ │ ____/                           ║
║ │          \___         │ │                                 ║
║ │  ↓16%      ╲__       │ │ Max: 2.4%                        ║
║ ├─────────→───────     │ │                                  ║
║ │                     │ │                                  ║
║ 3.0 steps →           │ 0% steps →                         ║
╠──────────────────────────┼──────────────────────────────────╣
║ Training Throughput     │ Summary Statistics               ║
║ ↑ 150k                 │                                   ║
║ │        ___            │ Total Steps:     245             ║
║ │ ______/               │ Initial Loss:    4.123 BPB       ║
║ │                      │ Final Loss:      3.456 BPB       ║
║ │ Avg: 134.6k tok/s   │ Improvement:     ↓ 16%           ║
║ ├─────────→───────     │ Avg MFU:         2.1%            ║
║ │                     │ Peak Throughput: 136k tok/s       ║
║ 100k steps →           │ Time Budget Used: 600s           ║
╚──────────────────────────┴──────────────────────────────────╝
```

---

## 🏁 THE END: Results Ready!

```
✨ Complete Pipeline Finished! ✨

⏱️  Timing Summary:
  🔧 Hardware Verification: 3s
  📥 Data Preparation: 45s
  🚀 Training: 600s (10 minutes)
  📊 Graph Generation: 3s
  ─────────────────────────
  ⏰ Total: 651 seconds (~11 minutes)

📝 Training Log:
   logs/training_20260311_120000.log (2,048 lines)
   ↳ Every step, every metric, full history

📈 Metrics Graph (⭐ MAIN OUTPUT):
   assets/training_metrics_latest.png (1.2 MB)
   ↳ Beautiful 4-subplot visualization
   
📚 Historical Graphs:
   assets/training_metrics_20260311_120000.png
   ↳ Timestamped for comparison

🎯 Final Metrics:
   Total Steps Completed: 245
   Loss Improvement: 4.12 BPB → 3.45 BPB (↓ 16%)
   Peak MFU: 2.4%
   Peak Throughput: 136k tokens/sec
   Average Throughput: 134.6k tokens/sec

📍 Next Steps:
   1. View graph: open assets/training_metrics_latest.png
   2. Analyze logs: cat logs/training_20260311_120000.log
   3. Try different dataset: ./autoresearcher --dataset arxiv
   4. Longer training: ./autoresearcher --hours 2
   5. Enable AI: ./autoresearcher --deepseek

🎉 Pipeline complete! Check your beautiful graph! 🎉
```

---

## 📊 Comparison: What Changes Between Runs

### Run 1: ClimbMix, 10 minutes
```
Final Loss: 3.45
Peak MFU: 2.4%
Throughput: 134k tok/sec
```

### Run 2: ArXiv Papers, 10 minutes
```
Final Loss: 3.32 (BETTER - academic text helps!)
Peak MFU: 2.4%
Throughput: 134k tok/sec (SAME - same model)
```

### Run 3: ClimbMix, 30 minutes (longer!)
```
Final Loss: 3.12 (MUCH BETTER - more training!)
Peak MFU: 2.4%
Throughput: 134k tok/sec (SAME)
```

**Observation:** Dataset + duration affect loss, GPU metrics stay constant!

---

## 🎓 Understanding the Flow

```
QUICK SUMMARY:

1️⃣  Config (5s)       → What to do?
2️⃣  Hardware (5s)     → Can we do it?
3️⃣  Data (45s)        → Get the training data
4️⃣  Tokenize (built-in) → Convert text to numbers
5️⃣  DeepSeek (opt)    → Optional AI help
6️⃣  Train (600s)      → Actual GPU work happens here ⚡
7️⃣  Visualize (5s)    → Show results
8️⃣  Done! (5s)        → Celebrate! 🎉
```

**Total: ~11 minutes for a 10-minute training run** (data prep overhead)

---

## 🚀 Pro Tips

✅ **Monitor Real-Time:**
```bash
# In another terminal, tail the log
tail -f logs/training_*.log | grep "step"
```

✅ **Compare Runs:**
```bash
# Run multiple times, compare PNG graphs
ls -lh assets/training_metrics_*.png
```

✅ **Faster Reruns:**
```bash
# Data cached after first download!
# Second run same dataset: ~70 seconds total (no download)
```

✅ **Check GPU Usage:**
```bash
# In another terminal, monitor A100
nvidia-smi dmon -s pcm
```

---

## 🎯 Success Indicators

| Indicator | ✓ Good | ⚠️ Warning | ✗ Problem |
|-----------|--------|-----------|----------|
| **Loss** | Decreases 10-20% | Flat/plateau | Increases |
| **MFU** | 2-5% | <2% or >50% | 0% or errors |
| **tok/sec** | 100k+ | 50-100k | <50k |
| **Graph** | Smooth curves | Noisy/spiky | Straight lines |
| **Training** | Completes | Stops early | Crashes |

If most are ✓: **Success!** 🎉

---

## 📋 Files Referenced During Execution

```
autoresearcher (main bash orchestrator)
  ├─ Calls → setup_a100.py
  │          (checks hardware)
  │
  ├─ Calls → prepare.py
  │          (downloads & tokenizes)
  │          ├─ Creates → ~/.cache/autoresearch/data/
  │          └─ Creates → ~/.cache/autoresearch/tokenizer/
  │
  ├─ Calls → ollama_deepseek.py (optional)
  │          (sets up AI model)
  │
  ├─ Calls → train.py
  │          (the training loop)
  │          ├─ Reads from → prepare.py functions
  │          ├─ Reads from → ~/.cache/autoresearch/
  │          └─ Writes to → logs/training_*.log (via tee)
  │
  └─ Embedded Python Script
             (graph generation)
             ├─ Reads from → logs/training_*.log
             └─ Writes to → assets/training_metrics_*.png
```

---

## 🎬 The End

You now understand the **complete 8-phase pipeline** from start to finish! 

Each phase builds on the previous one:
- ✅ Hardware = ✓ green light
- ✅ Hardware + Data = training ready
- ✅ Data + Model = training happens
- ✅ Training = beautiful results!

**Happy training!** 🚀
