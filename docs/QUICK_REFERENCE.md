# AutoResearcher Quick Reference Guide

## 🚀 Quick Start

```bash
chmod +x autoresearcher
./autoresearcher                    # Interactive mode
./autoresearcher --minutes 10       # 10-minute training
./autoresearcher --hours 2          # 2-hour training
./autoresearcher --minutes 5 --dataset arxiv  # 5 min, ArXiv data
./autoresearcher --help             # Show all options
```

---

## 📋 What Happens When You Press Enter

### Immediate (0-5 seconds)
1. ✓ Script starts, shows banner
2. ✓ Checks if arguments provided (if not, shows interactive prompts)
3. ✓ Creates `logs/` directory
4. ✓ Sets up all variables

### Phase 1: Hardware Check (2-5 seconds)
```
Runs: python setup_a100.py

Checks:
✓ Python 3.12+ installed
✓ PyTorch with CUDA support
✓ NVIDIA A100 GPU detected (80GB)
✓ CPU has 42+ cores
✓ System has 256GB+ RAM

Status: Hardware verification passed!
```

### Phase 2: Data Download (30-120 seconds)
```
Runs: python prepare.py --dataset climbmix --num-shards 10

What happens:
1. Downloads 10 data shards in parallel (42 workers)
   • Each shard: ~400MB of raw text
   • Total: ~4GB downloaded
   • Saved to: ~/.cache/autoresearch/data/

2. Trains BPE Tokenizer
   • Learns vocabulary of 8192 tokens
   • Converts text → numbers
   • Saved to: ~/.cache/autoresearch/tokenizer/

3. Creates Data Loaders
   • Batches data for training
   • Each batch: 2048 tokens per sample
   • Ready to feed into GPU

Status: Data preparation completed!
```

### Phase 3: DeepSeek Setup (optional, 20-60 seconds)
```
If you chose --deepseek:

Runs: ollama pull deepseek-coder:6.7b-base-q4_0

What happens:
1. Starts Ollama server (localhost:11434)
2. Downloads 6.7B parameter model (~4.5 GB)
3. Model ready for code generation

Status: DeepSeek integration ready!
```

### Phase 4: Training Loop (TIME_BUDGET seconds)
```
Runs: export TIME_BUDGET_SECONDS=600 && uv run train.py

What you see:
╔════════════════════════════════════════════════════════════╗
║          🚀 TRAINING IN PROGRESS - LIVE METRICS 🚀         ║
╚════════════════════════════════════════════════════════════╝

⏱️  Training Progress:
  Elapsed: 2m 30s | Remaining: 7m 30s
  Progress: [##########░░░░░░░░░░] 25%

📊 Latest Training Metrics:
  step 00245 | loss: 3.456 | mfu: 2.2% | tok/sec: 135,454

📈 Detailed Metrics:
  Loss (BPB):        3.456      ← Should go DOWN
  MFU (%):           2.2        ← GPU efficiency
  LR Multiplier:     1.00       ← Learning rate schedule
  Tokens/sec:        135,454    ← Training speed
  Epoch:             1          ← Data epoch

What's happening:
• GPU processing 2048 tokens per batch
• Computing gradients via backpropagation
• Updating 35 million model parameters
• Logging metrics every ~100 steps
• Auto-stops at TIME_BUDGET (you set this!)
```

### Phase 5: Graph Generation (2-5 seconds)
```
Runs: Embedded Python script

What happens:
1. Reads: logs/training_20260311_120000.log
2. Extracts: steps, losses, MFU, throughput
3. Creates: 4-subplot figure
   • Loss Curve (learning progress)
   • MFU % (GPU efficiency)
   • Throughput (tokens/sec)
   • Summary Statistics

Status: ✓ Graph generated!
Saved to: assets/training_metrics_latest.png
          assets/training_metrics_20260311_120000.png
```

### Final Output (5-10 seconds)
```
🎉 Complete Pipeline Finished! 🎉

⏱️  Timing Summary:
  🔧 Hardware Verification: 3s
  📥 Data Preparation: 45s
  🚀 Training: 600s (10 min)
  📊 Total: 648s (~11 minutes)

📁 Output Files:
  📝 logs/training_20260311_120000.log
     → Full training log (2000+ lines)
  
  📈 assets/training_metrics_latest.png ⭐
     → Beautiful graph with 4 subplots
  
  📚 assets/README.md
     → Documentation about the graphs

🎯 Final Metrics:
  Total Steps: 245
  Initial Loss: 4.123 BPB
  Final Loss: 3.456 BPB
  Improvement: ↓ 16%
  Peak MFU: 2.4%
  Peak Throughput: 136k tok/sec
```

---

## 📊 Understanding the Metrics

| Metric | Range | What It Means | Good Value |
|--------|-------|--------------|-----------|
| **Loss (BPB)** | 0-6 | Bits per byte prediction error. Lower = better | Starts ~4.0, ends <3.5 |
| **MFU (%)** | 0-100 | % of A100's peak compute being used | 15-30% is normal |
| **tok/sec** | 0-1M | Tokens processed per second | >100k is good |
| **LR Multiplier** | 0-1 | Learning rate schedule (cosine annealing) | Starts 1.0, decays to 0 |
| **dt (ms)** | 100-2000 | Time per training step | Should be stable |
| **Epoch** | 0-∞ | How many times we've seen all training data | Increases as you train |

---

## 🗂️ File Locations

### Input Files (Created by prepare.py)
```
~/.cache/autoresearch/
├── data/
│   ├── shard_00001.parquet    ← Raw dataset, ~400MB each
│   ├── shard_00002.parquet
│   ├── shard_00003.parquet
│   └── ... (up to NUM_SHARDS)
│
└── tokenizer/
    ├── tokenizer.model        ← BPE tokenizer
    └── tokenizer.vocab        ← Vocabulary file
```

### Output Files (Created during training)
```
./logs/
├── training_20260311_120000.log     ← All training metrics
├── training_20260311_121245.log     ← Each run gets new timestamp
├── ollama.log                        ← DeepSeek logs (if --deepseek)
└── metrics.json                      ← Structured metrics

./assets/
├── training_metrics_latest.png       ← ⭐ Current graph
├── training_metrics_20260311_120000.png ← Historical graphs
└── README.md                         ← Graph docs
```

---

## 🎛️ Command-Line Options

```bash
./autoresearcher [OPTIONS]

TIME:
  --minutes N       Training duration in minutes (default: 10)
  --hours N         Training duration in hours

DATA:
  --dataset NAME    Dataset choice:
                    • climbmix (default) - balanced, general
                    • arxiv - research papers
                    • wiki - general knowledge
                    • code - programming code
                    • stackexchange - Q&A content
  
  --shards N        Number of data shards to download (default: 10)
  --workers N       Parallel download workers (default: 42)

AI:
  --deepseek        Enable DeepSeek Coder integration
  --improve-interval N   Feedback interval in minutes (default: 5)
                        (only used with --deepseek)

HELP:
  --help            Show this help message
```

### Examples

```bash
# Quick 5-minute test
./autoresearcher --minutes 5

# Standard 10-minute training with ArXiv data
./autoresearcher --minutes 10 --dataset arxiv

# 30-minute training with DeepSeek feedback
./autoresearcher --minutes 30 --deepseek

# 2-hour training with custom workers and shards
./autoresearcher --hours 2 --shards 20 --workers 50

# Extended training with all options
./autoresearcher --hours 1 --dataset code --deepseek --improve-interval 10
```

---

## 🔍 Monitoring Training

### Live Dashboard
The script automatically shows this every 5 seconds:
```
Elapsed: 2m 30s | Remaining: 7m 30s
Progress: [##########░░░░░░░░░░] 25%
Loss: 3.456 | MFU: 2.2% | Tokens/sec: 135,454
```

### Tail the Log File (in another terminal)
```bash
tail -f logs/training_*.log | grep "step"
# Shows each step as it happens
```

### Check the Graph
```bash
# After training finishes
open assets/training_metrics_latest.png    # macOS
xdg-open assets/training_metrics_latest.png # Linux
```

---

## ⚠️ Common Issues & Solutions

### "CUDA not available"
```bash
# Problem: GPU not detected
Solution: 
1. Check: nvidia-smi
2. Verify NVIDIA drivers installed
3. Reinstall PyTorch with CUDA support
```

### "Ollama connection failed"
```bash
# Problem: DeepSeek setup failed
Solution:
1. Install Ollama: curl https://ollama.ai/install.sh | sh
2. Run manually: ollama serve
3. Retry: ./autoresearcher --deepseek
```

### "Out of memory" (OOM)
```bash
# Problem: GPU ran out of memory
Solution:
1. Reduce batch size: (in train.py, find BATCH_SIZE, reduce it)
2. Use fewer shards: --shards 5 instead of 10
3. Use smaller model: (not configurable, but is a limit)
```

### Training stopped early
```bash
# Problem: Training didn't reach time limit
Solution:
1. Check logs: tail logs/training_*.log
2. Look for error messages
3. Ensure adequate cooling (A100 has thermal limits)
```

### No graph generated
```bash
# Problem: Missing .png file in assets/
Solution:
1. Check if training actually logged steps
2. Look for: "step 00000" in training log
3. Verify matplotlib installed: pip list | grep matplotlib
```

---

## 📈 Interpreting Your Results

### Loss Trajectory
```
Good:    4.5 → 4.2 → 3.9 → 3.6 → 3.4 (smooth decrease)
Bad:     4.5 → 4.1 → 4.6 → 3.8 → 3.9 (oscillating)
Bad:     4.5 → 4.5 → 4.5 → 4.5 → 4.5 (not learning)
```

### MFU Interpretation
```
2-5% MFU:     Poor GPU utilization (expected on A100, optimal for large models)
5-20% MFU:    Good utilization
20-50% MFU:   Excellent (rarely achieved on A100 with large batch sizes)
```

### Throughput Expectations
```
100k tok/sec:  Baseline (good)
135k tok/sec:  Average (very good!)
200k+ tok/sec: Excellent (rare, requires full A100 utilization)
<50k tok/sec:  Investigate - GPU might be bottlenecked
```

---

## 💾 Reusing Data Across Runs

The codebase caches downloaded data:
```
First run (--dataset arxiv):
• Downloads ArXiv shards
• Caches in ~/.cache/autoresearch/data/
• Time: ~45 seconds

Subsequent runs (same dataset):
• Checks cache first
• Skips download if already present
• Time: ~5 seconds

Different dataset:
• Downloads new shards
• All datasets cached separately
• Only one vocabulary per dataset
```

**Tip:** If you want to start fresh:
```bash
rm -rf ~/.cache/autoresearch/
./autoresearcher --minutes 10   # Fresh download
```

---

## 🎓 What to Try Next

After your first run:

1. **Compare Datasets**
   ```bash
   ./autoresearcher --minutes 10 --dataset arxiv
   ./autoresearcher --minutes 10 --dataset code
   # Compare graphs: which trains better?
   ```

2. **Scale Training Duration**
   ```bash
   ./autoresearcher --hours 1
   # Train longer to see loss go even lower
   ```

3. **Use More Data**
   ```bash
   ./autoresearcher --minutes 10 --shards 50
   # More data = potentially better loss curves
   ```

4. **Enable DeepSeek**
   ```bash
   ./autoresearcher --minutes 10 --deepseek
   # Get AI-powered code suggestions during training
   ```

5. **Batch Analysis**
   ```bash
   # Run multiple trainings, compare graphs
   for i in {1..5}; do
     ./autoresearcher --minutes 5 --dataset wiki
   done
   # Analyze consistency/variance of results
   ```

---

## 🎯 Summary: What Each File Does

| File | Purpose | Execution Time | Input | Output |
|------|---------|-----------------|-------|--------|
| `setup_a100.py` | Verify hardware | 2-5s | (none) | ✓/✗ status |
| `prepare.py` | Download & tokenize | 30-120s | Dataset name | `~/.cache/autoresearch/` |
| `train.py` | ML training loop | TIME_BUDGET | Tokenized data | `logs/training_*.log` |
| Graph script | Visualize results | 2-5s | Training log | `assets/*.png` |
| `ollama_deepseek.py` | AI integration | 20-60s | (if enabled) | Ollama server ready |

**Total time for 10-minute training: ~12 minutes** (data prep takes longer than training!)

---

## ✨ Final Tips

✅ **Do This:**
- Run with `--help` to see all options
- Check `assets/README.md` for graph documentation
- View logs with `tail -f` to monitor real-time
- Compare graphs from different datasets
- Increase training time for better models

❌ **Don't Do This:**
- Kill the script mid-training (let auto-stop handle it)
- Modify model architecture without understanding code
- Run 8-hour trainings without monitoring
- Trust just one run (run 2-3 times to validate)

🚀 **Performance Notes:**
- A100 80GB should achieve 135k+ tokens/sec
- Loss should improve by 10-20% over 10 minutes
- MFU 2-5% is normal for this model size
- Data download is usually the bottleneck!

