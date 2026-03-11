# [DATA] AutoResearcher Training Metrics

This folder contains automatically generated training metrics graphs from each training run.

## [METRICS] Graph Files

- **`training_metrics_latest.png`** ⭐ **START HERE** - The most recent training run graph
- **`training_metrics_YYYYMMDD_HHMMSS.png`** - Timestamped archives of all previous runs

## [DATA] What the Graph Shows

The main metrics graph is a 4-panel visualization showing:

### Panel 1: Training Loss Curve (Top Left)
- **Blue line**: Loss in BPB (Bits Per Byte) over training steps
- **Key insight**: Should consistently decrease as the model learns
- **Improvement %**: Shows total loss reduction from start to end
- **Normal behavior**: Rapid initial drop, then slower improvements

**Example interpretation:**
- Loss 9.0 → 5.4 = 40% improvement (great!)
- Loss should improve exponentially fast early, then plateau

---

### Panel 2: Model FLOPs Utilization (Top Right)
- **Green line**: MFU (Model FLOPs Utilization) percentage over steps
- **Range**: 0-100%, but typically 1-3% for this model
- **Key insight**: Shows GPU compute efficiency
- **What to watch**: Should stabilize after first few steps

**Normal ranges:**
- Step 0: Very low (0.3%) - model initialization overhead
- Steps 1+: Stable at 1.5-2.2% - running efficiently
- Higher MFU = better hardware utilization

---

### Panel 3: Training Throughput (Bottom Left)
- **Red line**: Tokens/second processed by the model
- **Measured in**: Million tokens/sec (M tok/sec)
- **Key insight**: Shows data processing speed
- **What to watch**: Should stabilize quickly

**Normal ranges:**
- Step 0: Very low (17M tok/sec) - cold start with weight initialization
- Steps 1+: Stable at 90-140M tok/sec - sustained throughput
- Spikes up then down = normal I/O and cache behavior

---

### Panel 4: Summary Statistics (Bottom Right)
- **Total Steps**: Number of gradient update steps completed
- **Initial/Final Loss**: Starting and ending loss values
- **Loss Improvement**: Percentage reduction in loss
- **Avg/Peak MFU**: Average and maximum GPU utilization
- **Avg/Peak Throughput**: Average and maximum tokens/second
- **Time Budget Remaining**: Seconds left in the training window

## [SEARCH] Interpreting the Results

### Good Training Indicators [OK]
- Loss curve is smooth and continuously decreasing
- Loss improvement is positive (>0%)
- MFU stabilizes quickly (by step 5)
- Throughput is consistent (~90-140M tok/sec)
- No sudden drops or spikes in throughput

### Warning Signs [WARNING]
- Loss increases instead of decreases
- Loss curve is very noisy/jagged
- MFU stuck at very low values (<0.5%) after step 10
- Throughput varies wildly between steps
- Training stops prematurely

### Expected Patterns [DATA]

For a 5-minute training run with ~24 steps:
```
Loss:       9.0 → 5.4 (40% improvement)
MFU:        0.3% → 1.5% (stabilized)
Throughput: 17M → 95M tok/sec (stabilized)
Steps:      24 completed in 5 minutes
```

For longer runs (1+ hour):
- Loss should improve 50-70% overall
- MFU should stabilize even higher
- Throughput should remain consistent

## 📚 Technical Details

### Model Configuration
- **Architecture**: 8-layer Transformer (GPT-style)
- **Parameters**: ~151M total
- **Batch Size**: 96 samples
- **Sequence Length**: 2048 tokens
- **Precision**: BF16 (mixed precision)

### Metrics Explanation

**BPB (Bits Per Byte)**
- Standard metric for language model loss
- Lower is better
- Typical starting value: 8-10 BPB
- After training: 4-6 BPB

**MFU (Model FLOPs Utilization)**
- Percentage of GPU's peak compute capability used
- Theoretical peak on A100: 4,976 TFLOPS (BF16)
- Our model achieves: 1.5-2.2% of peak (normal for 8-layer model)

**Throughput (M tok/sec)**
- Million tokens processed per second
- Includes gradient accumulation overhead
- Normal range: 80-140M tok/sec
- Step 0 is slower due to JIT compilation

## [CORE] Common Questions

**Q: Why is step 0 so slow?**
A: First step includes model JIT compilation, weight initialization, and cache warmup. It's normal for step 0 to be 10x slower than subsequent steps.

**Q: Is 1.5% MFU good?**
A: Yes! With a ~150M parameter model, 1.5% MFU is excellent. Larger models (>1B) typically achieve higher MFU.

**Q: Why only 1 epoch?**
A: Each training run covers only one pass through the data (1 epoch) because we run for a fixed time (e.g., 5 minutes), not a fixed number of epochs. With 24 steps at ~8KB per step, we've only covered a small fraction of the dataset.

**Q: How long should I train?**
A: 
- **5 minutes**: Quick test, see if setup works [FAST]
- **10 minutes**: Standard test run [OK]
- **30+ minutes**: Real training with meaningful results [DATA]
- **1+ hour**: Production training [STARTUP]

**Q: Can I use these graphs?**
A: Yes! All graphs are automatically saved and ready to share. Use `training_metrics_latest.png` in presentations, reports, or documentation.

## 🔄 How Graphs Are Generated

1. **During training**: Each step is logged with metrics
2. **After training**: `autoresearcher` script automatically generates graphs
3. **Saved as**: 
   - Timestamped: `training_metrics_20260311_013221.png` (for archive)
   - Latest: `training_metrics_latest.png` (for quick reference)

The script parses the training log file and extracts:
- Step numbers
- Loss values (BPB)
- MFU percentages
- Throughput (tokens/sec)
- Remaining time budget

## 📖 Example Metrics from a 5-min Run

```
Total Steps:           24
Initial Loss:          9.0114 BPB
Final Loss:            5.4391 BPB
Loss Improvement:      ↓ 39.7%

Avg MFU:               1.5%
Peak MFU:              2.2%

Avg Throughput:        98.5M tok/sec
Peak Throughput:       141.1M tok/sec

Time Budget Remaining: 0s (stopped at limit)
```

This shows healthy training behavior! The loss improved significantly, MFU stabilized quickly, and throughput was consistent.

---

**Last Updated:** Generated automatically after each training run
**Format:** PNG images (high resolution, 120 DPI)
**Size:** ~160KB per graph
