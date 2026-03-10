"""
plot_train.py — Generate a training curve plot from a captured train log.

Usage:
    uv run python plot_train.py                        # uses latest train_*.log in ~/.cache/autoresearcher/
    uv run python plot_train.py path/to/train.log      # explicit log file
    uv run python plot_train.py --out my_plot.png      # custom output path
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def find_latest_log():
    cache = os.path.join(os.path.expanduser("~"), ".cache")
    candidates = [
        os.path.join(cache, "train_5min.log"),
        os.path.join(cache, "train_smoke.log"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def parse_log(path):
    steps, losses, tok_secs = [], [], []
    val_bpb = None
    pattern = re.compile(
        r"step (\d+) \([\d.]+%\) \| loss: ([\d.]+) \| .* \| tok/sec: ([\d,]+)"
    )
    bpb_pattern = re.compile(r"val_bpb:\s+([\d.]+)")
    with open(path, encoding="utf-16") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                tok_secs.append(int(m.group(3).replace(",", "")))
            bm = bpb_pattern.search(line)
            if bm:
                val_bpb = float(bm.group(1))
    return steps, losses, tok_secs, val_bpb


def smooth(values, window=50):
    return [
        sum(values[max(0, i - window): i + 1]) / len(values[max(0, i - window): i + 1])
        for i in range(len(values))
    ]


def plot(steps, losses, tok_secs, val_bpb, out_path):
    BG      = "#0d1117"
    PANEL   = "#161b22"
    BORDER  = "#30363d"
    TEXT    = "#c9d1d9"
    TITLE   = "#e6edf3"
    BLUE    = "#58a6ff"
    GREEN   = "#3fb950"
    ORANGE  = "#f0883e"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.patch.set_facecolor(BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT)
        ax.spines[:].set_color(BORDER)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(TEXT)

    # ---- Loss ----
    ax1.plot(steps, losses, color=BLUE, linewidth=0.6, alpha=0.25)
    ax1.plot(steps, smooth(losses), color=BLUE, linewidth=2.0, label="loss (smoothed)")

    title = "autoresearcher — training run  |  RTX 4060 Laptop"
    if val_bpb is not None:
        title += f"  |  val_bpb = {val_bpb:.4f}"
    ax1.set_title(title, color=TITLE, fontsize=12, pad=10)
    ax1.set_ylabel("Training Loss", color=TEXT)
    ax1.legend(facecolor="#21262d", edgecolor=BORDER, labelcolor=TEXT)

    if val_bpb is not None:
        ax1.text(
            0.02, 0.08, f"val_bpb = {val_bpb:.4f}",
            transform=ax1.transAxes, color=ORANGE,
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d", edgecolor=ORANGE, alpha=0.9),
        )

    ax1.annotate(
        f"final: {losses[-1]:.3f}",
        xy=(steps[-1], losses[-1]), xytext=(-80, 14),
        textcoords="offset points", color=GREEN, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
    )

    # ---- Throughput ----
    tok_k = [t / 1000 for t in tok_secs]
    ax2.plot(steps, tok_k, color=GREEN, linewidth=0.6, alpha=0.25)
    ax2.plot(steps, smooth(tok_k), color=GREEN, linewidth=2.0, label="tok/sec (smoothed)")
    ax2.set_ylabel("Throughput (k tok/sec)", color=TEXT)
    ax2.set_xlabel("Step", color=TEXT)
    ax2.legend(facecolor="#21262d", edgecolor=BORDER, labelcolor=TEXT)

    plt.tight_layout(h_pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default=None, help="Path to train log file")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    log_path = args.log or find_latest_log()
    if not log_path or not os.path.exists(log_path):
        print("No log file found. Pass a log path as argument.", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or os.path.splitext(log_path)[0] + "_plot.png"
    # Place output next to this script if log is in cache
    if ".cache" in log_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, os.path.basename(out_path))

    steps, losses, tok_secs, val_bpb = parse_log(log_path)
    print(f"Parsed {len(steps)} steps from {log_path}")
    if val_bpb:
        print(f"val_bpb = {val_bpb}")

    plot(steps, losses, tok_secs, val_bpb, out_path)


if __name__ == "__main__":
    main()
