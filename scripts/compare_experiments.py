#!/usr/bin/env python3
"""
Compare win rate curves across multiple experiment run directories.

Reads eval_results.csv produced by eval_checkpoint.py from each run directory
and plots overlaid win_rate ± CI vs epoch curves.

Usage:
    python scripts/compare_experiments.py \
        --runs logs/gr_football/exp_a/2026-03-24 logs/gr_football/exp_b/2026-03-24 \
        --labels "IPPO" "MAT" \
        --output logs/comparison.png

    # Auto-labels from run directory names if --labels not provided
    python scripts/compare_experiments.py \
        --runs logs/gr_football/exp_a/2026-03-24 logs/gr_football/exp_b/2026-03-24
"""

import argparse
import os
import sys
import pathlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_eval_results(run_dir):
    csv_path = os.path.join(run_dir, "eval_results.csv")
    if not os.path.exists(csv_path):
        print(f"[WARNING] No eval_results.csv found in {run_dir}, skipping.")
        return None
    df = pd.read_csv(csv_path)
    # Keep only epoch checkpoints (epoch >= 0)
    df = df[df["epoch"] >= 0].sort_values("epoch").reset_index(drop=True)
    if df.empty:
        print(f"[WARNING] No epoch checkpoints in {run_dir}, skipping.")
        return None
    return df


def auto_label(run_dir):
    """Generate a readable label from the run directory path."""
    parts = pathlib.Path(run_dir).parts
    # Use expr_name/timestamp if structure matches, else just the last two parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1]


def main():
    parser = argparse.ArgumentParser(description="Compare win rate curves across experiments")
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="Run directories containing eval_results.csv"
    )
    parser.add_argument(
        "--labels", nargs="*", default=None,
        help="Display labels for each run (defaults to run directory names)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for the plot (default: comparison.png in current directory)"
    )
    parser.add_argument(
        "--metric", type=str, default="win_rate",
        choices=["win_rate", "reward_mean", "goal_mean"],
        help="Metric to plot (default: win_rate)"
    )
    parser.add_argument(
        "--no-ci", action="store_true",
        help="Disable confidence interval shading"
    )
    parser.add_argument(
        "--interval", type=int, default=100,
        help="Resample all curves to this epoch interval (default: 100)"
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.runs):
        print(f"[ERROR] --labels count ({len(args.labels)}) must match --runs count ({len(args.runs)})")
        sys.exit(1)

    labels = args.labels or [auto_label(r) for r in args.runs]

    ci_col = {
        "win_rate": "win_ci",
        "reward_mean": "reward_ci",
        "goal_mean": "goal_ci",
    }[args.metric]

    metric_title = {
        "win_rate": "Win Rate",
        "reward_mean": "Mean Reward",
        "goal_mean": "Mean Goals",
    }[args.metric]

    # Load all runs first to determine shared epoch range
    loaded = []
    for run_dir, label in zip(args.runs, labels):
        df = load_eval_results(run_dir)
        if df is not None:
            loaded.append((label, df))

    if not loaded:
        print("[ERROR] No valid eval results found in any of the provided run directories.")
        sys.exit(1)

    max_epoch = min(df["epoch"].max() for _, df in loaded)
    grid = np.arange(args.interval, max_epoch + 1, args.interval)

    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = 0
    for label, df in loaded:
        # Crop to shared range then interpolate onto the common grid
        df = df[df["epoch"] <= max_epoch]
        values = np.interp(grid, df["epoch"].values, df[args.metric].values)
        if ci_col in df.columns:
            cis = np.interp(grid, df["epoch"].values, df[ci_col].values)
        else:
            cis = np.zeros_like(values)

        line, = ax.plot(grid, values, marker="o", linewidth=1.5, markersize=4, label=label)

        if not args.no_ci:
            ax.fill_between(grid, values - cis, values + cis,
                            alpha=0.15, color=line.get_color())
        plotted += 1

    if plotted == 0:
        print("[ERROR] No valid eval results found in any of the provided run directories.")
        sys.exit(1)

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel(metric_title)
    ax.set_title(f"{metric_title} vs Training Epoch")
    if args.metric == "win_rate":
        ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path = args.output or "comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
