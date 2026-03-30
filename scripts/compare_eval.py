#!/usr/bin/env python3
"""
Compare evaluation results across multiple experiments on a single chart.

Reads eval_results.csv files produced by eval_checkpoint.py and overlays
win-rate curves (with 95% CI error bars) for easy visual comparison.

Usage:
    # Compare baseline, POC, and LLM for counterattack_easy × IPPO
    python scripts/compare_eval.py \
        --csv results/eval_ce_ippo_baseline/eval_results.csv "Baseline" \
             results/eval_ce_ippo_poc/eval_results.csv "POC (hand-written φ)" \
             results/eval_ce_ippo_llm/eval_results.csv "LLM (DeepSeek φ)" \
        --title "counterattack_easy  ×  IPPO" \
        --output results/compare_ce_ippo.png

    # Compare with TensorBoard training curves instead of eval CSVs
    python scripts/compare_eval.py \
        --tb logs/gr_football \
        --expr benchmark_academy_counterattack_easy_ippo "Baseline" \
               benchmark_academy_counterattack_easy_ippo_eureka_poc "POC" \
               benchmark_academy_counterattack_easy_ippo_eureka_llm "LLM" \
        --title "counterattack_easy × IPPO (training)" \
        --output results/compare_ce_ippo_training.png

    # Cap x-axis range when one run has many more eval checkpoints than others
    python scripts/compare_eval.py --csv ... --max_epoch 1500 --title "..." --output ...
"""

import argparse
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def read_eval_csv(path):
    """Read eval_results.csv and return rows sorted by epoch (epoch >= 0 only)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            epoch = int(r["epoch"])
            if epoch < 0:
                continue
            rows.append({
                "epoch": epoch,
                "win_rate": float(r["win_rate"]),
                "win_ci": float(r["win_ci"]),
                "reward_mean": float(r["reward_mean"]),
                "reward_ci": float(r["reward_ci"]),
                "goal_mean": float(r["goal_mean"]),
                "goal_ci": float(r["goal_ci"]),
            })
    rows.sort(key=lambda r: r["epoch"])
    return rows


def read_tb_win_rate(log_dir, expr_name):
    """Read win-rate from TensorBoard event files for a given experiment."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[ERROR] tensorboard is required for --tb mode. Install with: pip install tensorboard")
        sys.exit(1)

    import glob as globmod
    expr_dir = os.path.join(log_dir, expr_name)
    if not os.path.isdir(expr_dir):
        print(f"[ERROR] Experiment dir not found: {expr_dir}")
        sys.exit(1)

    all_events = []
    for run_dir in sorted(os.listdir(expr_dir)):
        run_path = os.path.join(expr_dir, run_dir)
        if not os.path.isdir(run_path):
            continue
        event_files = globmod.glob(os.path.join(run_path, "**", "events.out.tfevents.*"), recursive=True)
        for ef in event_files:
            ea = EventAccumulator(os.path.dirname(ef))
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                if "win" in tag.lower() and "RolloutEval" in tag:
                    for e in ea.Scalars(tag):
                        all_events.append((e.step, e.value))

    if not all_events:
        for run_dir in sorted(os.listdir(expr_dir)):
            run_path = os.path.join(expr_dir, run_dir)
            if not os.path.isdir(run_path):
                continue
            event_files = globmod.glob(os.path.join(run_path, "**", "events.out.tfevents.*"), recursive=True)
            for ef in event_files:
                ea = EventAccumulator(os.path.dirname(ef))
                ea.Reload()
                for tag in ea.Tags().get("scalars", []):
                    if "win" in tag.lower() and "Rollout" in tag:
                        for e in ea.Scalars(tag):
                            all_events.append((e.step, e.value))

    if not all_events:
        print(f"[WARN] No win-rate data found for {expr_name}")
        return []

    from collections import defaultdict
    step_vals = defaultdict(list)
    for step, val in all_events:
        step_vals[step].append(val)

    rows = []
    for step in sorted(step_vals):
        vals = step_vals[step]
        rows.append({
            "epoch": step,
            "win_rate": np.mean(vals),
            "win_ci": np.std(vals) if len(vals) > 1 else 0.0,
        })
    return rows


def truncate_series(rows, max_epoch):
    """Keep only points with epoch <= max_epoch (inclusive)."""
    if max_epoch is None:
        return rows
    out = [r for r in rows if r["epoch"] <= max_epoch]
    return out


def plot_comparison(series_list, title, output_path, ylabel="Win Rate", metric="win_rate", ci_key="win_ci"):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, rows) in enumerate(series_list):
        if not rows:
            print(f"  [WARN] No data for '{label}', skipping")
            continue
        epochs = [r["epoch"] for r in rows]
        values = [r[metric] for r in rows]
        cis = [r.get(ci_key, 0.0) for r in rows]
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        ax.errorbar(epochs, values, yerr=cis, fmt=f"{marker}-", capsize=3,
                     linewidth=1.8, markersize=5, color=color, label=label)

    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    if metric == "win_rate":
        ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def parse_pairs(flat_list):
    """Parse [path, label, path, label, ...] into [(path, label), ...]."""
    if len(flat_list) % 2 != 0:
        print("[ERROR] --csv / --expr arguments must be <path> <label> pairs")
        sys.exit(1)
    return [(flat_list[i], flat_list[i + 1]) for i in range(0, len(flat_list), 2)]


def main():
    parser = argparse.ArgumentParser(
        description="Compare eval results across experiments")
    parser.add_argument("--csv", nargs="*", default=None,
                        help="Pairs of: <eval_results.csv> <legend label>")
    parser.add_argument("--tb", type=str, default=None,
                        help="TensorBoard log root (e.g. logs/gr_football)")
    parser.add_argument("--expr", nargs="*", default=None,
                        help="Pairs of: <expr_name> <legend label> (requires --tb)")
    parser.add_argument("--title", type=str, default="Win Rate Comparison")
    parser.add_argument("--output", type=str, default="results/comparison.png")
    parser.add_argument("--metric", type=str, default="win_rate",
                        choices=["win_rate", "reward_mean", "goal_mean"],
                        help="Which metric to plot (default: win_rate)")
    parser.add_argument("--max_epoch", type=float, default=None,
                        help="Only plot points with epoch <= this value (all series). "
                             "Useful when one run has many more checkpoints than others.")
    args = parser.parse_args()

    series = []

    if args.csv:
        pairs = parse_pairs(args.csv)
        for csv_path, label in pairs:
            if not os.path.isfile(csv_path):
                print(f"[ERROR] CSV not found: {csv_path}")
                sys.exit(1)
            rows = read_eval_csv(csv_path)
            n_raw = len(rows)
            rows = truncate_series(rows, args.max_epoch)
            if args.max_epoch is not None:
                print(f"  Loaded {csv_path}: {n_raw} epoch data points (after --max_epoch {args.max_epoch}: {len(rows)})")
            else:
                print(f"  Loaded {csv_path}: {len(rows)} epoch data points")
            series.append((label, rows))

    if args.expr:
        if not args.tb:
            print("[ERROR] --expr requires --tb <log_dir>")
            sys.exit(1)
        pairs = parse_pairs(args.expr)
        for expr_name, label in pairs:
            rows = read_tb_win_rate(args.tb, expr_name)
            n_raw = len(rows)
            rows = truncate_series(rows, args.max_epoch)
            if args.max_epoch is not None:
                print(f"  Loaded TB {expr_name}: {n_raw} data points (after --max_epoch {args.max_epoch}: {len(rows)})")
            else:
                print(f"  Loaded TB {expr_name}: {len(rows)} data points")
            series.append((label, rows))

    if not series:
        print("[ERROR] Provide at least one data source via --csv or --expr")
        sys.exit(1)

    ylabel_map = {"win_rate": "Win Rate", "reward_mean": "Reward", "goal_mean": "Goals"}
    ci_map = {"win_rate": "win_ci", "reward_mean": "reward_ci", "goal_mean": "goal_ci"}

    plot_comparison(series, args.title, args.output,
                    ylabel=ylabel_map[args.metric],
                    metric=args.metric,
                    ci_key=ci_map[args.metric])


if __name__ == "__main__":
    main()
