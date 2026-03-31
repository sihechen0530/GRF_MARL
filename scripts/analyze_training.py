#!/usr/bin/env python3
"""
Analyze training logs from TensorBoard event files.

Discovers all runs for a given experiment, extracts metrics (win rate, reward,
etc.), aggregates across runs, exports to CSV (aggregated + raw), and generates
one PNG per metric.

Usage:
    python scripts/analyze_training.py --log_dir logs/gr_football \
        --expr_name benchmark_academy_3_vs_1_with_keeper_mappo

    python scripts/analyze_training.py --log_dir logs/gr_football \
        --expr_name benchmark_academy_3_vs_1_with_keeper_mappo \
        --output_dir results/3v1_mappo

    python scripts/analyze_training.py --log_dir logs/gr_football \
        --expr_name benchmark_academy_3_vs_1_with_keeper_mappo \
        --metrics win reward score
"""

import argparse
import os
import sys
import glob
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def discover_runs(log_dir, expr_name):
    """Find all timestamped run directories for a given experiment."""
    expr_dir = os.path.join(log_dir, expr_name)
    if not os.path.isdir(expr_dir):
        print(f"[ERROR] Experiment directory not found: {expr_dir}")
        print(f"Available experiments in {log_dir}:")
        if os.path.isdir(log_dir):
            for d in sorted(os.listdir(log_dir)):
                if os.path.isdir(os.path.join(log_dir, d)):
                    print(f"  - {d}")
        sys.exit(1)

    runs = []
    for entry in sorted(os.listdir(expr_dir)):
        run_dir = os.path.join(expr_dir, entry)
        if os.path.isdir(run_dir):
            runs.append(run_dir)

    if not runs:
        print(f"[ERROR] No run directories found in {expr_dir}")
        sys.exit(1)

    print(f"Found {len(runs)} run(s) for '{expr_name}':")
    for r in runs:
        print(f"  - {os.path.basename(r)}")
    return runs


def find_event_files(run_dir):
    """Find all TensorBoard event files in a run directory."""
    pattern = os.path.join(run_dir, "**", "events.out.tfevents.*")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def parse_tb_events(event_files, metric_filter=None):
    """
    Parse TensorBoard event files and extract scalar data.
    Returns a dict: {tag: [(wall_time, step, value), ...]}
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[ERROR] tensorboard is required. Install with: pip install tensorboard")
        sys.exit(1)

    all_scalars = defaultdict(list)

    # Deduplicate by directory: EventAccumulator loads all event files in a
    # directory at once, so passing the same directory multiple times wastes time.
    unique_dirs = list(dict.fromkeys(os.path.dirname(f) for f in event_files))

    for event_dir in unique_dirs:
        ea = EventAccumulator(event_dir)
        ea.Reload()

        tags = ea.Tags().get("scalars", [])
        for tag in tags:
            if metric_filter and not any(m in tag for m in metric_filter):
                continue
            events = ea.Scalars(tag)
            for e in events:
                all_scalars[tag].append((e.wall_time, e.step, e.value))

    for tag in all_scalars:
        all_scalars[tag].sort(key=lambda x: x[1])

    return dict(all_scalars)


def classify_tags(tags):
    """Classify tags into categories by their top-level prefix."""
    categories = defaultdict(list)
    for tag in tags:
        parts = tag.split("/")
        if parts:
            categories[parts[0]].append(tag)
    return dict(categories)


def extract_metric_name(tag):
    """Extract the metric name from a TensorBoard tag: Rollout/agent_0/pid/win -> win"""
    parts = tag.rstrip("/").split("/")
    return parts[-1] if parts else tag


def aggregate_runs(all_runs_data):
    """
    Aggregate scalar data across multiple runs.
    Returns: {tag: DataFrame with columns [step, mean, std, min, max, n_runs]}
    """
    tag_data = defaultdict(lambda: defaultdict(list))

    for run_data in all_runs_data:
        for tag, events in run_data.items():
            for wall_time, step, value in events:
                tag_data[tag][step].append(value)

    aggregated = {}
    for tag, step_values in tag_data.items():
        rows = []
        for step in sorted(step_values.keys()):
            values = step_values[step]
            rows.append({
                "step": step,
                "mean": np.mean(values),
                "std": np.std(values) if len(values) > 1 else 0.0,
                "min": np.min(values),
                "max": np.max(values),
                "n_runs": len(values),
            })
        aggregated[tag] = pd.DataFrame(rows)

    return aggregated


def export_csv(aggregated, all_runs_data, runs, output_dir):
    """
    Export two sets of CSV files:
      - aggregated/<category>.csv  — mean/std/min/max across runs per step
      - raw/<tag_sanitised>.csv    — raw per-run values (run_id, step, value)
    """
    os.makedirs(output_dir, exist_ok=True)
    agg_dir = os.path.join(output_dir, "aggregated")
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(agg_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # --- aggregated CSVs (one per category) ---
    categories = classify_tags(aggregated.keys())
    for category, tags in categories.items():
        rows = []
        for tag in tags:
            metric = extract_metric_name(tag)
            df = aggregated[tag]
            for _, row in df.iterrows():
                rows.append({
                    "tag": tag,
                    "metric": metric,
                    "step": int(row["step"]),
                    "mean": row["mean"],
                    "std": row["std"],
                    "min": row["min"],
                    "max": row["max"],
                    "n_runs": int(row["n_runs"]),
                })
        out_df = pd.DataFrame(rows)
        csv_path = os.path.join(agg_dir, f"{category}.csv")
        out_df.to_csv(csv_path, index=False)
        print(f"  Aggregated CSV: {csv_path} ({len(out_df)} rows)")

    # --- raw CSVs (one per tag) ---
    run_names = [os.path.basename(r) for r in runs]
    all_tags = set()
    for run_data in all_runs_data:
        all_tags.update(run_data.keys())

    for tag in sorted(all_tags):
        rows = []
        for run_idx, run_data in enumerate(all_runs_data):
            run_name = run_names[run_idx] if run_idx < len(run_names) else f"run_{run_idx}"
            if tag not in run_data:
                continue
            for wall_time, step, value in run_data[tag]:
                rows.append({"run": run_name, "step": step, "value": value, "wall_time": wall_time})
        if not rows:
            continue
        safe_name = tag.replace("/", "__")
        csv_path = os.path.join(raw_dir, f"{safe_name}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"  Raw CSVs saved to: {raw_dir}/")


def save_single_plot(df, metric, tag, output_dir, x_axis_label, title_prefix, category):
    """Save one PNG for a single metric."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["step"], df["mean"], linewidth=1.5, label="mean")
    if df["n_runs"].iloc[0] > 1:
        ax.fill_between(
            df["step"],
            df["mean"] - df["std"],
            df["mean"] + df["std"],
            alpha=0.25,
            label="±1 std",
        )
        ax.legend(fontsize=9)

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(metric)
    ax.set_title(f"{title_prefix}[{category}] {metric}")
    ax.grid(True, alpha=0.3)

    if metric in ("win", "lose", "score"):
        ax.set_ylim(-0.05, 1.05)

    safe_metric = metric.replace("/", "_")
    fig_path = os.path.join(output_dir, f"{category}__{safe_metric}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_metrics(aggregated, output_dir, x_axis_label="Global Step", title_prefix=""):
    """Generate one PNG per metric."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    categories = classify_tags(aggregated.keys())

    key_metrics = {"win", "reward", "score", "lose", "my_goal", "goal_diff"}
    training_metrics = {"policy_loss", "value_loss", "entropy", "approx_kl"}

    saved = []
    for category, tags in categories.items():
        if "Timer" in category:
            continue

        for tag in tags:
            metric = extract_metric_name(tag)

            if category in ("Rollout", "RolloutEval") and metric not in key_metrics:
                continue
            if category == "Training" and metric not in training_metrics:
                continue

            df = aggregated[tag]
            path = save_single_plot(df, metric, tag, plots_dir, x_axis_label, title_prefix, category)
            saved.append(path)
            print(f"  Saved: {path}")

    # Dedicated combined win-rate plot (Rollout + RolloutEval overlaid)
    win_tags = [t for t in aggregated if "win" in t.lower() and "Timer" not in t]
    if len(win_tags) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        for tag in win_tags:
            df = aggregated[tag]
            label = tag.split("/")[0]
            ax.plot(df["step"], df["mean"], linewidth=1.5, label=label)
            if df["n_runs"].iloc[0] > 1:
                ax.fill_between(
                    df["step"],
                    df["mean"] - df["std"],
                    df["mean"] + df["std"],
                    alpha=0.2,
                )
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel("Win Rate")
        ax.set_title(f"{title_prefix}Win Rate (Train vs Eval)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        fig_path = os.path.join(plots_dir, "win_rate_combined.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(fig_path)
        print(f"  Saved: {fig_path}")

    return saved


def print_summary(aggregated):
    """Print a summary table of final metric values."""
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY (final values)")
    print("=" * 70)

    for tag, df in sorted(aggregated.items()):
        if "Timer" in tag:
            continue
        metric = extract_metric_name(tag)
        category = tag.split("/")[0]
        last = df.iloc[-1]
        n_runs = int(last["n_runs"])
        if n_runs > 1:
            print(f"  [{category}] {metric:20s}: {last['mean']:.4f} ± {last['std']:.4f}  (n={n_runs}, step={int(last['step'])})")
        else:
            print(f"  [{category}] {metric:20s}: {last['mean']:.4f}  (step={int(last['step'])})")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze training logs from TensorBoard events")
    parser.add_argument("--log_dir", type=str, default="logs/gr_football",
                        help="Root log directory (default: logs/gr_football)")
    parser.add_argument("--expr_name", type=str, required=True,
                        help="Experiment name (subdirectory under log_dir)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for CSVs and plots (default: results/<expr_name>)")
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="Filter to specific metrics (e.g. win reward score)")
    parser.add_argument("--x_axis", type=str, default="global_step",
                        choices=["global_step", "epoch"],
                        help="X-axis type (default: global_step)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join("results", args.expr_name)

    runs = discover_runs(args.log_dir, args.expr_name)

    print("\nParsing TensorBoard events...")
    all_runs_data = []
    valid_runs = []
    for run_dir in runs:
        event_files = find_event_files(run_dir)
        if not event_files:
            print(f"  [WARN] No event files in {run_dir}, skipping")
            continue
        run_data = parse_tb_events(event_files, metric_filter=args.metrics)
        if run_data:
            all_runs_data.append(run_data)
            valid_runs.append(run_dir)
            print(f"  Parsed {os.path.basename(run_dir)}: {len(run_data)} tags")
        else:
            print(f"  [WARN] No scalar data in {run_dir}")

    if not all_runs_data:
        print("[ERROR] No data found across any runs")
        sys.exit(1)

    print(f"\nAggregating {len(all_runs_data)} run(s)...")
    aggregated = aggregate_runs(all_runs_data)

    x_label = "Global Step" if args.x_axis == "global_step" else "Rollout Epoch"

    print("\nExporting CSVs...")
    export_csv(aggregated, all_runs_data, valid_runs, args.output_dir)

    print_summary(aggregated)

    if not args.no_plot:
        print("\nGenerating plots (one per metric)...")
        plot_metrics(aggregated, args.output_dir, x_axis_label=x_label,
                     title_prefix=f"{args.expr_name}\n")

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
