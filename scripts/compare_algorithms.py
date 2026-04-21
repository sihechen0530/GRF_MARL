#!/usr/bin/env python3
"""
Compare training curves across different algorithms on the same scenario.

Scans all experiment directories matching a scenario pattern, aggregates
runs per algorithm, and produces overlay plots for direct comparison.

Usage:
    # Compare all 3v1 algorithms
    python scripts/compare_algorithms.py \
        --log_dir logs/gr_football \
        --scenario 3_vs_1_with_keeper

    # Specify algorithms explicitly
    python scripts/compare_algorithms.py \
        --log_dir logs/gr_football \
        --scenario 3_vs_1_with_keeper \
        --algorithms mappo ippo happo

    # Custom output
    python scripts/compare_algorithms.py \
        --log_dir logs/gr_football \
        --scenario 3_vs_1_with_keeper \
        --output_dir results/3v1_comparison
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

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def linestyle_for_algorithm(algo_name: str, dashed_substrings) -> str:
    """Solid unless ``algo_name`` contains any of ``dashed_substrings`` (case-insensitive)."""
    if not dashed_substrings:
        return "-"
    low = algo_name.lower()
    for s in dashed_substrings:
        if s and str(s).lower() in low:
            return "--"
    return "-"


def discover_experiments(log_dir, scenario, algorithms=None):
    """
    Discover experiment directories matching a scenario.

    Looks for directories like:
        log_dir/benchmark_academy_<scenario>_<algorithm>/
    """
    if not os.path.isdir(log_dir):
        print(f"[ERROR] Log directory not found: {log_dir}")
        sys.exit(1)

    experiments = {}
    for entry in sorted(os.listdir(log_dir)):
        full_path = os.path.join(log_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if scenario not in entry:
            continue

        algo_name = entry.split(scenario + "_")[-1] if scenario + "_" in entry else entry
        if algorithms and algo_name not in algorithms:
            continue

        runs = []
        for run_entry in sorted(os.listdir(full_path)):
            run_dir = os.path.join(full_path, run_entry)
            if os.path.isdir(run_dir):
                runs.append(run_dir)

        if runs:
            experiments[algo_name] = {
                "expr_name": entry,
                "runs": runs,
            }

    if not experiments:
        print(f"[ERROR] No experiments found for scenario '{scenario}' in {log_dir}")
        print("Available directories:")
        for d in sorted(os.listdir(log_dir)):
            if os.path.isdir(os.path.join(log_dir, d)):
                print(f"  - {d}")
        sys.exit(1)

    print(f"Found {len(experiments)} algorithm(s) for scenario '{scenario}':")
    for algo, info in experiments.items():
        print(f"  - {algo}: {len(info['runs'])} run(s)")

    return experiments


def parse_tb_events_for_metric(run_dirs, metric_name="win"):
    """Parse TensorBoard events from multiple runs, extracting a specific metric."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[ERROR] tensorboard is required. Install with: pip install tensorboard")
        sys.exit(1)

    runs_data = []
    for run_dir in run_dirs:
        pattern = os.path.join(run_dir, "**", "events.out.tfevents.*")
        event_files = glob.glob(pattern, recursive=True)
        if not event_files:
            continue

        event_dir = os.path.dirname(event_files[0])
        ea = EventAccumulator(event_dir)
        ea.Reload()

        tags = ea.Tags().get("scalars", [])
        target_tags = [t for t in tags if t.endswith(f"/{metric_name}")]

        for tag in target_tags:
            category = tag.split("/")[0]
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            runs_data.append({
                "category": category,
                "steps": np.array(steps),
                "values": np.array(values),
            })
            break  # take the first matching tag per run

    return runs_data


def interpolate_and_aggregate(runs_data):
    """Interpolate runs onto a common step grid and compute mean ± std."""
    if not runs_data:
        return None

    all_steps = set()
    for rd in runs_data:
        all_steps.update(rd["steps"].tolist())
    common_steps = np.array(sorted(all_steps))

    interpolated = []
    for rd in runs_data:
        interp_values = np.interp(common_steps, rd["steps"], rd["values"])
        interpolated.append(interp_values)

    interpolated = np.array(interpolated)
    return {
        "steps": common_steps,
        "mean": np.mean(interpolated, axis=0),
        "std": np.std(interpolated, axis=0) if len(interpolated) > 1 else np.zeros(len(common_steps)),
        "n_runs": len(interpolated),
    }


def smooth(values, window=5):
    """Simple moving average smoothing."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def main():
    parser = argparse.ArgumentParser(description="Compare algorithms on the same scenario")
    parser.add_argument("--log_dir", type=str, default="logs/gr_football",
                        help="Root log directory")
    parser.add_argument("--scenario", type=str, required=True,
                        help="Scenario name pattern (e.g. 3_vs_1_with_keeper)")
    parser.add_argument("--algorithms", nargs="*", default=None,
                        help="Filter to specific algorithms (e.g. mappo ippo happo)")
    parser.add_argument("--metrics", nargs="*", default=["win", "reward", "score"],
                        help="Metrics to compare (default: win reward score)")
    parser.add_argument("--smooth_window", type=int, default=1,
                        help="Smoothing window size (default: 1 = no smoothing)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument(
        "--dashed-substr",
        nargs="*",
        default=["llm"],
        metavar="SUBSTR",
        help="Algorithm key substring(s); if any appear in the discovered algo name "
        "(case-insensitive), that curve uses a dashed line. Default: llm. "
        "Pass multiple tokens, e.g. --dashed-substr llm eureka_llm.",
    )
    args = parser.parse_args()

    dashed_substrs = tuple(s for s in (args.dashed_substr or []) if s)

    if args.output_dir is None:
        args.output_dir = os.path.join("results", f"compare_{args.scenario}")
    os.makedirs(args.output_dir, exist_ok=True)

    experiments = discover_experiments(args.log_dir, args.scenario, args.algorithms)

    for metric in args.metrics:
        print(f"\nProcessing metric: {metric}")

        algo_aggregated = {}
        for algo, info in experiments.items():
            runs_data = parse_tb_events_for_metric(info["runs"], metric)
            if not runs_data:
                print(f"  [WARN] No '{metric}' data for {algo}")
                continue
            agg = interpolate_and_aggregate(runs_data)
            if agg is not None:
                algo_aggregated[algo] = agg
                print(f"  {algo}: {agg['n_runs']} run(s), "
                      f"final {metric} = {agg['mean'][-1]:.4f} ± {agg['std'][-1]:.4f}")

        if not algo_aggregated:
            print(f"  No data for metric '{metric}'")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (algo, agg) in enumerate(sorted(algo_aggregated.items())):
            color = COLORS[i % len(COLORS)]
            ls = linestyle_for_algorithm(algo, dashed_substrs)
            mean = smooth(agg["mean"], args.smooth_window)
            std = smooth(agg["std"], args.smooth_window)

            label = f"{algo.upper()} (n={agg['n_runs']})"
            print(f"  plot: {algo} -> linestyle={ls!r} (dashed if match {dashed_substrs!r})")
            ax.plot(
                agg["steps"],
                mean,
                color=color,
                linestyle=ls,
                linewidth=1.5,
                label=label,
            )
            ax.fill_between(
                agg["steps"], mean - std, mean + std,
                color=color, alpha=0.15,
            )

        ax.set_xlabel("Global Step")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{args.scenario}: {metric.replace('_', ' ').title()} Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)

        if metric in ("win", "score", "lose"):
            ax.set_ylim(-0.05, 1.05)

        fig_path = os.path.join(args.output_dir, f"compare_{metric}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig_path}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Algorithm':>12s}", end="")
    for metric in args.metrics:
        print(f" | {metric:>18s}", end="")
    print(f" | {'runs':>4s}")
    print("-" * 80)

    for algo in sorted(experiments.keys()):
        print(f"{algo.upper():>12s}", end="")
        for metric in args.metrics:
            runs_data = parse_tb_events_for_metric(experiments[algo]["runs"], metric)
            agg = interpolate_and_aggregate(runs_data) if runs_data else None
            if agg:
                print(f" | {agg['mean'][-1]:>7.4f} ± {agg['std'][-1]:<7.4f}", end="")
            else:
                print(f" | {'N/A':>18s}", end="")
        n = len(experiments[algo]["runs"])
        print(f" | {n:>4d}")
    print("=" * 80)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
