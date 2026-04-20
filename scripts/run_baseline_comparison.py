#!/usr/bin/env python3
"""
Generate per-scenario baseline comparison plots for corner, counterattack,
counterattack_easy, and 11vs11, comparing IPPO / MAPPO / MAT algorithms.

For each (scenario, algorithm) pair, all non-warmup timestamped run directories
are scanned; the one with the highest peak win_rate in eval_results.csv is
selected as the representative run.

Outputs (default: results/baselines/):
  corner.png
  counterattack.png
  counterattack_easy.png
  11vs11.png
  manifest.json   -- records which run_dir was chosen for every entry

Usage:
    python scripts/run_baseline_comparison.py
    python scripts/run_baseline_comparison.py --output-dir results/baselines --interval 50
"""

import argparse
import json
import os
import pathlib
import sys
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

GROUPS = {
    "corner": {
        "IPPO":  "benchmark_academy_corner_ippo",
        "MAPPO": "benchmark_academy_corner_mappo",
        "MAT":   "benchmark_academy_corner_mat",
    },
    "counterattack": {
        "IPPO":  "benchmark_academy_counterattack_ippo",
        "MAPPO": "benchmark_academy_counterattack_mappo",
        "MAT":   "benchmark_academy_counterattack_mat",
    },
    "counterattack_easy": {
        "IPPO":  "benchmark_academy_counterattack_easy_ippo",
        "MAPPO": "benchmark_academy_counterattack_easy_mappo",
        "MAT":   "benchmark_academy_counterattack_easy_mat",
    },
    "11vs11": {
        # No IPPO baseline exists for 11v11
        "MAPPO": "full_game_11_v_11_hard_mappo",
        "MAT":   "benchmark_full_game_11_v_11_hard_mat",
    },
}

SCENARIO_TITLES = {
    "corner":            "Academy Corner",
    "counterattack":     "Academy Counterattack (Hard)",
    "counterattack_easy":"Academy Counterattack (Easy)",
    "11vs11":            "11 vs 11 (Hard)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_csv(run_dir: pathlib.Path) -> Optional[pd.DataFrame]:
    """Load eval_results.csv from a run directory; return None if absent/empty."""
    csv = run_dir / "eval_results.csv"
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv)
        df = df[df["epoch"] >= 0].sort_values("epoch").reset_index(drop=True)
        return df if not df.empty else None
    except Exception as exc:
        print(f"  [WARN] Could not read {csv}: {exc}")
        return None


def find_best_run(log_base: pathlib.Path, expr_name: str) -> Tuple:
    """
    Scan all timestamped subdirs of log_base/expr_name (excluding warmup dirs).
    Return (best_run_dir, peak_win_rate, dataframe) for the run whose
    eval_results.csv has the highest peak win_rate.
    Returns (None, -1.0, None) if nothing is found.
    """
    expr_dir = log_base / expr_name
    if not expr_dir.exists():
        return None, -1.0, None

    best_dir, best_peak, best_df = None, -1.0, None

    for run_dir in sorted(expr_dir.glob("????-??-??-??-??-??"), key=lambda p: p.name):
        df = load_eval_csv(run_dir)
        if df is None:
            continue
        peak = float(df["win_rate"].max())
        if peak > best_peak:
            best_peak = peak
            best_dir = run_dir
            best_df = df

    return best_dir, best_peak, best_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scenario(
    scenario: str,
    algo_entries: dict,        # label -> (run_dir, df)
    output_path: pathlib.Path,
    metric: str = "win_rate",
    interval: int = 50,
) -> None:
    """Plot overlaid metric curves for all algorithms in one scenario."""
    ci_col = {
        "win_rate":    "win_ci",
        "reward_mean": "reward_ci",
        "goal_mean":   "goal_ci",
    }[metric]
    metric_title = {
        "win_rate":    "Win Rate",
        "reward_mean": "Mean Reward",
        "goal_mean":   "Mean Goals",
    }[metric]

    loaded = [(label, df) for label, (_, df) in algo_entries.items() if df is not None]
    if not loaded:
        print(f"  [WARN] No valid data for {scenario} — skipping plot.")
        return

    max_epoch = min(df["epoch"].max() for _, df in loaded)
    grid = np.arange(interval, max_epoch + 1, interval)
    if len(grid) == 0:
        print(f"  [WARN] Epoch range too small for interval={interval} in {scenario}.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, df in loaded:
        df_crop = df[df["epoch"] <= max_epoch]
        values = np.interp(grid, df_crop["epoch"].values, df_crop[metric].values)
        if ci_col in df_crop.columns:
            cis = np.interp(grid, df_crop["epoch"].values, df_crop[ci_col].values)
        else:
            cis = np.zeros_like(values)
        line, = ax.plot(grid, values, marker="o", linewidth=1.5, markersize=3, label=label)
        ax.fill_between(grid, values - cis, values + cis, alpha=0.15, color=line.get_color())

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel(metric_title)
    title = SCENARIO_TITLES.get(scenario, scenario)
    ax.set_title(f"{title} — {metric_title}")
    if metric == "win_rate":
        ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline comparison plots across scenarios and algorithms"
    )
    parser.add_argument(
        "--log-dir", default="logs/gr_football",
        help="Root log directory (default: logs/gr_football)",
    )
    parser.add_argument(
        "--output-dir", default="results/baselines",
        help="Output directory for plots and manifest (default: results/baselines)",
    )
    parser.add_argument(
        "--metric", default="win_rate",
        choices=["win_rate", "reward_mean", "goal_mean"],
    )
    parser.add_argument(
        "--interval", type=int, default=50,
        help="Epoch sampling interval for plot curves (default: 50)",
    )
    args = parser.parse_args()

    log_base = pathlib.Path(args.log_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}

    for scenario, algos in GROUPS.items():
        print(f"\n{'='*55}")
        print(f"Scenario: {SCENARIO_TITLES.get(scenario, scenario)}")
        print(f"{'='*55}")

        algo_entries: dict = {}   # label -> (run_dir, df)
        manifest[scenario] = {}

        for label, expr_name in algos.items():
            run_dir, peak, df = find_best_run(log_base, expr_name)
            if run_dir is None:
                print(f"  {label:6s}: no eval_results.csv found — skipped")
                manifest[scenario][label] = None
                algo_entries[label] = (None, None)
                continue

            max_ep = int(df["epoch"].max())
            print(f"  {label:6s}: {run_dir}  peak_wr={peak:.4f}  max_ep={max_ep}")
            algo_entries[label] = (run_dir, df)
            manifest[scenario][label] = {
                "expr_name":     expr_name,
                "run_dir":       str(run_dir),
                "peak_win_rate": round(peak, 4),
                "max_epoch":     max_ep,
            }

        plot_path = output_dir / f"{scenario}.png"
        plot_scenario(scenario, algo_entries, plot_path, args.metric, args.interval)

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
