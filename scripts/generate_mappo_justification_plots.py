#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SCENARIOS = {
    "counterattack": {
        "title": "Counterattack MAPPO",
        "baseline_dir": RESULTS / "benchmark_academy_counterattack_mappo" / "raw",
        "ours_dir": RESULTS / "llm_mask_academy_counterattack_mappo" / "raw",
    },
    "counterattack_easy": {
        "title": "Counterattack Easy MAPPO",
        "baseline_dir": RESULTS / "benchmark_academy_counterattack_easy_mappo" / "raw",
        "ours_dir": RESULTS / "llm_mask_academy_counterattack_easy_mappo" / "raw",
    },
}

METRICS = [
    ("reward", "Reward", True),
    ("goal_diff", "Goal Difference", True),
    ("my_goal", "Goals Scored", True),
    ("num_shot", "Shots", True),
    ("good_shot", "Good Shots", True),
    ("num_pass", "Passes", True),
    ("good_pass", "Good Passes", True),
    ("get_intercepted", "Got Intercepted", False),
    ("get_tackled", "Got Tackled", False),
]

BASELINE_COLOR = "#1f77b4"
OURS_COLOR = "#d95f02"


def load_metric_csv(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None

    steps: List[float] = []
    values: List[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(float(row["step"]))
                values.append(float(row["value"]))
            except (KeyError, ValueError):
                continue

    if not steps:
        return None

    step_arr = np.asarray(steps, dtype=np.float32)
    val_arr = np.asarray(values, dtype=np.float32)
    order = np.argsort(step_arr)
    step_arr = step_arr[order]
    val_arr = val_arr[order]

    unique_steps = np.unique(step_arr)
    unique_vals = []
    for step in unique_steps:
        idx = np.where(step_arr == step)[0][-1]
        unique_vals.append(val_arr[idx])

    return unique_steps, np.asarray(unique_vals, dtype=np.float32)


def metric_path(raw_dir: Path, metric: str) -> Path:
    return raw_dir / f"RolloutEval__agent_0__agent_0-default-1__{metric}.csv"


def shared_grid(series_list: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    max_step = min(float(steps.max()) for steps, _ in series_list)
    return np.arange(200_000, max_step + 1, 200_000, dtype=np.float32)


def interpolate(series: Tuple[np.ndarray, np.ndarray], grid: np.ndarray) -> np.ndarray:
    steps, values = series
    return np.interp(grid, steps, values)


def plot_metric_grid(
    scenario_key: str,
    title: str,
    baseline_dir: Path,
    ours_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    available = []
    for metric, _, _ in METRICS:
        base = load_metric_csv(metric_path(baseline_dir, metric))
        ours = load_metric_csv(metric_path(ours_dir, metric))
        if base is None or ours is None:
            continue
        available.append((metric, base, ours))

    series_list: List[Tuple[np.ndarray, np.ndarray]] = []
    for _, base, ours in available:
        series_list.append(base)
        series_list.append(ours)
    grid = shared_grid(series_list)
    epochs = grid / 2000.0

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    axes = axes.ravel()
    peak_rows = []

    for ax, (metric, base, ours), (_, metric_title, higher_better) in zip(
        axes,
        available,
        [m for m in METRICS if any(m[0] == x[0] for x in available)],
    ):
        base_y = interpolate(base, grid)
        ours_y = interpolate(ours, grid)

        ax.plot(epochs, base_y, color=BASELINE_COLOR, linewidth=2.0, label="Baseline")
        ax.plot(epochs, ours_y, color=OURS_COLOR, linewidth=2.0, label="LLM Mask")
        ax.set_title(metric_title, fontsize=11)
        ax.grid(True, alpha=0.25)

        peak_base = float(np.max(base_y))
        peak_ours = float(np.max(ours_y))
        signed_delta = peak_ours - peak_base
        effective_delta = signed_delta if higher_better else -signed_delta
        peak_rows.append(
            {
                "metric": metric,
                "title": metric_title,
                "peak_baseline": peak_base,
                "peak_ours": peak_ours,
                "effective_delta": effective_delta,
                "higher_better": higher_better,
            }
        )

    for idx in range(len(available), len(axes)):
        axes[idx].axis("off")

    axes[0].legend(loc="upper left")
    fig.suptitle(f"{title}: Evaluation Metrics Over Training", fontsize=15, fontweight="bold")
    fig.supxlabel("Approx. Training Epoch")
    fig.tight_layout()
    output_path = output_dir / f"{scenario_key}_mappo_eval_metrics.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    peak_rows.sort(key=lambda row: row["effective_delta"], reverse=True)
    bar_fig, bar_ax = plt.subplots(figsize=(10, 5.5))
    labels = [row["title"] for row in peak_rows]
    vals = [row["effective_delta"] for row in peak_rows]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    y = np.arange(len(labels))
    bar_ax.barh(y, vals, color=colors)
    bar_ax.axvline(0.0, color="black", linewidth=1.0)
    bar_ax.set_yticks(y)
    bar_ax.set_yticklabels(labels)
    bar_ax.invert_yaxis()
    bar_ax.grid(True, axis="x", alpha=0.25)
    bar_ax.set_xlabel("Peak Metric Delta vs Baseline (positive = better for our method)")
    bar_ax.set_title(f"{title}: Peak Metric Scorecard", fontsize=14, fontweight="bold")
    for yi, val in zip(y, vals):
        bar_ax.text(val, yi, f" {val:+.2f}", va="center", ha="left" if val >= 0 else "right")
    bar_fig.tight_layout()
    scorecard_path = output_dir / f"{scenario_key}_mappo_peak_delta_scorecard.png"
    bar_fig.savefig(scorecard_path, dpi=160, bbox_inches="tight")
    plt.close(bar_fig)

    return {
        "scenario": scenario_key,
        "metric_plot": str(output_path),
        "scorecard_plot": str(scorecard_path),
        "metrics": peak_rows,
    }


def main() -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS / f"mappo_justification_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "output_dir": str(output_dir),
        "scenarios": [],
    }

    for scenario_key, cfg in SCENARIOS.items():
        manifest["scenarios"].append(
            plot_metric_grid(
                scenario_key=scenario_key,
                title=cfg["title"],
                baseline_dir=cfg["baseline_dir"],
                ours_dir=cfg["ours_dir"],
                output_dir=output_dir,
            )
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved MAPPO justification plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
