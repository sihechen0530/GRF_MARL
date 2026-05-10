#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover - runtime environment guard
    raise SystemExit(
        "tensorboard is required. Run this script in an environment with tensorboard "
        "installed, e.g. `conda run -n grf_env python ...`."
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results"
EXTERNAL_LOG_ROOT = Path("/home/zhang.kaiqi1/GRF_MARL/logs/gr_football")

SCENARIO_TITLE = {
    "corner": "Corner",
    "counterattack": "Counterattack",
    "counterattack_easy": "Counterattack Easy",
    "full_game_11v11_hard": "11v11 Hard",
}

ALGO_COLOR = {
    "ippo": "#2e7d32",
    "mappo": "#1565c0",
    "mat": "#8e24aa",
}

VARIANT_LABEL = {
    "baseline": "Baseline",
    "action_mask": "Action Masking",
    "reward_llm": "Reward Shaping",
}

VARIANT_HATCH = {
    "baseline": "",
    "action_mask": "//",
    "reward_llm": "..",
}

REWARD_EXPERIMENTS = {
    ("corner", "ippo"): "benchmark_academy_corner_ippo_eureka_llm",
    ("corner", "mappo"): "benchmark_academy_corner_mappo_eureka_llm",
    ("corner", "mat"): "benchmark_academy_corner_mat_eureka_llm",
    ("counterattack", "ippo"): "benchmark_academy_counterattack_ippo_eureka_llm",
    ("counterattack", "mappo"): "benchmark_academy_counterattack_mappo_eureka_llm",
    ("counterattack_easy", "ippo"): "benchmark_academy_counterattack_easy_ippo_eureka_llm",
    ("counterattack_easy", "mappo"): "benchmark_academy_counterattack_easy_mappo_eureka_llm",
    ("full_game_11v11_hard", "ippo"): "full_game_11_vs_11_hard_ippo_eureka_llm",
    ("full_game_11v11_hard", "mappo"): "full_game_11_vs_11_hard_mappo_eureka_llm",
    ("full_game_11v11_hard", "mat"): "full_game_11_vs_11_hard_mat_eureka_llm",
}


@dataclass
class Curve:
    scenario: str
    algo: str
    variant: str
    source: str
    epochs: np.ndarray
    goal_diff: np.ndarray

    @property
    def peak_goal_diff(self) -> float:
        return float(np.max(self.goal_diff))

    @property
    def max_epoch(self) -> int:
        return int(np.max(self.epochs))


def lighten(color: str, amount: float) -> Tuple[float, float, float]:
    base = np.array(mcolors.to_rgb(color), dtype=np.float32)
    return tuple(base + (1.0 - base) * amount)


def variant_color(algo: str, variant: str) -> Tuple[float, float, float]:
    if variant == "baseline":
        return mcolors.to_rgb(ALGO_COLOR[algo])
    if variant == "action_mask":
        return lighten(ALGO_COLOR[algo], 0.35)
    if variant == "reward_llm":
        return lighten(ALGO_COLOR[algo], 0.65)
    return mcolors.to_rgb(ALGO_COLOR[algo])


def latest_method_manifest() -> Path:
    manifests = sorted(RESULTS_ROOT.glob("method_comparison_*/manifest.json"))
    if not manifests:
        raise FileNotFoundError("No method comparison manifest found under results/")
    return manifests[-1]


def load_scalar_series(run_dir: Path, metric: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None

    ea = EventAccumulator(
        str(run_dir),
        size_guidance={
            "compressedHistograms": 0,
            "images": 0,
            "audio": 0,
            "scalars": 0,
            "histograms": 0,
            "tensors": 0,
        },
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    wanted = [
        tag for tag in tags if tag.startswith("RolloutEval/") and tag.endswith(f"/{metric}")
    ]
    if not wanted:
        return None

    events = ea.Scalars(wanted[0])
    if not events:
        return None

    step_to_value: Dict[int, float] = {}
    for event in events:
        step_to_value[int(event.step)] = float(event.value)

    steps = np.asarray(sorted(step_to_value.keys()), dtype=np.int32)
    values = np.asarray([step_to_value[step] for step in steps], dtype=np.float32)
    epochs = steps.astype(np.float32) / 2000.0
    return epochs, values


def load_local_result_series(
    experiment_name: str, run_name: str, metric: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    csv_path = (
        RESULTS_ROOT
        / experiment_name
        / "raw"
        / f"RolloutEval__agent_0__agent_0-default-1__{metric}.csv"
    )
    if not csv_path.exists():
        return None

    step_to_value: Dict[int, float] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("run") != run_name:
                continue
            try:
                step = int(float(row["step"]))
                value = float(row["value"])
            except (KeyError, ValueError):
                continue
            step_to_value[step] = value

    if not step_to_value:
        return None

    steps = np.asarray(sorted(step_to_value.keys()), dtype=np.int32)
    values = np.asarray([step_to_value[step] for step in steps], dtype=np.float32)
    epochs = steps.astype(np.float32) / 2000.0
    return epochs, values


def choose_best_reward_run(scenario: str, algo: str) -> Optional[Path]:
    expr_name = REWARD_EXPERIMENTS.get((scenario, algo))
    if expr_name is None:
        return None

    expr_dir = EXTERNAL_LOG_ROOT / expr_name
    if not expr_dir.exists():
        return None

    run_dirs = sorted(d for d in expr_dir.iterdir() if d.is_dir())
    if len(run_dirs) == 1:
        return run_dirs[0]

    best: Optional[Tuple[float, int, Path]] = None
    for run_dir in run_dirs:
        win_series = load_scalar_series(run_dir, "win")
        if win_series is None:
            continue
        epochs, values = win_series
        candidate = (float(np.max(values)), int(np.max(epochs)), run_dir)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    return best[2] if best else None


def curve_from_run_dir(
    scenario: str, algo: str, variant: str, run_dir: Path
) -> Optional[Curve]:
    series = load_scalar_series(run_dir, "goal_diff")
    if series is None:
        return None
    epochs, values = series
    return Curve(
        scenario=scenario,
        algo=algo,
        variant=variant,
        source=str(run_dir),
        epochs=epochs,
        goal_diff=values,
    )


def curve_from_local_results(
    scenario: str, algo: str, variant: str, run_dir: Path
) -> Optional[Curve]:
    experiment_name = run_dir.parent.name
    run_name = run_dir.name
    series = load_local_result_series(experiment_name, run_name, "goal_diff")
    if series is None:
        return curve_from_run_dir(scenario, algo, variant, run_dir)
    epochs, values = series
    return Curve(
        scenario=scenario,
        algo=algo,
        variant=variant,
        source=str(run_dir),
        epochs=epochs,
        goal_diff=values,
    )


def load_curves() -> List[Curve]:
    manifest = json.loads(latest_method_manifest().read_text())
    curves: List[Curve] = []

    for scenario_entry in manifest["scenarios"]:
        scenario = scenario_entry["scenario"]
        for curve_info in scenario_entry["curves"]:
            algo = curve_info["algo"]
            variant = curve_info["variant"]
            if variant not in ("baseline", "action_mask", "reward_llm"):
                continue

            if variant == "reward_llm":
                reward_run = choose_best_reward_run(scenario, algo)
                if reward_run is None:
                    continue
                curve = curve_from_run_dir(scenario, algo, variant, reward_run)
            else:
                run_dir = Path(curve_info["source"])
                curve = curve_from_local_results(scenario, algo, variant, run_dir)

            if curve is not None:
                curves.append(curve)

    return curves


def plot_goal_diff_lines(curves: List[Curve], output_dir: Path) -> Path:
    scenarios = list(SCENARIO_TITLE.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, scenario in zip(axes, scenarios):
        scenario_curves = [c for c in curves if c.scenario == scenario]
        for curve in scenario_curves:
            ax.plot(
                curve.epochs,
                curve.goal_diff,
                linewidth=2.0,
                color=variant_color(curve.algo, curve.variant),
                linestyle={
                    "baseline": "-",
                    "action_mask": "--",
                    "reward_llm": ":",
                }[curve.variant],
                label=f"{curve.algo.upper()} {VARIANT_LABEL[curve.variant]}",
            )
        ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
        ax.set_title(SCENARIO_TITLE[scenario], fontsize=13, fontweight="bold")
        ax.set_xlabel("Approx. Training Epoch")
        ax.set_ylabel("Avg Goal Difference")
        ax.grid(True, alpha=0.25)
        if scenario_curves:
            ax.legend(fontsize=8, ncol=1, loc="best")

    fig.suptitle("Average Goal Difference Over Training", fontsize=16, fontweight="bold")
    fig.tight_layout()
    output_path = output_dir / "avg_goal_diff_over_training.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_goal_diff_bars(curves: List[Curve], output_dir: Path) -> Path:
    scenarios = list(SCENARIO_TITLE.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    axes = axes.ravel()
    algo_order = ("ippo", "mappo", "mat")
    variant_order = ("baseline", "action_mask", "reward_llm")
    offsets = {
        "baseline": -0.24,
        "action_mask": 0.0,
        "reward_llm": 0.24,
    }
    width = 0.22

    by_key = {(c.scenario, c.algo, c.variant): c for c in curves}

    for ax, scenario in zip(axes, scenarios):
        x = np.arange(len(algo_order), dtype=np.float32)
        for algo_idx, algo in enumerate(algo_order):
            for variant in variant_order:
                curve = by_key.get((scenario, algo, variant))
                if curve is None:
                    continue
                xpos = x[algo_idx] + offsets[variant]
                ax.bar(
                    xpos,
                    curve.peak_goal_diff,
                    width=width,
                    color=variant_color(algo, variant),
                    edgecolor="#222222",
                    linewidth=0.9,
                    hatch=VARIANT_HATCH[variant],
                )

        ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([algo.upper() for algo in algo_order])
        ax.set_ylabel("Peak Avg Goal Difference")
        ax.set_title(SCENARIO_TITLE[scenario], fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)

    legend_handles = [
        Patch(
            facecolor="#d9d9d9",
            edgecolor="#222222",
            hatch=VARIANT_HATCH[variant],
            label=VARIANT_LABEL[variant],
        )
        for variant in variant_order
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "GRF-Style Goal Difference Summary",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = output_dir / "avg_goal_diff_summary.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_manifest(curves: List[Curve], outputs: Dict[str, str], output_dir: Path) -> Path:
    payload = {
        "output_dir": str(output_dir),
        "notes": [
            "Baseline and action-masking runs come from the latest method comparison manifest.",
            "Reward-shaping runs are loaded from external TensorBoard logs to use true RolloutEval goal_diff.",
            "Reward shaping POC is intentionally excluded.",
            "Summary bars use peak RolloutEval average goal difference from each selected run.",
        ],
        "outputs": outputs,
        "curves": [
            {
                "scenario": curve.scenario,
                "algo": curve.algo,
                "variant": curve.variant,
                "source": curve.source,
                "peak_goal_diff": curve.peak_goal_diff,
                "max_epoch": curve.max_epoch,
            }
            for curve in curves
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def main() -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_ROOT / f"goal_diff_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    curves = load_curves()
    if not curves:
        raise SystemExit("No goal-difference curves found.")

    outputs = {
        "line_plot": str(plot_goal_diff_lines(curves, output_dir)),
        "summary_plot": str(plot_goal_diff_bars(curves, output_dir)),
    }
    manifest_path = write_manifest(curves, outputs, output_dir)
    print(output_dir)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
