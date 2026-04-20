#!/usr/bin/env python3
"""
Generate per-scenario win-rate plots comparing baseline, action masking, and
reward-shaping variants for IPPO / MAPPO / MAT.

Data sources
------------
- Baselines: logs/gr_football/<benchmark/full_game experiment>/*
  Only experiments whose directory name does NOT contain "warmup" are scanned.
  Symlinked warmup-selected runs inside those base directories are allowed.
- Action masking: logs/gr_football/llm_mask_*/*
- Reward shaping: external eval CSVs under /home/zhang.kaiqi1/GRF_MARL/results/eval_*

Output
------
Creates a fresh timestamped folder under results/ and writes:
- one PNG per scenario
- manifest.json describing selected runs and epoch ranges
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
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
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = ROOT / "logs" / "gr_football"
EXTERNAL_EVAL_ROOT = Path("/home/zhang.kaiqi1/GRF_MARL/results")
OUTPUT_ROOT = ROOT / "results"

ALGORITHMS = ("ippo", "mappo", "mat")
SCENARIO_ORDER = (
    "corner",
    "counterattack",
    "counterattack_easy",
    "full_game_11v11_hard",
)

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

VARIANT_STYLE = {
    "baseline": {"linestyle": "-", "marker": "o"},
    "action_mask": {"linestyle": "--", "marker": "s"},
    "reward_llm": {"linestyle": ":", "marker": "^"},
}

VARIANT_LABEL = {
    "baseline": "Baseline",
    "action_mask": "Action Masking",
    "reward_llm": "Reward Shaping (LLM)",
}

SUMMARY_VARIANT_COLOR = {
    "action_mask": "#ef6c00",
    "reward_llm": "#546e7a",
}

BASELINE_EXPERIMENTS = {
    ("corner", "ippo"): "benchmark_academy_corner_ippo",
    ("corner", "mappo"): "benchmark_academy_corner_mappo",
    ("corner", "mat"): "benchmark_academy_corner_mat",
    ("counterattack", "ippo"): "benchmark_academy_counterattack_ippo",
    ("counterattack", "mappo"): "benchmark_academy_counterattack_mappo",
    ("counterattack", "mat"): "benchmark_academy_counterattack_mat",
    ("counterattack_easy", "ippo"): "benchmark_academy_counterattack_easy_ippo",
    ("counterattack_easy", "mappo"): "benchmark_academy_counterattack_easy_mappo",
    ("counterattack_easy", "mat"): "benchmark_academy_counterattack_easy_mat",
    ("full_game_11v11_hard", "ippo"): "full_game_10_vs_10_hard_ippo",
    ("full_game_11v11_hard", "mappo"): "full_game_11_v_11_hard_mappo",
    ("full_game_11v11_hard", "mat"): "benchmark_full_game_11_v_11_hard_mat",
}

ACTION_MASK_EXPERIMENTS = {
    ("corner", "ippo"): "llm_mask_academy_corner_ippo",
    ("corner", "mappo"): "llm_mask_academy_corner_mappo",
    ("corner", "mat"): "llm_mask_academy_corner_mat",
    ("counterattack", "ippo"): "llm_mask_academy_counterattack_ippo",
    ("counterattack", "mappo"): "llm_mask_academy_counterattack_mappo",
    ("counterattack", "mat"): "llm_mask_academy_counterattack_mat",
    ("counterattack_easy", "ippo"): "llm_mask_academy_counterattack_easy_ippo",
    ("counterattack_easy", "mappo"): "llm_mask_academy_counterattack_easy_mappo",
    ("counterattack_easy", "mat"): "llm_mask_academy_counterattack_easy_mat",
    ("full_game_11v11_hard", "mappo"): "llm_mask_full_game_11_vs_11_hard_mappo",
}

EXTERNAL_PATTERNS = [
    (
        re.compile(
            r"^eval_(corner|counterattack|counterattack_easy)_(ippo|mappo|mat)_eureka_llm$"
        ),
        "reward_llm",
    ),
    (
        re.compile(
            r"^eval_(corner|counterattack|counterattack_easy)_(ippo|mappo|mat)_llm$"
        ),
        "reward_llm",
    ),
    (
        re.compile(
            r"^eval_(corner|counterattack|counterattack_easy)_(ippo|mappo|mat)_poc$"
        ),
        "reward_poc",
    ),
    (
        re.compile(
            r"^eval_full_game_11_vs_11_hard_(ippo|mappo|mat)_eureka_llm$"
        ),
        "reward_llm",
    ),
    (
        re.compile(
            r"^eval_full_game_11_vs_11_hard_(ippo|mappo|mat)_eureka_poc$"
        ),
        "reward_poc",
    ),
]


@dataclass
class Curve:
    scenario: str
    algo: str
    variant: str
    epochs: np.ndarray
    win_rate: np.ndarray
    win_ci: np.ndarray
    source: str
    peak_win_rate: float
    max_epoch: int


def lighten(color: str, amount: float) -> Tuple[float, float, float]:
    base = np.array(mcolors.to_rgb(color), dtype=np.float32)
    return tuple(base + (1.0 - base) * amount)


def variant_color(algo: str, variant: str) -> Tuple[float, float, float]:
    if variant == "baseline":
        return mcolors.to_rgb(ALGO_COLOR[algo])
    if variant == "action_mask":
        return lighten(ALGO_COLOR[algo], 0.35)
    if variant == "reward_llm":
        return lighten(ALGO_COLOR[algo], 0.6)
    return mcolors.to_rgb(ALGO_COLOR[algo])


def read_eval_csv(csv_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    epochs: List[int] = []
    win_rate: List[float] = []
    win_ci: List[float] = []
    if not csv_path.exists():
        return None

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(float(row["epoch"]))
            except (KeyError, ValueError):
                continue
            if epoch < 0:
                continue
            try:
                win = float(row["win_rate"])
            except (KeyError, ValueError):
                continue
            try:
                ci = float(row.get("win_ci", 0.0) or 0.0)
            except ValueError:
                ci = 0.0
            epochs.append(epoch)
            win_rate.append(win)
            win_ci.append(ci)

    if not epochs:
        return None

    order = np.argsort(np.asarray(epochs, dtype=np.int32))
    ep = np.asarray(epochs, dtype=np.int32)[order]
    wr = np.asarray(win_rate, dtype=np.float32)[order]
    ci = np.asarray(win_ci, dtype=np.float32)[order]

    unique_epochs: List[int] = []
    unique_win: List[float] = []
    unique_ci: List[float] = []
    for epoch in np.unique(ep):
        idx = np.where(ep == epoch)[0][-1]
        unique_epochs.append(int(ep[idx]))
        unique_win.append(float(wr[idx]))
        unique_ci.append(float(ci[idx]))

    return (
        np.asarray(unique_epochs, dtype=np.int32),
        np.asarray(unique_win, dtype=np.float32),
        np.asarray(unique_ci, dtype=np.float32),
    )


def curve_from_csv(
    scenario: str, algo: str, variant: str, csv_path: Path, source: str
) -> Optional[Curve]:
    loaded = read_eval_csv(csv_path)
    if loaded is None:
        return None
    epochs, win_rate, win_ci = loaded
    return Curve(
        scenario=scenario,
        algo=algo,
        variant=variant,
        epochs=epochs,
        win_rate=win_rate,
        win_ci=win_ci,
        source=source,
        peak_win_rate=float(np.max(win_rate)),
        max_epoch=int(np.max(epochs)),
    )


def choose_best_run(
    scenario: str, algo: str, variant: str, experiment_dir: Path
) -> Optional[Curve]:
    if not experiment_dir.exists():
        return None

    best: Optional[Curve] = None
    for run_dir in sorted(experiment_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        curve = curve_from_csv(
            scenario=scenario,
            algo=algo,
            variant=variant,
            csv_path=run_dir / "eval_results.csv",
            source=str(run_dir),
        )
        if curve is None:
            continue
        if best is None:
            best = curve
            continue
        best_key = (best.peak_win_rate, best.max_epoch)
        curve_key = (curve.peak_win_rate, curve.max_epoch)
        if curve_key > best_key:
            best = curve
    return best


def load_baselines() -> Dict[Tuple[str, str, str], Curve]:
    selected: Dict[Tuple[str, str, str], Curve] = {}
    for (scenario, algo), exp_name in BASELINE_EXPERIMENTS.items():
        if "warmup" in exp_name:
            continue
        curve = choose_best_run(scenario, algo, "baseline", LOG_ROOT / exp_name)
        if curve is not None:
            selected[(scenario, algo, "baseline")] = curve
    return selected


def load_action_masks() -> Dict[Tuple[str, str, str], Curve]:
    selected: Dict[Tuple[str, str, str], Curve] = {}
    for (scenario, algo), exp_name in ACTION_MASK_EXPERIMENTS.items():
        curve = choose_best_run(scenario, algo, "action_mask", LOG_ROOT / exp_name)
        if curve is not None:
            selected[(scenario, algo, "action_mask")] = curve
    return selected


def parse_external_dir(path: Path) -> Optional[Tuple[str, str, str]]:
    name = path.name
    for pattern, variant in EXTERNAL_PATTERNS:
        match = pattern.match(name)
        if not match:
            continue
        groups = match.groups()
        if name.startswith("eval_full_game_11_vs_11_hard_"):
            scenario = "full_game_11v11_hard"
            algo = groups[0]
        else:
            scenario, algo = groups
        return scenario, algo, variant
    return None


def load_external_reward_shaping() -> Dict[Tuple[str, str, str], Curve]:
    selected: Dict[Tuple[str, str, str], Curve] = {}
    if not EXTERNAL_EVAL_ROOT.exists():
        return selected
    for path in sorted(EXTERNAL_EVAL_ROOT.glob("eval_*")):
        parsed = parse_external_dir(path)
        if parsed is None:
            continue
        scenario, algo, variant = parsed
        if variant != "reward_llm":
            continue
        curve = curve_from_csv(
            scenario=scenario,
            algo=algo,
            variant=variant,
            csv_path=path / "eval_results.csv",
            source=str(path),
        )
        if curve is None:
            continue
        selected[(scenario, algo, variant)] = curve
    return selected


def shared_epoch_grid(curves: List[Curve], interval: int = 100) -> np.ndarray:
    max_epoch = min(curve.max_epoch for curve in curves)
    if max_epoch < interval:
        return np.asarray([], dtype=np.int32)
    return np.arange(interval, max_epoch + 1, interval, dtype=np.int32)


def resample_curve(curve: Curve, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    win = np.interp(grid, curve.epochs, curve.win_rate)
    ci = np.interp(grid, curve.epochs, curve.win_ci)
    return win, ci


def plot_scenario(
    scenario: str,
    curves: List[Curve],
    output_dir: Path,
) -> Dict[str, object]:
    grid = shared_epoch_grid(curves, interval=100)
    if grid.size == 0:
        return {
            "scenario": scenario,
            "title": SCENARIO_TITLE[scenario],
            "shared_max_epoch": 0,
            "curves": [],
            "output": None,
        }

    fig, ax = plt.subplots(figsize=(11, 6))

    manifest_curves = []
    for algo in ALGORITHMS:
        algo_curves = [c for c in curves if c.algo == algo]
        for curve in sorted(
            algo_curves,
            key=lambda c: ("baseline", "action_mask", "reward_llm").index(
                c.variant
            ),
        ):
            values, ci = resample_curve(curve, grid)
            style = VARIANT_STYLE[curve.variant]
            color = variant_color(curve.algo, curve.variant)
            label = f"{curve.algo.upper()} {VARIANT_LABEL[curve.variant]}"
            ax.plot(
                grid,
                values,
                label=label,
                color=color,
                linewidth=2.2,
                markersize=4,
                markevery=max(1, len(grid) // 10),
                **style,
            )
            ax.fill_between(grid, values - ci, values + ci, color=color, alpha=0.10)
            manifest_curves.append(
                {
                    "algo": curve.algo,
                    "variant": curve.variant,
                    "label": VARIANT_LABEL[curve.variant],
                    "source": curve.source,
                    "peak_win_rate": round(curve.peak_win_rate, 4),
                    "max_epoch": curve.max_epoch,
                }
            )

    ax.set_title(SCENARIO_TITLE[scenario], fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, alpha=0.28)

    algo_handles = [
        Line2D([0], [0], color=ALGO_COLOR[algo], linewidth=3, label=algo.upper())
        for algo in ALGORITHMS
    ]
    variant_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.2,
            label=VARIANT_LABEL[variant],
            **VARIANT_STYLE[variant],
        )
        for variant in ("baseline", "action_mask", "reward_llm")
        if any(c.variant == variant for c in curves)
    ]

    first_legend = ax.legend(handles=algo_handles, title="Algorithm", loc="upper left")
    ax.add_artist(first_legend)
    ax.legend(handles=variant_handles, title="Variant", loc="lower right")

    fig.tight_layout()
    output_path = output_dir / f"{scenario}.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "scenario": scenario,
        "title": SCENARIO_TITLE[scenario],
        "shared_max_epoch": int(grid[-1]),
        "curves": manifest_curves,
        "output": str(output_path),
    }


def build_peak_lookup(selected: Dict[Tuple[str, str, str], Curve]) -> Dict[Tuple[str, str, str], float]:
    return {
        key: curve.peak_win_rate * 100.0
        for key, curve in selected.items()
        if key[2] in ("baseline", "action_mask", "reward_llm")
    }


def plot_summary_heatmap(selected: Dict[Tuple[str, str, str], Curve], output_dir: Path) -> str:
    variants = ("baseline", "action_mask", "reward_llm")
    rows = []
    row_labels = []
    peak = build_peak_lookup(selected)
    for scenario in SCENARIO_ORDER:
        for algo in ALGORITHMS:
            vals = []
            has_any = False
            for variant in variants:
                key = (scenario, algo, variant)
                if key in peak:
                    vals.append(peak[key])
                    has_any = True
                else:
                    vals.append(np.nan)
            if has_any:
                rows.append(vals)
                row_labels.append(f"{SCENARIO_TITLE[scenario]} | {algo.upper()}")

    data = np.asarray(rows, dtype=np.float32)
    masked = np.ma.masked_invalid(data)

    fig_h = max(4.5, 0.42 * len(row_labels))
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels([VARIANT_LABEL[v] for v in variants], rotation=0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Peak Win Rate Heatmap (%)", fontsize=14, fontweight="bold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Peak Win Rate (%)")
    fig.tight_layout()
    output_path = output_dir / "summary_heatmap.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_delta_vs_baseline(selected: Dict[Tuple[str, str, str], Curve], output_dir: Path) -> str:
    variants = ("action_mask", "reward_llm")
    fig, axes = plt.subplots(1, len(SCENARIO_ORDER), figsize=(16, 5), sharey=True)
    peak = build_peak_lookup(selected)

    for ax, scenario in zip(axes, SCENARIO_ORDER):
        x = np.arange(len(ALGORITHMS), dtype=np.float32)
        width = 0.34
        offsets = (-width / 2, width / 2)

        for idx, variant in enumerate(variants):
            deltas = []
            for algo in ALGORITHMS:
                base = peak.get((scenario, algo, "baseline"))
                other = peak.get((scenario, algo, variant))
                if base is None or other is None:
                    deltas.append(np.nan)
                else:
                    deltas.append(other - base)

            vals = np.asarray(deltas, dtype=np.float32)
            mask = ~np.isnan(vals)
            ax.bar(
                x[mask] + offsets[idx],
                vals[mask],
                width=width,
                color=SUMMARY_VARIANT_COLOR[variant],
                alpha=0.9,
                label=VARIANT_LABEL[variant],
            )

        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([algo.upper() for algo in ALGORITHMS])
        ax.set_title(SCENARIO_TITLE[scenario], fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Peak Win Rate Delta vs Baseline (pp)")
    axes[-1].legend(loc="upper left")
    fig.suptitle("Peak Win Rate Improvement vs Baseline", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path = output_dir / "summary_delta_vs_baseline.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def main() -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"method_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    selected: Dict[Tuple[str, str, str], Curve] = {}
    selected.update(load_baselines())
    selected.update(load_action_masks())
    selected.update(load_external_reward_shaping())

    scenario_manifests = []
    for scenario in SCENARIO_ORDER:
        curves = [
            curve
            for key, curve in selected.items()
            if key[0] == scenario
            and not (scenario == "full_game_11v11_hard" and key[2] == "action_mask")
        ]
        if not curves:
            continue
        scenario_manifests.append(plot_scenario(scenario, curves, output_dir))

    summary_heatmap = plot_summary_heatmap(selected, output_dir)
    summary_delta = plot_delta_vs_baseline(selected, output_dir)

    manifest = {
        "output_dir": str(output_dir),
        "notes": [
            "Baselines were selected from experiment directories without 'warmup' in the directory name.",
            "Run selection within each experiment keeps the run with highest peak win rate, breaking ties by longer epoch range.",
            "Each scenario plot is trimmed to the shared epoch range across all curves shown.",
            "11v11 action-masking curves are intentionally omitted.",
            "The 11v11 IPPO baseline source is the repository's existing full-game IPPO run directory named 'full_game_10_vs_10_hard_ippo'; its config points to benchmark_full_game_11_vs_11_hard.",
        ],
        "summary_outputs": {
            "heatmap": summary_heatmap,
            "delta_vs_baseline": summary_delta,
        },
        "scenarios": scenario_manifests,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
