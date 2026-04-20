#!/usr/bin/env python3
"""
Generate comparison plots for all experiments under exps/gr_football.

Produces one figure per scenario:
  - Left panel : baseline algorithm comparison (a2po / happo / ippo / mappo / mat)
  - Right panel: LLM-mask vs baseline for algorithms that have both variants

Also produces a summary bar chart of peak win rates.

Output goes to results/plots/ (created if absent).
"""

import os
import re
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ────────────────────────────────────────────────────────────────────
ROOT  = pathlib.Path(__file__).resolve().parent.parent
EXPS  = ROOT / "exps" / "gr_football"
OUT   = ROOT / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# ── colour / style maps ───────────────────────────────────────────────────────
# Baseline: saturated colour.  LLM mask: lightened version of the same hue.
ALGO_COLOR = {
    "a2po":  "#e41a1c",
    "happo": "#ff7f00",
    "ippo":  "#4daf4a",
    "mappo": "#377eb8",
    "mat":   "#984ea3",
}

def _lighten(hex_color, amount=0.55):
    """Mix hex_color with white by `amount` (0=original, 1=white)."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return (r, g, b)

def variant_color(algo, variant):
    base = ALGO_COLOR[algo]
    return base if variant == "baseline" else _lighten(base, 0.55)

VARIANT_STYLE = {
    "baseline": dict(linestyle="-", linewidth=2.2),
    "llm_mask": dict(linestyle="-", linewidth=2.2),
}

# ── experiment name parser ────────────────────────────────────────────────────
# Returns (scenario_key, algo, variant) or None
_PATTERNS = [
    # benchmark_academy_<scenario>_<algo>
    (r"^benchmark_academy_(?P<scenario>3_vs_1_with_keeper|corner|counterattack_easy|counterattack)_(?P<algo>a2po|happo|ippo|mappo|mat)$",
     "baseline"),
    # llm_mask_academy_<scenario>_<algo>
    (r"^llm_mask_academy_(?P<scenario>corner|counterattack_easy|counterattack)_(?P<algo>ippo|mappo|mat)$",
     "llm_mask"),
    # full game (treat as single scenario)
    (r"^benchmark_full_game_11_v_11_hard_(?P<algo>mat)$",
     "baseline"),
    (r"^full_game_10_vs_10_hard_(?P<algo>ippo)$",
     "baseline"),
    (r"^full_game_11_v_11_hard_(?P<algo>mappo)$",
     "baseline"),
    (r"^llm_mask_full_game_11_vs_11_hard_(?P<algo>mappo)$",
     "llm_mask"),
]

def parse_exp(name):
    for pattern, variant in _PATTERNS:
        m = re.match(pattern, name)
        if m:
            d = m.groupdict()
            scenario = d.get("scenario", "full_game_11v11_hard")
            # normalise full-game entries to one scenario key
            if "full_game" in name or "11_v_11" in name or "10_vs_10" in name:
                scenario = "full_game_11v11_hard"
            return scenario, d["algo"], variant
    return None

# ── load data ─────────────────────────────────────────────────────────────────
def load_run(exp_dir):
    """Return DataFrame of epoch checkpoints sorted by epoch, or None."""
    run_dirs = [p for p in exp_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        return None
    # should be exactly one symlink per exps entry
    run_dir = run_dirs[0]
    csv = run_dir / "eval_results.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df = df[df["epoch"] >= 0].sort_values("epoch").reset_index(drop=True)
    return df if not df.empty else None

# ── collect all experiments ───────────────────────────────────────────────────
experiments = {}  # (scenario, algo, variant) -> df

for exp_path in sorted(EXPS.iterdir()):
    if not exp_path.is_dir():
        continue
    parsed = parse_exp(exp_path.name)
    if parsed is None:
        print(f"[skip] {exp_path.name}")
        continue
    scenario, algo, variant = parsed
    df = load_run(exp_path)
    if df is None:
        print(f"[no data] {exp_path.name}")
        continue
    key = (scenario, algo, variant)
    if key in experiments:
        # keep the one with more epochs
        if df["epoch"].max() > experiments[key]["epoch"].max():
            experiments[key] = df
    else:
        experiments[key] = df
    print(f"[loaded] {exp_path.name}  -> scenario={scenario} algo={algo} variant={variant}  epochs={df['epoch'].max()}")

# ── plotting helpers ──────────────────────────────────────────────────────────
def plot_curve(ax, df, label, algo, variant, metric="win_rate", ci_col="win_ci", show_ci=True):
    color = variant_color(algo, variant)
    style = VARIANT_STYLE[variant]
    epochs = df["epoch"].values
    values = df[metric].values
    line, = ax.plot(epochs, values, color=color, label=label, marker="o",
                    markersize=3, **style)
    if show_ci and ci_col in df.columns:
        ci = df[ci_col].values
        ax.fill_between(epochs, values - ci, values + ci, alpha=0.12, color=color)
    return line

def finish_ax(ax, title, ylabel="Win Rate", ylim=(0, 1.05)):
    ax.set_xlabel("Training Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

# ── per-scenario figures ──────────────────────────────────────────────────────
SCENARIO_TITLES = {
    "3_vs_1_with_keeper": "3v1 with Keeper",
    "corner":             "Corner",
    "counterattack":      "Counterattack",
    "counterattack_easy": "Counterattack Easy",
    "full_game_11v11_hard": "Full Game 11v11 Hard",
}

ALGO_ORDER = ["a2po", "happo", "ippo", "mappo", "mat"]

all_scenarios = sorted({sc for sc, _, _ in experiments})

for scenario in all_scenarios:
    title = SCENARIO_TITLES.get(scenario, scenario)

    # Separate baseline and llm_mask entries for this scenario
    baseline_algos = {algo: experiments[(scenario, algo, "baseline")]
                      for algo in ALGO_ORDER
                      if (scenario, algo, "baseline") in experiments}
    mask_algos     = {algo: experiments[(scenario, algo, "llm_mask")]
                      for algo in ALGO_ORDER
                      if (scenario, algo, "llm_mask") in experiments}

    has_mask = bool(mask_algos)
    ncols = 2 if has_mask else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    # ── left panel: baseline algorithms ──────────────────────────────────────
    ax = axes[0]
    for algo in ALGO_ORDER:
        if algo not in baseline_algos:
            continue
        plot_curve(ax, baseline_algos[algo],
                   label=algo.upper(), algo=algo, variant="baseline")
    finish_ax(ax, "Baseline: Algorithm Comparison")

    # ── right panel: baseline vs LLM-mask (shared algos only) ───────────────
    if has_mask:
        ax = axes[1]
        shared = [a for a in ALGO_ORDER if a in baseline_algos and a in mask_algos]
        for algo in shared:
            plot_curve(ax, baseline_algos[algo],
                       label=f"{algo.upper()} baseline",
                       algo=algo, variant="baseline")
            plot_curve(ax, mask_algos[algo],
                       label=f"{algo.upper()} + LLM mask",
                       algo=algo, variant="llm_mask")
        finish_ax(ax, "Baseline vs LLM-Guided Mask")

    fig.tight_layout()
    out_path = OUT / f"{scenario}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

# ── summary bar chart: peak win rate per experiment ───────────────────────────
records = []
for (scenario, algo, variant), df in experiments.items():
    peak = df["win_rate"].max()
    records.append(dict(scenario=scenario, algo=algo, variant=variant, peak=peak))

summary = pd.DataFrame(records).sort_values(["scenario", "variant", "algo"])

scenarios = sorted(summary["scenario"].unique())
variants  = ["baseline", "llm_mask"]
n_sc = len(scenarios)

fig, axes = plt.subplots(1, n_sc, figsize=(4.5 * n_sc, 5), sharey=True)
if n_sc == 1:
    axes = [axes]

for ax, scenario in zip(axes, scenarios):
    sub = summary[summary["scenario"] == scenario]
    x_labels, heights, colors, hatches = [], [], [], []
    for variant in variants:
        v = sub[sub["variant"] == variant].sort_values("algo")
        for _, row in v.iterrows():
            lbl = f"{row['algo'].upper()}\n({'LLM' if variant == 'llm_mask' else 'base'})"
            x_labels.append(lbl)
            heights.append(row["peak"])
            colors.append(variant_color(row["algo"], variant))
            hatches.append("")

    xs = np.arange(len(x_labels))
    bars = ax.bar(xs, heights, color=colors, hatch=hatches, edgecolor="white", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_title(SCENARIO_TITLES.get(scenario, scenario), fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, axis="y", alpha=0.3)
    # annotate bar values
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7)

axes[0].set_ylabel("Peak Win Rate", fontsize=11)

# legend patches
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="grey",            label="Baseline (saturated)"),
    Patch(facecolor=_lighten("grey"), label="LLM Mask (light)"),
]
fig.legend(handles=legend_handles, loc="upper right", fontsize=9)
fig.suptitle("Peak Win Rate by Experiment", fontsize=13, fontweight="bold")
fig.tight_layout()
out_path = OUT / "summary_peak_winrate.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out_path}")
print("\nDone. All plots written to", OUT)
