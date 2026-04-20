#!/usr/bin/env python3
"""
Overlay six eval curves: IPPO / MAPPO / MAT × (baseline vs LLM-shaped reward).

Same algorithm shares one color: baseline dashed, LLM solid (clear pairwise link).

CSV format: same as eval_checkpoint.py → eval_results.csv (epoch, win_rate, win_ci, ...).

Example:
  python scripts/plot_algo_llm_sixway.py \\
    --title "Academy corner — win rate" \\
    --output results/corner_sixway.png \\
    --ippo-baseline  logs/.../ippo/eval_results.csv \\
    --ippo-llm       logs/.../ippo_llm/eval_results.csv \\
    --mappo-baseline logs/.../mappo/eval_results.csv \\
    --mappo-llm      logs/.../mappo_llm/eval_results.csv \\
    --mat-baseline   logs/.../mat/eval_results.csv \\
    --mat-llm        logs/.../mat_llm/eval_results.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load read_eval_csv + truncate_series from compare_eval.py (same directory)
_scripts_dir = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("compare_eval", _scripts_dir / "compare_eval.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("Cannot load compare_eval.py")
_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compare)
read_eval_csv = _compare.read_eval_csv
truncate_series = _compare.truncate_series

# Tab10-like, distinct for three algorithms
ALGO_COLORS = {
    "IPPO": "#1f77b4",
    "MAPPO": "#ff7f0e",
    "MAT": "#2ca02c",
}


def _plot_one_algo(
    ax,
    algo: str,
    color: str,
    rows_base,
    rows_llm,
    metric: str,
    ci_key: str,
    show_ci: bool,
    markersize: float,
):
    def draw(rows, linestyle, label_suffix, marker):
        if not rows:
            return
        epochs = [r["epoch"] for r in rows]
        vals = [r[metric] for r in rows]
        cis = [r.get(ci_key, 0.0) for r in rows]
        kw = dict(
            color=color,
            linestyle=linestyle,
            linewidth=2.4 if linestyle == "-" else 1.9,
            marker=marker,
            markersize=markersize,
            label=f"{algo} ({label_suffix})",
        )
        eb_kw = {
            "color": color,
            "linestyle": linestyle,
            "linewidth": kw["linewidth"],
            "marker": marker,
            "markersize": markersize,
            "label": kw["label"],
        }
        if show_ci and any(c > 0 for c in cis):
            ax.errorbar(epochs, vals, yerr=cis, capsize=2.5, **eb_kw)
        else:
            ax.plot(epochs, vals, **eb_kw)

    draw(rows_base, "--", "baseline", "o")
    draw(rows_llm, "-", "LLM φ", "s")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Six-way plot: IPPO/MAPPO/MAT baseline (dashed) vs LLM shaping (solid), same color per algo."
    )
    ap.add_argument("--title", type=str, default="Eval comparison")
    ap.add_argument("--output", "-o", type=str, required=True, help="Output .png path")
    ap.add_argument("--metric", choices=("win_rate", "reward_mean", "goal_mean"), default="win_rate")
    ap.add_argument("--max-epoch", type=float, default=None, help="Truncate all series at this epoch")
    ap.add_argument("--no-ci", action="store_true", help="Omit error bars")
    ap.add_argument("--figsize", type=str, default="10,6", help="W,H in inches, e.g. 11,6.5")

    ap.add_argument("--ippo-baseline", type=str, required=True)
    ap.add_argument("--ippo-llm", type=str, required=True)
    ap.add_argument("--mappo-baseline", type=str, required=True)
    ap.add_argument("--mappo-llm", type=str, required=True)
    ap.add_argument("--mat-baseline", type=str, required=True)
    ap.add_argument("--mat-llm", type=str, required=True)

    args = ap.parse_args()

    triples = [
        ("IPPO", args.ippo_baseline, args.ippo_llm),
        ("MAPPO", args.mappo_baseline, args.mappo_llm),
        ("MAT", args.mat_baseline, args.mat_llm),
    ]

    ci_map = {"win_rate": "win_ci", "reward_mean": "reward_ci", "goal_mean": "goal_ci"}
    ylabel_map = {"win_rate": "Win rate", "reward_mean": "Mean reward", "goal_mean": "Goals (mean)"}
    ci_key = ci_map[args.metric]

    w, h = (float(x) for x in args.figsize.split(","))
    fig, ax = plt.subplots(figsize=(w, h))

    for algo, path_b, path_l in triples:
        for p, tag in ((path_b, "baseline"), (path_l, "LLM")):
            if not os.path.isfile(p):
                print(f"[ERROR] Missing CSV ({algo} {tag}): {p}", file=sys.stderr)
                sys.exit(1)
        rb = truncate_series(read_eval_csv(path_b), args.max_epoch)
        rl = truncate_series(read_eval_csv(path_l), args.max_epoch)
        print(f"  {algo}: baseline {len(rb)} pts, LLM {len(rl)} pts")
        _plot_one_algo(
            ax,
            algo,
            ALGO_COLORS[algo],
            rb,
            rl,
            args.metric,
            ci_key,
            show_ci=not args.no_ci,
            markersize=4.0,
        )

    ax.set_xlabel("Training epoch", fontsize=12)
    ax.set_ylabel(ylabel_map[args.metric], fontsize=12)
    ax.set_title(args.title, fontsize=14)
    if args.metric == "win_rate":
        ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc="best", framealpha=0.92)

    out = args.output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
