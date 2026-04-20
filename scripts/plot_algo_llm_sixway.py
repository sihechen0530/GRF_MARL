#!/usr/bin/env python3
"""
Overlay six eval curves: IPPO / MAPPO / MAT × (baseline vs LLM-shaped reward).

Same algorithm shares one color: baseline dashed, LLM solid.
Legend: two columns — left = baselines (IPPO / MAPPO / MAT), right = LLM φ (same order).

CSV format: same as eval_checkpoint.py → eval_results.csv.

Example:
  python scripts/plot_algo_llm_sixway.py \\
    --title "Academy corner — win rate" \\
    --output results/corner_sixway.png \\
    --max-epoch 1100 \\
    --ippo-baseline  .../ippo/eval_results.csv \\
    --ippo-llm       .../ippo_llm/eval_results.csv \\
    ...
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_scripts_dir = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("compare_eval", _scripts_dir / "compare_eval.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("Cannot load compare_eval.py")
_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compare)
read_eval_csv = _compare.read_eval_csv
truncate_series = _compare.truncate_series

ALGO_COLORS = {
    "IPPO": "#1f77b4",
    "MAPPO": "#ff7f0e",
    "MAT": "#2ca02c",
}


def _plot_series(ax, rows, metric, ci_key, color, linestyle, marker, markersize, show_ci):
    """Draw one series; return artist for legend (Line2D or ErrorbarContainer)."""
    if not rows:
        return None
    epochs = [r["epoch"] for r in rows]
    vals = [r[metric] for r in rows]
    cis = [r.get(ci_key, 0.0) for r in rows]
    lw = 2.4 if linestyle == "-" else 1.9
    kw = dict(
        color=color,
        linestyle=linestyle,
        linewidth=lw,
        marker=marker,
        markersize=markersize,
    )
    if show_ci and any(float(c) > 0 for c in cis):
        return ax.errorbar(epochs, vals, yerr=cis, capsize=2.5, **kw)
    (ln,) = ax.plot(epochs, vals, **kw)
    return ln


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Six-way plot: IPPO/MAPPO/MAT baseline (dashed) vs LLM (solid); "
        "legend columns = baseline | LLM."
    )
    ap.add_argument("--title", type=str, default="Eval comparison")
    ap.add_argument("--output", "-o", type=str, required=True, help="Output .png path")
    ap.add_argument("--metric", choices=("win_rate", "reward_mean", "goal_mean"), default="win_rate")
    ap.add_argument(
        "--max-epoch",
        type=float,
        default=None,
        help="Truncate all CSV rows to epoch <= this value AND cap x-axis here (e.g. 1100 when LLM runs stop earlier).",
    )
    ap.add_argument("--no-ci", action="store_true", help="Omit error bars")
    ap.add_argument("--figsize", type=str, default="10,6", help="W,H in inches, e.g. 11,6.5")
    ap.add_argument(
        "--legend-loc",
        type=str,
        default="upper left",
        help="Matplotlib legend loc= (default: upper left).",
    )
    ap.add_argument(
        "--legend-bbox",
        type=str,
        default="0.01,0.99",
        help="bbox_to_anchor as x,y in axes fraction (default: 0.01,0.99).",
    )
    ap.add_argument(
        "--legend-alpha",
        type=float,
        default=0.92,
        help="Legend frame alpha (default: 0.92).",
    )

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

    # Interleave per row for legend ncol=2: (baseline_i, llm_i) so col0 = all baseline, col1 = all LLM
    legend_handles = []
    legend_labels = []
    ms = 4.0
    show_ci = not args.no_ci

    for algo, path_b, path_l in triples:
        for p, tag in ((path_b, "baseline"), (path_l, "LLM")):
            if not os.path.isfile(p):
                print(f"[ERROR] Missing CSV ({algo} {tag}): {p}", file=sys.stderr)
                sys.exit(1)
        rb = truncate_series(read_eval_csv(path_b), args.max_epoch)
        rl = truncate_series(read_eval_csv(path_l), args.max_epoch)
        print(f"  {algo}: baseline {len(rb)} pts, LLM {len(rl)} pts")

        color = ALGO_COLORS[algo]
        hb = _plot_series(ax, rb, args.metric, ci_key, color, "--", "o", ms, show_ci)
        hl = _plot_series(ax, rl, args.metric, ci_key, color, "-", "s", ms, show_ci)
        if hb is not None:
            legend_handles.append(hb)
            legend_labels.append(f"{algo} (baseline)")
        if hl is not None:
            legend_handles.append(hl)
            legend_labels.append(f"{algo} (LLM φ)")

    ax.set_xlabel("Training epoch", fontsize=12)
    ax.set_ylabel(ylabel_map[args.metric], fontsize=12)
    ax.set_title(args.title, fontsize=14)
    if args.metric == "win_rate":
        ax.set_ylim(-0.05, 1.05)
    if args.max_epoch is not None:
        ax.set_xlim(0, float(args.max_epoch))
    ax.grid(True, alpha=0.3)

    # ncol=2 + order [b0,l0,b1,l1,...] → column 0 = baselines, column 1 = LLM
    bx, by = (float(x.strip()) for x in args.legend_bbox.split(","))
    ax.legend(
        legend_handles,
        legend_labels,
        ncol=2,
        loc=args.legend_loc,
        bbox_to_anchor=(bx, by),
        borderaxespad=0.0,
        fontsize=9,
        framealpha=args.legend_alpha,
        facecolor="white",
        edgecolor="0.75",
        fancybox=True,
        columnspacing=1.2,
        handlelength=2.8,
    )

    out = args.output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
