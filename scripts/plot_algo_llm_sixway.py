#!/usr/bin/env python3
"""
Overlay eval curves for IPPO / MAPPO / MAT × (baseline vs LLM-shaped reward).

Each CSV path is optional — pass only the runs you want (e.g. four curves: two algos × baseline + LLM).
Same algorithm shares one color: baseline dashed, LLM solid.
Legend (ncol=2): rows pair (baseline | LLM φ) per algorithm in IPPO → MAPPO → MAT order.

CSV format: same as eval_checkpoint.py → eval_results.csv.

Example (four lines: IPPO + MAPPO only):
  python scripts/plot_algo_llm_sixway.py \\
    --output results/fourway.png \\
    --max-epoch 1100 \\
    --ippo-baseline .../ippo/eval_results.csv \\
    --ippo-llm .../ippo_llm/eval_results.csv \\
    --mappo-baseline .../mappo/eval_results.csv \\
    --mappo-llm .../mappo_llm/eval_results.csv
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


def _norm_path(p: str | None) -> str | None:
    if p is None:
        return None
    s = str(p).strip()
    return s if s else None


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
        description="Optional IPPO/MAPPO/MAT baseline + LLM CSVs; same color per algo, dashed=baseline, solid=LLM."
    )
    ap.add_argument("--title", type=str, default="Eval comparison")
    ap.add_argument("--output", "-o", type=str, required=True, help="Output .png path")
    ap.add_argument("--metric", choices=("win_rate", "reward_mean", "goal_mean"), default="win_rate")
    ap.add_argument(
        "--max-epoch",
        type=float,
        default=None,
        help="Truncate CSV rows to epoch <= this value; if set, also sets x-axis limit to [0, max-epoch].",
    )
    ap.add_argument(
        "--xmax-auto",
        action="store_true",
        help="If set (and --max-epoch not set), set x-axis upper limit to max epoch seen in plotted data.",
    )
    ap.add_argument("--no-ci", action="store_true", help="Omit error bars")
    ap.add_argument("--figsize", type=str, default="10,6", help="W,H in inches, e.g. 11,6.5")
    ap.add_argument("--legend-loc", type=str, default="upper left", help="Matplotlib legend loc=")
    ap.add_argument(
        "--legend-bbox",
        type=str,
        default="0.01,0.99",
        help="bbox_to_anchor as x,y in axes fraction (default: 0.01,0.99).",
    )
    ap.add_argument("--legend-alpha", type=float, default=0.92, help="Legend frame alpha")

    ap.add_argument("--ippo-baseline", type=str, default=None, help="Optional eval_results.csv")
    ap.add_argument("--ippo-llm", type=str, default=None)
    ap.add_argument("--mappo-baseline", type=str, default=None)
    ap.add_argument("--mappo-llm", type=str, default=None)
    ap.add_argument("--mat-baseline", type=str, default=None)
    ap.add_argument("--mat-llm", type=str, default=None)

    args = ap.parse_args()

    triples = [
        ("IPPO", _norm_path(args.ippo_baseline), _norm_path(args.ippo_llm)),
        ("MAPPO", _norm_path(args.mappo_baseline), _norm_path(args.mappo_llm)),
        ("MAT", _norm_path(args.mat_baseline), _norm_path(args.mat_llm)),
    ]

    ci_map = {"win_rate": "win_ci", "reward_mean": "reward_ci", "goal_mean": "goal_ci"}
    ylabel_map = {"win_rate": "Win rate", "reward_mean": "Mean reward", "goal_mean": "Goals (mean)"}
    ci_key = ci_map[args.metric]

    w, h = (float(x) for x in args.figsize.split(","))
    fig, ax = plt.subplots(figsize=(w, h))

    legend_handles = []
    legend_labels = []
    ms = 4.0
    show_ci = not args.no_ci
    max_epoch_seen = 0.0
    n_curves = 0

    for algo, path_b, path_l in triples:
        color = ALGO_COLORS[algo]

        if path_b is not None:
            if not os.path.isfile(path_b):
                print(f"[ERROR] File not found ({algo} baseline): {path_b}", file=sys.stderr)
                sys.exit(1)
            rb = truncate_series(read_eval_csv(path_b), args.max_epoch)
            if rb:
                max_epoch_seen = max(max_epoch_seen, max(r["epoch"] for r in rb))
            hb = _plot_series(ax, rb, args.metric, ci_key, color, "--", "o", ms, show_ci)
            if hb is not None:
                legend_handles.append(hb)
                legend_labels.append(f"{algo} (baseline)")
                n_curves += 1
                print(f"  {algo} baseline: {len(rb)} pts from {path_b}")
            else:
                print(f"  [WARN] {algo} baseline CSV empty after truncate: {path_b}")

        if path_l is not None:
            if not os.path.isfile(path_l):
                print(f"[ERROR] File not found ({algo} LLM): {path_l}", file=sys.stderr)
                sys.exit(1)
            rl = truncate_series(read_eval_csv(path_l), args.max_epoch)
            if rl:
                max_epoch_seen = max(max_epoch_seen, max(r["epoch"] for r in rl))
            hl = _plot_series(ax, rl, args.metric, ci_key, color, "-", "s", ms, show_ci)
            if hl is not None:
                legend_handles.append(hl)
                legend_labels.append(f"{algo} (LLM φ)")
                n_curves += 1
                print(f"  {algo} LLM: {len(rl)} pts from {path_l}")
            else:
                print(f"  [WARN] {algo} LLM CSV empty after truncate: {path_l}")

    if n_curves == 0:
        print("[ERROR] No curves plotted — provide at least one valid --*-baseline or --*-llm CSV.", file=sys.stderr)
        sys.exit(1)

    ax.set_xlabel("Training epoch", fontsize=12)
    ax.set_ylabel(ylabel_map[args.metric], fontsize=12)
    ax.set_title(args.title, fontsize=14)
    if args.metric == "win_rate":
        ax.set_ylim(-0.05, 1.05)
    if args.max_epoch is not None:
        ax.set_xlim(0, float(args.max_epoch))
    elif args.xmax_auto and max_epoch_seen > 0:
        ax.set_xlim(0, max_epoch_seen * 1.02)
    ax.grid(True, alpha=0.3)

    ncol = 2 if len(legend_handles) >= 2 else 1
    bx, by = (float(x.strip()) for x in args.legend_bbox.split(","))
    ax.legend(
        legend_handles,
        legend_labels,
        ncol=ncol,
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
    print(f"Saved: {out} ({n_curves} curves)")


if __name__ == "__main__":
    main()
