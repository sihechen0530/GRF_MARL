#!/usr/bin/env python3
"""
Find the best warm-up run by win_rate at/near warmup_epochs, then submit
a continuation SLURM job from that checkpoint using the original config.

Normally invoked automatically by SLURM (via --wrap) after all warm-up jobs
finish.  Can also be run manually:

    python scripts/select_best_warmup.py --state-file logs/warmup_state_<name>_<ts>.json
    python scripts/select_best_warmup.py --state-file ... --dry-run   # preview only
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
from typing import Optional, Tuple


def find_latest_run_dir(log_base: str, expr_name: str) -> Optional[pathlib.Path]:
    """Return the most recently started run directory for expr_name, or None."""
    base = pathlib.Path(log_base) / expr_name
    candidates = sorted(
        base.glob("????-??-??-??-??-??"),
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


# Pattern: "Rollout Eval 400: average reward: 141.7, average win: 0.6854"
_EVAL_RE = re.compile(
    r"Rollout Eval\s+(\d+):.*?average win:\s*([\d.]+)",
    re.IGNORECASE,
)


def best_win_rate_at_or_before(run_dir: pathlib.Path, max_epoch: int) -> Tuple[float, int]:
    """
    Parse the SLURM training log for lines like:
        Rollout Eval <epoch>: ... average win: <value>
    Returns (best_win_rate, best_epoch) for epochs ≤ max_epoch.
    Returns (-1.0, -1) if no usable data is found.
    """
    log_files = sorted(run_dir.glob("slurm_*.log"))
    if not log_files:
        print(f"  [WARN] No slurm_*.log found in {run_dir}")
        return -1.0, -1

    best_win = -1.0
    best_epoch = -1
    for log_file in log_files:
        try:
            text = log_file.read_text(errors="replace")
        except Exception as exc:
            print(f"  [WARN] Could not read {log_file}: {exc}")
            continue
        for m in _EVAL_RE.finditer(text):
            epoch = int(m.group(1))
            win = float(m.group(2))
            if epoch <= max_epoch and win > best_win:
                best_win = win
                best_epoch = epoch

    return best_win, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select best warm-up checkpoint and submit continuation"
    )
    parser.add_argument("--state-file", required=True,
                        help="JSON state file written by warmup_select_train.py")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selection result but do not submit the continuation job")
    args = parser.parse_args()

    with open(args.state_file) as f:
        state = json.load(f)

    orig_config = state["orig_config"]
    warmup_epochs = state["warmup_epochs"]
    warmup_expr_names = state["warmup_expr_names"]
    log_base = state["log_base"]

    project_root = pathlib.Path(args.state_file).resolve().parent.parent

    print("=" * 60)
    print("Warm-up selection")
    print("=" * 60)
    print(f"Config:         {orig_config}")
    print(f"Warmup epochs:  {warmup_epochs}")
    print()

    best_score: float = -1.0
    best_run_dir: Optional[pathlib.Path] = None
    best_name: str = ""
    best_epoch_found: int = -1

    for name in warmup_expr_names:
        run_dir = find_latest_run_dir(log_base, name)
        if run_dir is None:
            print(f"  {name}: NO RUN DIRECTORY FOUND")
            continue
        score, epoch_found = best_win_rate_at_or_before(run_dir, warmup_epochs)
        marker = ""
        if score > best_score:
            best_score = score
            best_run_dir = run_dir
            best_name = name
            best_epoch_found = epoch_found
            marker = "  ← BEST"
        epoch_str = f"(epoch {epoch_found})" if epoch_found >= 0 else "(no data)"
        print(f"  {name}")
        print(f"    peak win_rate = {score:.4f} {epoch_str}{marker}")
        print(f"    dir = {run_dir}")

    print()

    if best_run_dir is None or best_score < 0:
        print("[ERROR] No valid warm-up results found. Aborting.")
        return

    print(f"Winner: {best_name}")
    print(f"  peak win_rate ≤ epoch {warmup_epochs}: {best_score:.4f} (epoch {best_epoch_found})")
    print(f"  checkpoint:  {best_run_dir}")
    print()

    # Update state file with the selection result for traceability
    state["selected_name"] = best_name
    state["selected_dir"] = str(best_run_dir)
    state["selected_win_rate"] = best_score
    state["selected_epoch"] = best_epoch_found
    with open(args.state_file, "w") as f:
        json.dump(state, f, indent=2)

    # ------------------------------------------------------------------
    # Submit continuation job: original config + resume from best dir
    # ------------------------------------------------------------------
    rel_config = os.path.relpath(orig_config, project_root)
    cmd = [
        "sbatch",
        str(project_root / "train.slurm"),
        "--config", rel_config,
        "--resume", str(best_run_dir),
    ]

    print(f"Submitting continuation:")
    print(f"  {' '.join(cmd)}")

    if args.dry_run:
        print("[DRY RUN] Not submitting.")
        return

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
            cwd=str(project_root),
        )
        job_id = result.stdout.strip().split(";")[0]
        print(f"\nContinuation job submitted: {job_id}")
        print(f"Monitor:  squeue -u $USER -j {job_id}")
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] sbatch failed:\n{exc.stderr}")


if __name__ == "__main__":
    main()
