#!/usr/bin/env python3
"""
Submit multiple warm-up training runs with different seeds, then automatically
continue from the best checkpoint once all warm-ups finish.

Run this from the project root (login node or any node with sbatch access):

    python scripts/warmup_select_train.py \\
        --config expr_configs/cooperative_MARL_benchmark/academy/counterattack/mappo.yaml \\
        [--num-warmups 3] \\
        [--warmup-epochs 400]

What happens:
  1. N modified configs are created in expr_configs/warmup_<name>_<ts>/ with
     different seeds and max_steps=warmup_epochs.
  2. N SLURM GPU training jobs are submitted (one per warm-up seed).
  3. A lightweight selector job is submitted with afterok dependency on all N jobs.
     It reads the exact warm-up log dirs from a state file written in step 2,
     picks the one with the highest win_rate at epoch ≤ warmup_epochs, and
     submits a continuation GPU job using the original config + --resume <dir>.
"""

import argparse
import json
import os
import pathlib
import copy
import subprocess
import tempfile
import time
from typing import List, Optional

import yaml


SEEDS = [42, 137, 271, 503, 997]  # first N used for N warm-ups
PYTHON = "/home/chen.sihe1/.conda/envs/grf_env/bin/python"

DEFAULT_SLURM = {
    "num_gpus": 1,
    "gpu_type": None,
    "cpus_per_task": 16,
    "memory_gb": 128,
    "time_hours": 8,
    "partition": "gpu",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_slurm_config(cfg: dict) -> dict:
    """Merge config's slurm_config section with defaults (mirrors submit_training.py)."""
    merged = {**DEFAULT_SLURM}
    merged.update(cfg.get("slurm_config", {}))
    return merged


def slurm_flags_from_config(slurm_cfg: dict) -> List[str]:
    """Build sbatch flags from a slurm_config dict."""
    if slurm_cfg.get("gpu_type"):
        gres = f"gpu:{slurm_cfg['gpu_type']}:{slurm_cfg['num_gpus']}"
    else:
        gres = f"gpu:{slurm_cfg['num_gpus']}"
    hours = int(slurm_cfg["time_hours"])
    return [
        f"--gres={gres}",
        f"--cpus-per-task={slurm_cfg['cpus_per_task']}",
        f"--mem={slurm_cfg['memory_gb']}G",
        f"--time={hours}:00:00",
        f"--partition={slurm_cfg['partition']}",
    ]


def dump_yaml(cfg: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def create_warmup_config(orig_config_path: str, seed_idx: int,
                         warmup_epochs: int, output_dir: str):
    """Return (config_path, warmup_expr_name, original_expr_name)."""
    cfg = load_yaml(orig_config_path)
    orig_name = cfg["expr_name"]
    orig_group = cfg.get("expr_group", "gr_football")

    cfg = copy.deepcopy(cfg)
    warmup_name = f"{orig_name}_warmup_seed{seed_idx}"
    cfg["expr_name"] = warmup_name
    cfg["seed"] = SEEDS[seed_idx]
    cfg["rollout_manager"]["seed"] = SEEDS[seed_idx] * 7 + 1
    cfg["framework"]["stopper"]["kwargs"]["max_steps"] = warmup_epochs

    config_filename = f"warmup_seed{seed_idx}_{pathlib.Path(orig_config_path).name}"
    config_path = os.path.join(output_dir, config_filename)
    dump_yaml(cfg, config_path)
    return config_path, warmup_name, orig_name, orig_group


def _sbatch(cmd: List[str]) -> str:
    """Run an sbatch command, raise with the actual SLURM error on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed:\n  cmd: {' '.join(cmd)}\n  stderr: {result.stderr.strip()}"
        )
    return result.stdout.strip().split(";")[0]


def num_queued_jobs() -> int:
    """Count the user's current pending+running SLURM jobs."""
    result = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "--format=%i", "--noheader"],
        capture_output=True, text=True,
    )
    return len([l for l in result.stdout.strip().splitlines() if l.strip()])


def sbatch_train(config_rel_path: str,
                 dependency_ids: Optional[List[str]] = None,
                 resume_dir: Optional[str] = None,
                 extra_flags: Optional[List[str]] = None) -> str:
    """Submit train.slurm and return job ID."""
    cmd = ["sbatch", "--parsable"]
    if dependency_ids:
        cmd += ["--dependency", "afterok:" + ":".join(dependency_ids)]
    if extra_flags:
        cmd += extra_flags
    cmd += ["train.slurm", "--config", config_rel_path]
    if resume_dir:
        cmd += ["--resume", resume_dir]
    return _sbatch(cmd)


def sbatch_wrap(wrap_cmd: str,
                dependency_ids: Optional[List[str]] = None,
                extra_flags: Optional[List[str]] = None) -> str:
    """Submit a lightweight --wrap job and return job ID."""
    cmd = ["sbatch", "--parsable"]
    if dependency_ids:
        # afterany: selector runs even if some warmups time out or fail,
        # so it can still pick the best from whichever seeds succeeded.
        cmd += ["--dependency", "afterany:" + ":".join(dependency_ids)]
    if extra_flags:
        cmd += extra_flags
    cmd += ["--wrap", wrap_cmd]
    return _sbatch(cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed warm-up + best-checkpoint selection"
    )
    parser.add_argument("--config", required=True,
                        help="Path to the original (full-length) experiment config")
    parser.add_argument("--num-warmups", type=int, default=3,
                        help="Number of parallel warm-up seeds (default: 3, max: 5)")
    parser.add_argument("--warmup-epochs", type=int, default=400,
                        help="Training epochs for each warm-up run (default: 400)")
    args = parser.parse_args()

    if args.num_warmups > len(SEEDS):
        raise ValueError(f"--num-warmups max is {len(SEEDS)}, got {args.num_warmups}")

    # Each run needs num_warmups training jobs + 1 selector job.
    # Warn if the cluster QOS limit might be exceeded.
    JOBS_NEEDED = args.num_warmups + 1
    current = num_queued_jobs()
    QOS_LIMIT = 8  # QOSMaxSubmitJobPerUserLimit observed on this cluster
    if current + JOBS_NEEDED > QOS_LIMIT:
        print(
            f"[WARNING] Currently {current} jobs in queue; need {JOBS_NEEDED} more slots. "
            f"Cluster limit is {QOS_LIMIT}. Wait for jobs to finish before submitting more.\n"
        )

    project_root = pathlib.Path(__file__).resolve().parent.parent
    orig_config_abs = str(pathlib.Path(args.config).resolve())
    orig_cfg = load_yaml(orig_config_abs)
    orig_name = orig_cfg["expr_name"]
    orig_group = orig_cfg.get("expr_group", "gr_football")
    log_base = project_root / "logs" / orig_group
    slurm_cfg = get_slurm_config(orig_cfg)
    job_flags = slurm_flags_from_config(slurm_cfg)

    # Directory for warm-up configs (kept so the selector SLURM job can read them)
    ts = time.strftime("%Y%m%d_%H%M%S")
    config_dir = project_root / "expr_configs" / f"warmup_{orig_name}_{ts}"
    config_dir.mkdir(parents=True, exist_ok=True)

    # State file: tells the selector exactly which log dirs to inspect
    state_file = project_root / "logs" / f"warmup_state_{orig_name}_{ts}.json"

    print(f"Warm-up configs: {config_dir}")
    print(f"State file:      {state_file}")
    print()

    # ------------------------------------------------------------------
    # Step 1: submit N warm-up GPU jobs
    # ------------------------------------------------------------------
    warmup_job_ids: List[str] = []
    warmup_expr_names: List[str] = []

    for i in range(args.num_warmups):
        config_path, warmup_name, _, _ = create_warmup_config(
            orig_config_abs, i, args.warmup_epochs, str(config_dir)
        )
        rel_config = os.path.relpath(config_path, project_root)
        job_id = sbatch_train(rel_config, extra_flags=job_flags)
        warmup_job_ids.append(job_id)
        warmup_expr_names.append(warmup_name)
        print(f"  Warm-up seed {i}: job {job_id}  expr={warmup_name}")

    # Write state so the selector knows which expr_names to inspect
    state = {
        "orig_config": orig_config_abs,
        "orig_name": orig_name,
        "orig_group": orig_group,
        "warmup_epochs": args.warmup_epochs,
        "warmup_expr_names": warmup_expr_names,
        "warmup_job_ids": warmup_job_ids,
        "log_base": str(log_base),
        "submitted_at": ts,
    }
    os.makedirs(str(project_root / "logs"), exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    # ------------------------------------------------------------------
    # Step 2: submit lightweight selector job (afterok all warm-ups)
    # ------------------------------------------------------------------
    selector_cmd = (
        f"cd {project_root} && "
        f"{PYTHON} scripts/select_best_warmup.py --state-file {state_file}"
    )

    # Selector is lightweight but must land on the same partition as the training jobs
    if slurm_cfg.get("gpu_type"):
        selector_gres = f"gpu:{slurm_cfg['gpu_type']}:1"
    else:
        selector_gres = "gpu:1"
    selector_flags = [
        "--job-name=warmup_select",
        "--cpus-per-task=2",
        f"--gres={selector_gres}",
        "--mem=8GB",
        "--time=0:30:00",
        f"--partition={slurm_cfg['partition']}",
        f"--output={project_root}/logs/warmup_select_{orig_name}_{ts}_%j.log",
    ]
    selector_id = sbatch_wrap(
        selector_cmd,
        dependency_ids=warmup_job_ids,
        extra_flags=selector_flags,
    )

    print()
    print(f"  Selector job: {selector_id}  (afterok: {warmup_job_ids})")
    print()
    print("Dependency chain:")
    print(f"  warmup jobs {warmup_job_ids}")
    print(f"  → selector  {selector_id}")
    print(f"  → continuation  (submitted by selector)")
    print()
    print(f"Monitor:  squeue -u $USER")
    print(f"Logs:     {project_root}/logs/warmup_select_{orig_name}_{ts}_<JOB>.log")


if __name__ == "__main__":
    main()
