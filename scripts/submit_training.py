#!/usr/bin/env python
# Copyright 2022 Digital Brain Laboratory
# Helper script to submit training jobs with sbatch, with checkpoint management

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Default SLURM configuration
DEFAULT_SLURM_CONFIG = {
    "num_gpus": 1,
    "gpu_type": None,         # Optional: v100-sxm2, h200, a100, etc. None = any GPU
    "cpus_per_task": 16,
    "memory_gb": 128,
    "time_hours": 72,
    "partition": "gpu"        # gpu, gpu-short, gpu-interactive, sharing
}


def _inject_conda_export_for_sbatch(sbatch_cmd, export_mode="all"):
    """Forward CONDA_ENV_NAME into the job (train.slurm uses conda activate "${CONDA_ENV_NAME:-grf_env}").

    export_mode:
      all     --export=ALL,CONDA_ENV_NAME=... (default; can be huge on login nodes)
      minimal --export=NONE,CONDA_ENV_NAME=... (smaller env; job still gets SLURM_* / basic vars from Slurm)
    """
    name = os.environ.get("CONDA_ENV_NAME")
    if not name:
        return
    if export_mode == "minimal":
        sbatch_cmd.insert(1, f"--export=NONE,CONDA_ENV_NAME={name}")
    else:
        sbatch_cmd.insert(1, f"--export=ALL,CONDA_ENV_NAME={name}")


def load_config_file(config_path):
    """Load YAML config file and extract slurm_config if available."""
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        return {}

def get_slurm_config(config_path):
    """Extract SLURM configuration from training config or use defaults."""
    cfg = load_config_file(config_path)
    slurm_cfg = cfg.get('slurm_config', {})

    # Merge with defaults
    config = {**DEFAULT_SLURM_CONFIG}
    config.update(slurm_cfg)

    return config


def _log_base_from_training_cfg(cfg):
    """Same layout as train.slurm auto-resume: <log_dir>/<expr_group>/<expr_name>."""
    if not cfg:
        return None
    log_dir = cfg.get("log_dir", "./logs")
    if isinstance(log_dir, str):
        log_dir = log_dir.strip().lstrip("./") or "logs"
    else:
        log_dir = "logs"
    expr_group = cfg.get("expr_group") or "gr_football"
    expr_name = cfg.get("expr_name")
    if not expr_name:
        return None
    return Path(log_dir) / expr_group / expr_name


def find_latest_checkpoint(expr_group, expr_name, log_dir="logs"):
    """Find the latest run directory (timestamp folder) under logs/<group>/<name>."""
    log_base = Path(log_dir) / expr_group / expr_name

    if not log_base.exists():
        return None

    checkpoints = sorted([d for d in log_base.iterdir() if d.is_dir()])

    if not checkpoints:
        return None

    return str(checkpoints[-1])


def find_latest_checkpoint_from_config(config_path):
    """Resolve latest checkpoint using expr_group / expr_name / log_dir in the yaml."""
    cfg = load_config_file(config_path)
    log_base = _log_base_from_training_cfg(cfg)
    if log_base is None or not log_base.exists():
        return None
    checkpoints = sorted([d for d in log_base.iterdir() if d.is_dir()])
    if not checkpoints:
        return None
    return str(checkpoints[-1])

def submit_training_job(
    config_path,
    checkpoint_dir=None,
    job_name=None,
    no_submit=False,
    slurm_export="all",
):
    """Submit a training job using sbatch."""

    # Validate config file
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Get SLURM configuration from config file
    slurm_cfg = get_slurm_config(config_path)

    print(f"SLURM Configuration:")
    print(f"  GPUs: {slurm_cfg['num_gpus']}", end="")
    if slurm_cfg.get('gpu_type'):
        print(f" ({slurm_cfg['gpu_type']})")
    else:
        print(" (any type)")
    print(f"  CPUs per task: {slurm_cfg['cpus_per_task']}")
    print(f"  Memory: {slurm_cfg['memory_gb']}GB")
    print(f"  Time limit: {slurm_cfg['time_hours']}h")
    print(f"  Partition: {slurm_cfg['partition']}")
    print()

    # Parse experiment info from config path or provide defaults
    if job_name is None:
        config_name = os.path.basename(config_path).replace(".yaml", "")
        job_name = f"marl_{config_name}"

    # Build sbatch command
    slurm_script = "train.slurm"
    if not os.path.exists(slurm_script):
        print(f"Error: SLURM script not found: {slurm_script}")
        sys.exit(1)

    # Convert time to HH:MM:SS format
    hours = int(slurm_cfg['time_hours'])
    time_str = f"{hours}:00:00"

    # Build GPU resource string (--gres)
    if slurm_cfg.get('gpu_type'):
        gpu_str = f"gpu:{slurm_cfg['gpu_type']}:{slurm_cfg['num_gpus']}"
    else:
        gpu_str = f"gpu:{slurm_cfg['num_gpus']}"

    sbatch_cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--gres={gpu_str}",
        f"--cpus-per-task={slurm_cfg['cpus_per_task']}",
        f"--mem={slurm_cfg['memory_gb']}G",
        f"--time={time_str}",
        f"--partition={slurm_cfg['partition']}"
    ]
    _inject_conda_export_for_sbatch(sbatch_cmd, export_mode=slurm_export)

    # Add training arguments
    sbatch_cmd.append(slurm_script)
    sbatch_cmd.extend(["--config", config_path])

    if checkpoint_dir:
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
            sys.exit(1)
        sbatch_cmd.extend(["--checkpoint", checkpoint_dir])
        print(f"Submitting training job to resume from: {checkpoint_dir}")
    else:
        print(f"Submitting fresh training job")

    print(f"Command: {' '.join(sbatch_cmd)}")

    if no_submit:
        print("(--no-submit flag set, not submitting)")
        return None

    try:
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"Job submitted successfully. Job ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job (exit {e.returncode})")
        if e.stdout:
            print("sbatch stdout:", e.stdout)
        if e.stderr:
            print("sbatch stderr:", e.stderr)
        sys.exit(1)

def list_checkpoints(expr_group=None, expr_name=None):
    """List available checkpoints."""
    log_base = Path("logs")

    if not log_base.exists():
        print("No logs directory found")
        return

    print("Available checkpoints:")
    print("=" * 80)

    for group_dir in sorted(log_base.iterdir()):
        if not group_dir.is_dir():
            continue

        if expr_group and group_dir.name != expr_group:
            continue

        for exp_dir in sorted(group_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            if expr_name and exp_dir.name != expr_name:
                continue

            for checkpoint_dir in sorted(exp_dir.iterdir()):
                if checkpoint_dir.is_dir():
                    config_file = checkpoint_dir / "config.yaml"
                    if config_file.exists():
                        print(f"{checkpoint_dir}")

def chain_submit_jobs(
    config_path, num_jobs=2, job_name=None, no_submit=False, slurm_export="all"
):
    """Submit a chain of dependent jobs that resume from checkpoints.

    Each job depends on the previous one completing, and automatically
    resumes from the latest checkpoint.
    """

    # Validate config file
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Extract experiment info from config path
    config_parts = Path(config_path).parts
    if len(config_parts) < 3:
        print("Error: Cannot determine experiment from config path")
        sys.exit(1)

    expr_group = config_parts[1]
    if "benchmark" in config_parts[2]:
        expr_name = "benchmark_" + config_parts[-2] + "_" + config_parts[-1].replace(".yaml", "")
    else:
        expr_name = "_".join(config_parts[2:-1]) + "_" + config_parts[-1].replace(".yaml", "")

    print(f"Chain submitting {num_jobs} dependent jobs")
    print(f"  Experiment: {expr_group}/{expr_name}")
    print(f"  Config: {config_path}")
    print(f"  Each job will resume from latest checkpoint\n")

    job_ids = []
    prev_job_id = None

    for job_num in range(1, num_jobs + 1):
        print(f"Submitting job {job_num}/{num_jobs}...")

        print(f"  Job {job_num} will dynamically find the latest checkpoint at execution time")

        # Get SLURM config
        slurm_cfg = get_slurm_config(config_path)

        # Build sbatch command
        slurm_script = "train.slurm"
        if not os.path.exists(slurm_script):
            print(f"Error: SLURM script not found: {slurm_script}")
            sys.exit(1)

        # Build base sbatch command
        hours = int(slurm_cfg['time_hours'])
        time_str = f"{hours}:00:00"

        if slurm_cfg.get('gpu_type'):
            gpu_str = f"gpu:{slurm_cfg['gpu_type']}:{slurm_cfg['num_gpus']}"
        else:
            gpu_str = f"gpu:{slurm_cfg['num_gpus']}"

        job_name_with_num = f"{job_name}_part{job_num}" if job_name else f"marl_chain_part{job_num}"

        sbatch_cmd = [
            "sbatch",
            f"--job-name={job_name_with_num}",
            f"--gres={gpu_str}",
            f"--cpus-per-task={slurm_cfg['cpus_per_task']}",
            f"--mem={slurm_cfg['memory_gb']}G",
            f"--time={time_str}",
            f"--partition={slurm_cfg['partition']}"
        ]
        _inject_conda_export_for_sbatch(sbatch_cmd, export_mode=slurm_export)

        # Add dependency if not first job
        if prev_job_id:
            sbatch_cmd.append(f"--dependency=afterany:{prev_job_id}")

        sbatch_cmd.append(slurm_script)
        sbatch_cmd.extend(["--config", config_path])

        # Enable dynamic auto-resume for the slurm script
        sbatch_cmd.append("--auto-resume")

        print(f"  Command: {' '.join(sbatch_cmd)}\n")

        if no_submit:
            job_ids.append(f"SIMULATED_{job_num}")
        else:
            try:
                result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
                print(f"  Job {job_num} submitted. Job ID: {job_id}\n")
                prev_job_id = job_id
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job {job_num} (exit {e.returncode})")
                if e.stdout:
                    print("sbatch stdout:", e.stdout)
                if e.stderr:
                    print("sbatch stderr:", e.stderr)
                sys.exit(1)

    # Print summary
    print("=" * 80)
    print(f"Chain submission complete:")
    print(f"  Total jobs: {num_jobs}")
    print(f"  Job IDs: {', '.join(job_ids)}")
    print(f"\nEach job will:")
    print(f"  1. Wait for previous job to complete")
    print(f"  2. Find latest checkpoint from: logs/{expr_group}/{expr_name}/")
    print(f"  3. Resume training from that checkpoint")
    print(f"  4. Save new checkpoint on completion")
    print(f"\nMonitor with: squeue -u $USER")
    print(f"Cancel all with: scancel {job_ids[0]} (will cancel chain)")
    print("=" * 80)


def main():
    """Main entry point for the training submission script."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Helper script for submitting Light-MALib training jobs to SLURM"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a new training job")
    submit_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to training config file"
    )
    submit_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint directory to resume from (optional)"
    )
    submit_parser.add_argument(
        "--auto-resume", action="store_true",
        help="Automatically find and resume from latest checkpoint"
    )
    submit_parser.add_argument(
        "--job-name", type=str, default=None,
        help="SLURM job name"
    )
    submit_parser.add_argument(
        "--no-submit", action="store_true",
        help="Print the command without submitting"
    )
    submit_parser.add_argument(
        "--slurm-export",
        choices=("all", "minimal"),
        default="all",
        help="Pass CONDA_ENV_NAME into the batch job: 'minimal' uses --export=NONE,... only "
        "(smaller env; try on login nodes that OOM-kill or reject huge --export=ALL).",
    )

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume a training job from latest checkpoint")
    resume_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to training config file"
    )
    resume_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Specific checkpoint directory (if not specified, uses latest)"
    )
    resume_parser.add_argument(
        "--job-name", type=str, default=None,
        help="SLURM job name"
    )
    resume_parser.add_argument(
        "--no-submit", action="store_true",
        help="Print the command without submitting"
    )
    resume_parser.add_argument(
        "--slurm-export",
        choices=("all", "minimal"),
        default="all",
        help="Same as submit --slurm-export.",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available checkpoints")
    list_parser.add_argument(
        "--group", type=str, default=None,
        help="Filter by experiment group"
    )
    list_parser.add_argument(
        "--name", type=str, default=None,
        help="Filter by experiment name"
    )

    # Chain submit command
    chain_parser = subparsers.add_parser("chain-submit", help="Submit a chain of dependent jobs that auto-continue on timeout")
    chain_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to training config file"
    )
    chain_parser.add_argument(
        "--num-jobs", type=int, default=2,
        help="Number of jobs to chain (each job auto-resumes from checkpoint)"
    )
    chain_parser.add_argument(
        "--job-name", type=str, default=None,
        help="SLURM job name (will be appended with _part1, _part2, etc)"
    )
    chain_parser.add_argument(
        "--no-submit", action="store_true",
        help="Print the commands without submitting"
    )
    chain_parser.add_argument(
        "--slurm-export",
        choices=("all", "minimal"),
        default="all",
        help="Same as submit --slurm-export.",
    )

    args = parser.parse_args()

    if args.command == "submit":
        checkpoint = args.checkpoint
        if args.auto_resume and not checkpoint:
            checkpoint = find_latest_checkpoint_from_config(args.config)
            if checkpoint:
                print(f"Found latest checkpoint: {checkpoint}")
            else:
                print("No checkpoint found in yaml log_dir/expr_group/expr_name; starting fresh.")

        submit_training_job(
            args.config,
            checkpoint,
            args.job_name,
            args.no_submit,
            slurm_export=args.slurm_export,
        )

    elif args.command == "resume":
        checkpoint = args.checkpoint
        if not checkpoint:
            checkpoint = find_latest_checkpoint_from_config(args.config)
            if checkpoint:
                print(f"Found latest checkpoint: {checkpoint}")
            else:
                cfg = load_config_file(args.config)
                log_base = _log_base_from_training_cfg(cfg)
                print(f"No checkpoint found under {log_base}")
                sys.exit(1)

        submit_training_job(
            args.config,
            checkpoint,
            args.job_name,
            args.no_submit,
            slurm_export=args.slurm_export,
        )

    elif args.command == "list":
        list_checkpoints(args.group, args.name)

    elif args.command == "chain-submit":
        chain_submit_jobs(
            args.config,
            args.num_jobs,
            args.job_name,
            args.no_submit,
            slurm_export=args.slurm_export,
        )


if __name__ == "__main__":
    main()
