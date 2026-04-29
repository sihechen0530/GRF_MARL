#!/usr/bin/env python
# Copyright 2022 Digital Brain Laboratory
# Helper script to submit training jobs with sbatch, with checkpoint management.

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_SLURM_CONFIG = {
    "num_gpus": 1,
    "gpu_type": None,
    "cpus_per_task": 16,
    "memory_gb": 128,
    "time_hours": 72,
    "partition": "gpu",
    # Whole-node allocation avoids multiple Ray jobs fighting on the same node.
    # Set slurm_config.exclusive: false in YAML if the partition rejects it.
    "exclusive": True,
}


def _inject_conda_export_for_sbatch(sbatch_cmd, export_mode="all", conda_env=None):
    """Forward CONDA_ENV_NAME into the job when requested."""
    name = conda_env or os.environ.get("CONDA_ENV_NAME")
    if not name:
        return
    if export_mode == "minimal":
        sbatch_cmd.insert(1, f"--export=NONE,CONDA_ENV_NAME={name}")
    else:
        sbatch_cmd.insert(1, f"--export=ALL,CONDA_ENV_NAME={name}")


def load_config_file(config_path):
    """Load YAML config file and extract slurm_config if available."""
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg or {}
    except Exception as exc:
        print(f"Warning: Could not load config file {config_path}: {exc}")
        return {}


def get_slurm_config(config_path):
    """Extract SLURM configuration from training config or use defaults."""
    cfg = load_config_file(config_path)
    config = {**DEFAULT_SLURM_CONFIG}
    config.update(cfg.get("slurm_config", {}) or {})
    return config


def get_expr_info(config_path):
    """Read expr_group, expr_name, and log_dir directly from the config file."""
    cfg = load_config_file(config_path)
    expr_group = cfg.get("expr_group", "gr_football")
    expr_name = cfg.get("expr_name", Path(config_path).stem)
    log_dir = cfg.get("log_dir", "logs")
    if isinstance(log_dir, str):
        log_dir = log_dir.lstrip("./").rstrip("/") or "logs"
    else:
        log_dir = "logs"
    return expr_group, expr_name, log_dir


def find_latest_checkpoint(expr_group, expr_name, log_dir="logs"):
    """Find the latest timestamped run directory for an experiment."""
    log_base = Path(log_dir) / expr_group / expr_name
    if not log_base.exists():
        return None

    checkpoints = sorted(d for d in log_base.iterdir() if d.is_dir())
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def _sbatch_resource_args(config_path, job_name):
    slurm_cfg = get_slurm_config(config_path)

    print("SLURM Configuration:")
    print(f"  GPUs: {slurm_cfg['num_gpus']}", end="")
    if slurm_cfg.get("gpu_type"):
        print(f" ({slurm_cfg['gpu_type']})")
    else:
        print(" (any type)")
    print(f"  CPUs per task: {slurm_cfg['cpus_per_task']}")
    print(f"  Memory: {slurm_cfg['memory_gb']}GB")
    print(f"  Time limit: {slurm_cfg['time_hours']}h")
    print(f"  Partition: {slurm_cfg['partition']}")
    print(f"  Exclusive node: {slurm_cfg.get('exclusive', False)}")
    print()

    hours = int(slurm_cfg["time_hours"])
    time_str = f"{hours}:00:00"
    if slurm_cfg.get("gpu_type"):
        gpu_str = f"gpu:{slurm_cfg['gpu_type']}:{slurm_cfg['num_gpus']}"
    else:
        gpu_str = f"gpu:{slurm_cfg['num_gpus']}"

    args = [
        "sbatch",
        f"--job-name={job_name}",
        f"--gres={gpu_str}",
        f"--cpus-per-task={slurm_cfg['cpus_per_task']}",
        f"--mem={slurm_cfg['memory_gb']}G",
        f"--time={time_str}",
        f"--partition={slurm_cfg['partition']}",
    ]
    if slurm_cfg.get("exclusive", False):
        args.append("--exclusive")
    return args


def submit_eval_job(config_path, run_dir=None, depends_on=None, job_name=None, no_submit=False):
    """Submit an evaluation job, optionally dependent on a training job."""
    eval_script = "eval.slurm"
    if not os.path.exists(eval_script):
        print(f"Error: SLURM eval script not found: {eval_script}")
        sys.exit(1)

    if job_name is None:
        config_name = os.path.basename(config_path).replace(".yaml", "")
        job_name = f"eval_{config_name}"

    sbatch_cmd = ["sbatch", f"--job-name={job_name}"]
    if depends_on:
        sbatch_cmd.append(f"--dependency=afterany:{depends_on}")

    sbatch_cmd.append(eval_script)
    sbatch_cmd.extend(["--config", config_path])
    if run_dir:
        sbatch_cmd.extend(["--run-dir", run_dir])

    print(f"Submitting eval job (depends on training job {depends_on})...")
    print(f"Command: {' '.join(sbatch_cmd)}")

    if no_submit:
        print("(--no-submit flag set, not submitting)")
        return None

    try:
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"Eval job submitted. Job ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as exc:
        print(f"Error submitting eval job (exit {exc.returncode})")
        if exc.stdout:
            print("sbatch stdout:", exc.stdout)
        if exc.stderr:
            print("sbatch stderr:", exc.stderr)
        sys.exit(1)


def submit_training_job(
    config_path,
    run_dir=None,
    job_name=None,
    no_submit=False,
    slurm_export="all",
    conda_env=None,
):
    """Submit a training job using sbatch."""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    if job_name is None:
        config_name = os.path.basename(config_path).replace(".yaml", "")
        job_name = f"marl_{config_name}"

    slurm_script = "train.slurm"
    if not os.path.exists(slurm_script):
        print(f"Error: SLURM script not found: {slurm_script}")
        sys.exit(1)

    sbatch_cmd = _sbatch_resource_args(config_path, job_name)
    _inject_conda_export_for_sbatch(
        sbatch_cmd, export_mode=slurm_export, conda_env=conda_env
    )

    sbatch_cmd.append(slurm_script)
    sbatch_cmd.extend(["--config", config_path])

    if run_dir:
        if not os.path.exists(run_dir):
            print(f"Error: Resume directory not found: {run_dir}")
            sys.exit(1)
        sbatch_cmd.extend(["--resume", run_dir])
        print(f"Submitting training job to resume from: {run_dir}")
    else:
        print("Submitting fresh training job")

    print(f"Command: {' '.join(sbatch_cmd)}")

    if no_submit:
        print("(--no-submit flag set, not submitting)")
        return None

    try:
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"Job submitted successfully. Job ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as exc:
        print(f"Error submitting job (exit {exc.returncode})")
        if exc.stdout:
            print("sbatch stdout:", exc.stdout)
        if exc.stderr:
            print("sbatch stderr:", exc.stderr)
        sys.exit(1)


def list_checkpoints(expr_group=None, expr_name=None, log_dir="logs"):
    """List available timestamped run directories."""
    log_base = Path(log_dir)
    if not log_base.exists():
        print(f"No {log_dir} directory found")
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
                        print(checkpoint_dir)


def chain_submit_jobs(
    config_path,
    num_jobs=2,
    job_name=None,
    no_submit=False,
    with_eval=True,
    fresh=False,
    log_base="logs",
    slurm_export="all",
    conda_env=None,
):
    """Submit a chain of dependent jobs that auto-continue on timeout."""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    expr_group, expr_name, _ = get_expr_info(config_path)

    print(f"Chain submitting {num_jobs} dependent jobs")
    print(f"  Experiment: {expr_group}/{expr_name}")
    print(f"  Config: {config_path}")
    if fresh:
        print("  Mode: fresh (job 1 starts new run, subsequent jobs auto-resume)")
    else:
        print("  Each job will resume from latest checkpoint")
    print(f"  Log base: {log_base}/{expr_group}/{expr_name}/")
    print()

    job_ids = []
    prev_job_id = None

    for job_num in range(1, num_jobs + 1):
        print(f"Submitting job {job_num}/{num_jobs}...")
        is_first_fresh = fresh and job_num == 1
        if is_first_fresh:
            print(f"  Job {job_num} will start a fresh training run")
        else:
            print(f"  Job {job_num} will dynamically find the latest checkpoint")

        job_name_with_num = (
            f"{job_name}_part{job_num}" if job_name else f"marl_chain_part{job_num}"
        )
        sbatch_cmd = _sbatch_resource_args(config_path, job_name_with_num)
        _inject_conda_export_for_sbatch(
            sbatch_cmd, export_mode=slurm_export, conda_env=conda_env
        )

        if prev_job_id:
            sbatch_cmd.append(f"--dependency=afterany:{prev_job_id}")

        sbatch_cmd.append("train.slurm")
        sbatch_cmd.extend(["--config", config_path])

        if not is_first_fresh:
            sbatch_cmd.append("--auto-resume")
        if log_base != "logs":
            sbatch_cmd.extend(["--log-base", log_base])

        print(f"  Command: {' '.join(sbatch_cmd)}\n")

        if no_submit:
            job_id = f"SIMULATED_{job_num}"
            job_ids.append(job_id)
            prev_job_id = job_id
            continue

        try:
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            prev_job_id = job_id
            print(f"  Job {job_num} submitted. Job ID: {job_id}\n")
        except subprocess.CalledProcessError as exc:
            print(f"Error submitting job {job_num} (exit {exc.returncode})")
            if exc.stdout:
                print("sbatch stdout:", exc.stdout)
            if exc.stderr:
                print("sbatch stderr:", exc.stderr)
            sys.exit(1)

    if with_eval and job_ids:
        last_job_id = job_ids[-1] if not no_submit else None
        eval_name = f"{job_name}_eval" if job_name else "marl_chain_eval"
        print("\nSubmitting eval job after last training job...")
        submit_eval_job(config_path, None, last_job_id, eval_name, no_submit)

    print("=" * 80)
    print("Chain submission complete:")
    print(f"  Total jobs: {num_jobs}")
    print(f"  Job IDs: {', '.join(job_ids)}")
    print("\nJobs will:")
    if fresh:
        print(f"  Job 1: start fresh under {log_base}/{expr_group}/{expr_name}/<timestamp>/")
        print("  Jobs 2+: wait for previous job, then auto-resume from that new run")
    else:
        print("  1. Wait for previous job to complete")
        print(f"  2. Find latest checkpoint from: {log_base}/{expr_group}/{expr_name}/")
        print("  3. Resume training from that checkpoint")
        print("  4. Save new checkpoint on completion")
    print("\nMonitor with: squeue -u $USER")
    print(f"Cancel all with: scancel {job_ids[0]} (will cancel chain)")
    print("=" * 80)


def main():
    """Main entry point for the training submission script."""
    parser = argparse.ArgumentParser(
        description="Helper script for submitting Light-MALib training jobs to SLURM"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    submit_parser = subparsers.add_parser(
        "submit", help="Submit a training job, resuming from latest run if one exists"
    )
    submit_parser.add_argument("--config", type=str, required=True)
    submit_parser.add_argument("--run-dir", type=str, default=None)
    submit_parser.add_argument("--fresh", action="store_true")
    submit_parser.add_argument("--job-name", type=str, default=None)
    submit_parser.add_argument("--no-eval", action="store_true")
    submit_parser.add_argument("--no-submit", action="store_true")
    submit_parser.add_argument("--slurm-export", choices=("all", "minimal"), default="all")
    submit_parser.add_argument("--conda-env", default=None)

    resume_parser = subparsers.add_parser(
        "resume", help="Resume training from the latest run directory"
    )
    resume_parser.add_argument("--config", type=str, required=True)
    resume_parser.add_argument("--run-dir", type=str, default=None)
    resume_parser.add_argument("--job-name", type=str, default=None)
    resume_parser.add_argument("--no-eval", action="store_true")
    resume_parser.add_argument("--no-submit", action="store_true")
    resume_parser.add_argument("--slurm-export", choices=("all", "minimal"), default="all")
    resume_parser.add_argument("--conda-env", default=None)

    list_parser = subparsers.add_parser("list", help="List available checkpoints")
    list_parser.add_argument("--group", type=str, default=None)
    list_parser.add_argument("--name", type=str, default=None)
    list_parser.add_argument("--log-base", type=str, default="logs")

    chain_parser = subparsers.add_parser(
        "chain-submit", help="Submit a chain of dependent jobs"
    )
    chain_parser.add_argument("--config", type=str, required=True)
    chain_parser.add_argument("--num-jobs", type=int, default=2)
    chain_parser.add_argument("--job-name", type=str, default=None)
    chain_parser.add_argument("--fresh", action="store_true")
    chain_parser.add_argument("--no-eval", action="store_true")
    chain_parser.add_argument("--no-submit", action="store_true")
    chain_parser.add_argument("--slurm-export", choices=("all", "minimal"), default="all")
    chain_parser.add_argument("--conda-env", default=None)
    chain_parser.add_argument("--log-base", type=str, default="logs")

    args = parser.parse_args()

    if args.command == "submit":
        run_dir = args.run_dir
        if not run_dir and not args.fresh:
            expr_group, expr_name, log_dir = get_expr_info(args.config)
            run_dir = find_latest_checkpoint(expr_group, expr_name, log_dir)
            if run_dir:
                print(f"Found existing run, resuming: {run_dir}")
            else:
                print("No existing run found, starting fresh.")

        train_job_id = submit_training_job(
            args.config,
            run_dir,
            args.job_name,
            args.no_submit,
            slurm_export=args.slurm_export,
            conda_env=args.conda_env,
        )
        if not args.no_eval:
            submit_eval_job(args.config, run_dir, train_job_id, args.job_name, args.no_submit)

    elif args.command == "resume":
        run_dir = args.run_dir
        if not run_dir:
            expr_group, expr_name, log_dir = get_expr_info(args.config)
            run_dir = find_latest_checkpoint(expr_group, expr_name, log_dir)
            if run_dir:
                print(f"Found latest run: {run_dir}")
            else:
                print(f"No run found for {expr_group}/{expr_name} in {log_dir}/")
                sys.exit(1)

        train_job_id = submit_training_job(
            args.config,
            run_dir,
            args.job_name,
            args.no_submit,
            slurm_export=args.slurm_export,
            conda_env=args.conda_env,
        )
        if not args.no_eval:
            submit_eval_job(args.config, run_dir, train_job_id, args.job_name, args.no_submit)

    elif args.command == "list":
        list_checkpoints(args.group, args.name, args.log_base)

    elif args.command == "chain-submit":
        chain_submit_jobs(
            args.config,
            args.num_jobs,
            args.job_name,
            args.no_submit,
            with_eval=not args.no_eval,
            fresh=args.fresh,
            log_base=args.log_base,
            slurm_export=args.slurm_export,
            conda_env=args.conda_env,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
