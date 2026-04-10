#!/usr/bin/env python3
"""
Iterative Eureka: LLM-based reward shaping refinement loop.

Workflow per iteration:
  1. Read behavioral metrics JSON (from extract_behavior_metrics.py)
  2. Read current phi source code
  3. Send both to LLM for diagnosis + revision
  4. Parse revised phi, validate, and write to disk
  5. (Optional) Trigger next training round

Usage:
  # Single revision step
  python scripts/iterative_eureka.py revise \
      --metrics metrics/behavior_report.json \
      --current-phi generated_phi/phi_llm.py \
      --output generated_phi/phi_llm_v2.py

  # Full loop: eval → diagnose → revise (requires trained checkpoint)
  python scripts/iterative_eureka.py loop \
      --checkpoint logs/gr_football/<expr>/<ts> \
      --config expr_configs/.../mappo_eureka_llm.yaml \
      --num-iterations 3 \
      --num-eval-games 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_llm():
    """Lazy import LLM dependencies (only needed for revise/loop, not slurm-loop)."""
    from light_malib.llm import (
        chat_completions_text,
        ensure_phi_export,
        extract_python_code_block,
    )
    from light_malib.llm.prompts_phi import (
        build_iterative_eureka_messages,
        messages_as_openai_json,
    )
    return (
        chat_completions_text,
        ensure_phi_export,
        extract_python_code_block,
        build_iterative_eureka_messages,
        messages_as_openai_json,
    )


def _extract_diagnosis(raw: str) -> str:
    """Extract the ```text``` block containing the diagnosis."""
    m = re.search(r"```text\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _file_header(iteration: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"# Auto-generated potential Φ module (Iterative Eureka, iter={iteration}). UTC: {ts}\n"
    )


def revise_phi(
    current_phi_code: str,
    metrics_json: str,
    iteration: int = 1,
    diagnosis_request: str = "",
    scenario_name: str = "GRF 11v11 full game",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> tuple[str, str, str]:
    """Call LLM to diagnose and revise phi. Returns (diagnosis, new_phi_code, raw_response)."""
    (
        chat_completions_text,
        ensure_phi_export,
        extract_python_code_block,
        build_iterative_eureka_messages,
        messages_as_openai_json,
    ) = _import_llm()

    pairs = build_iterative_eureka_messages(
        current_phi_code=current_phi_code,
        behavior_metrics_json=metrics_json,
        iteration=iteration,
        diagnosis_request=diagnosis_request,
        scenario_name=scenario_name,
    )
    messages = messages_as_openai_json(pairs)

    raw = chat_completions_text(messages, temperature=temperature, max_tokens=max_tokens)

    diagnosis = _extract_diagnosis(raw)
    new_code = ensure_phi_export(extract_python_code_block(raw))

    return diagnosis, new_code, raw


def validate_phi_code(code: str, n_left: int = 10, n_right: int = 11) -> bool:
    """Quick sanity check: compile + call phi with a synthetic state."""
    import importlib.util
    import tempfile
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        spec = importlib.util.spec_from_file_location("_phi_test", f.name)
        if spec is None or spec.loader is None:
            return False
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"  Compilation error: {e}")
            return False

    phi = getattr(mod, "phi", None)
    if phi is None:
        print("  No 'phi' function found in generated code")
        return False

    rng = np.random.default_rng(42)
    test_state = {
        "player_idx": 1,
        "is_left_team": True,
        "my_pos": (rng.uniform(-1, 1), rng.uniform(-0.42, 0.42)),
        "my_vel": (0.01, -0.005),
        "has_ball": False,
        "ball_pos": (rng.uniform(-1, 1), rng.uniform(-0.42, 0.42), 0.0),
        "ball_vel": (0.0, 0.0, 0.0),
        "dist_to_goal": rng.uniform(0.5, 2.0),
        "ball_dist_to_goal": rng.uniform(0.3, 2.0),
        "dist_to_ball": rng.uniform(0.1, 1.5),
        "nearest_opp_dist": rng.uniform(0.1, 1.0),
        "teammates_pos": [(rng.uniform(-1, 1), rng.uniform(-0.42, 0.42)) for _ in range(n_left - 1)],
        "opponents_pos": [(rng.uniform(-1, 1), rng.uniform(-0.42, 0.42)) for _ in range(n_right)],
        "game_mode": 0,
        "score_diff": 0,
        "steps_left": 1500,
    }

    for role in ["default", "ball_carrier", "support_wide", "midfield_support"]:
        try:
            v = phi(test_state, role)
            if not isinstance(v, (int, float)):
                print(f"  phi returned non-numeric for role={role}: {type(v)}")
                return False
            if not (0.0 <= float(v) <= 1.0):
                print(f"  phi out of [0,1] for role={role}: {v}")
                return False
        except Exception as e:
            print(f"  phi raised exception for role={role}: {e}")
            return False

    return True


# ---------------------------------------------------------------------------
# CLI: revise
# ---------------------------------------------------------------------------
def cmd_revise(args):
    """Single revision step."""
    current_code = Path(args.current_phi).read_text(encoding="utf-8")
    metrics_text = Path(args.metrics).read_text(encoding="utf-8")

    print(f"Calling LLM to diagnose and revise phi (iteration {args.iteration})...")
    diagnosis, new_code, raw = revise_phi(
        current_phi_code=current_code,
        metrics_json=metrics_text,
        iteration=args.iteration,
        diagnosis_request=args.hint or "",
        scenario_name=args.scenario_name,
        temperature=args.temperature,
    )

    print("\n=== LLM DIAGNOSIS ===")
    print(diagnosis or "(no structured diagnosis extracted)")
    print("=====================\n")

    print("Validating revised phi...")
    if validate_phi_code(new_code):
        print("  Validation PASSED")
    else:
        print("  Validation FAILED — writing anyway (manual inspection needed)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = _file_header(args.iteration)
    out.write_text(header + new_code, encoding="utf-8")
    print(f"Revised phi written to {out}")

    # save diagnosis alongside
    diag_path = out.with_suffix(".diagnosis.txt")
    diag_path.write_text(
        f"Iteration: {args.iteration}\nMetrics: {args.metrics}\n\n{diagnosis}",
        encoding="utf-8",
    )
    print(f"Diagnosis saved to {diag_path}")


# ---------------------------------------------------------------------------
# CLI: loop
# ---------------------------------------------------------------------------
def cmd_loop(args):
    """Full iterative Eureka loop: eval → extract → diagnose → revise."""
    import subprocess

    phi_path = Path(args.phi_dir) / "phi_llm.py"
    if not phi_path.exists():
        print(f"Error: Initial phi not found at {phi_path}")
        sys.exit(1)

    for it in range(1, args.num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"  ITERATIVE EUREKA — Iteration {it}/{args.num_iterations}")
        print(f"{'='*60}\n")

        # Step 1: Extract behavioral metrics
        metrics_path = Path("metrics") / f"behavior_iter{it}.json"
        print(f"[Step 1] Extracting behavioral metrics → {metrics_path}")
        extract_cmd = [
            sys.executable, "scripts/extract_behavior_metrics.py",
            "--checkpoint", args.checkpoint,
            "--config", args.config,
            "--num-games", str(args.num_eval_games),
            "--num-workers", str(args.num_eval_workers),
            "--output", str(metrics_path),
            "--phi-module", f"generated_phi.phi_llm",
        ]
        result = subprocess.run(extract_cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  Metric extraction failed (exit {result.returncode})")
            break

        # Step 2: Revise phi
        new_phi_path = Path(args.phi_dir) / f"phi_llm_v{it + 1}.py"
        print(f"\n[Step 2] Revising phi → {new_phi_path}")
        current_code = phi_path.read_text(encoding="utf-8")
        metrics_text = metrics_path.read_text(encoding="utf-8")

        diagnosis, new_code, raw = revise_phi(
            current_phi_code=current_code,
            metrics_json=metrics_text,
            iteration=it,
            diagnosis_request=args.hint or "",
            scenario_name=args.scenario_name,
            temperature=args.temperature,
        )

        print("\n--- LLM Diagnosis ---")
        print(diagnosis or "(no structured diagnosis)")
        print("---------------------\n")

        if validate_phi_code(new_code):
            print("  Phi validation PASSED")
        else:
            print("  Phi validation FAILED — writing anyway")

        new_phi_path.parent.mkdir(parents=True, exist_ok=True)
        new_phi_path.write_text(_file_header(it) + new_code, encoding="utf-8")

        # Save diagnosis
        diag_path = new_phi_path.with_suffix(".diagnosis.txt")
        diag_path.write_text(
            f"Iteration: {it}\nMetrics: {metrics_path}\n\n{diagnosis}",
            encoding="utf-8",
        )

        # Update symlink/copy so next iteration uses the new phi
        # (overwrite phi_llm.py so the training config picks it up)
        active_phi = Path(args.phi_dir) / "phi_llm.py"
        active_phi.write_text(
            _file_header(it) + f"# Sourced from iteration {it}\n" + new_code,
            encoding="utf-8",
        )
        print(f"  Active phi updated: {active_phi}")

        print(f"\nIteration {it} complete. New phi ready for training.")
        if it < args.num_iterations:
            print("  NOTE: To get metrics for the NEXT iteration, "
                  "you need to train with the new phi first, then re-run.")
            if not args.auto_train:
                print("  Stopping here. Use --auto-train to submit training jobs automatically.")
                break

    print(f"\nIterative Eureka finished ({it} iteration(s) completed).")


# ---------------------------------------------------------------------------
# CLI: slurm-loop — fully automated SLURM job chain
# ---------------------------------------------------------------------------
def cmd_slurm_loop(args):
    """Submit a full SLURM job chain: [train × N] → [eval+revise] → [train × N] → ... repeated."""
    import os
    import subprocess

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: config not found: {config_path}")
        sys.exit(1)
    if not os.path.exists("train.slurm"):
        print("Error: train.slurm not found in project root")
        sys.exit(1)
    if not os.path.exists("eval_revise.slurm"):
        print("Error: eval_revise.slurm not found in project root")
        sys.exit(1)

    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        slurm_cfg = cfg.get("slurm_config", {})
    except ImportError:
        print("Warning: pyyaml not installed, using default SLURM config")
        slurm_cfg = {}

    train_hours = int(slurm_cfg.get("time_hours", 1))
    train_time = f"{train_hours}:00:00"
    train_partition = slurm_cfg.get("partition", "sharing")
    train_gpus = slurm_cfg.get("num_gpus", 1)
    train_cpus = slurm_cfg.get("cpus_per_task", 16)
    train_mem = slurm_cfg.get("memory_gb", 128)
    gpu_type = slurm_cfg.get("gpu_type")

    if gpu_type:
        gpu_str = f"gpu:{gpu_type}:{train_gpus}"
    else:
        gpu_str = f"gpu:{train_gpus}"

    base_name = args.job_name or "eureka"
    all_job_ids = []
    prev_job_id = None

    start_with_eval = getattr(args, "start_with_eval", False)

    if start_with_eval:
        total_jobs = args.num_iterations * (args.train_jobs_per_iter + 1)
        flow_desc = "[eval+revise] → [train×N] → repeat"
    else:
        total_jobs = args.num_iterations * (args.train_jobs_per_iter + 1)
        flow_desc = "[train×N] → [eval+revise] → repeat"

    print("=" * 70)
    print(f"  ITERATIVE EUREKA — SLURM Loop")
    print(f"  Config:        {config_path}")
    print(f"  Iterations:    {args.num_iterations}")
    print(f"  Train jobs/iter: {args.train_jobs_per_iter}")
    print(f"  Eval games:    {args.num_eval_games}")
    print(f"  Start with:    {'eval+revise' if start_with_eval else 'training'}")
    print(f"  Flow:          {flow_desc}")
    print(f"  Total SLURM jobs: ~{total_jobs}")
    print("=" * 70)

    conda_env = os.environ.get("CONDA_ENV_NAME")

    def _sbatch(cmd: list[str]) -> str | None:
        """Submit or dry-run an sbatch command. Returns job id or simulated id."""
        if conda_env:
            cmd.insert(1, f"--export=ALL,CONDA_ENV_NAME={conda_env}")
        print(f"  $ {' '.join(cmd)}")
        if args.no_submit:
            fake_id = f"SIM_{len(all_job_ids)+1}"
            all_job_ids.append(fake_id)
            return fake_id
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        jid = result.stdout.strip().split()[-1]
        all_job_ids.append(jid)
        return jid

    def _submit_eval_revise(iteration: int):
        nonlocal prev_job_id
        eval_label = f"{base_name}_iter{iteration}_eval_revise"
        eval_cmd = [
            "sbatch",
            f"--job-name={eval_label}",
            f"--gres=gpu:1",
            f"--cpus-per-task={min(train_cpus, 16)}",
            f"--mem=64G",
            f"--time=1:00:00",
            f"--partition={train_partition}",
        ]
        if prev_job_id:
            eval_cmd.append(f"--dependency=afterany:{prev_job_id}")

        eval_cmd.append("eval_revise.slurm")
        eval_cmd.extend([
            "--config", config_path,
            "--iteration", str(iteration),
            "--num-eval-games", str(args.num_eval_games),
            "--num-eval-workers", str(args.num_eval_workers),
            "--phi-dir", args.phi_dir,
            "--scenario-name", args.scenario_name,
            "--temperature", str(args.temperature),
        ])
        if args.hint:
            eval_cmd.extend(["--hint", args.hint])

        prev_job_id = _sbatch(eval_cmd)
        print(f"    → Eval+Revise: {prev_job_id}")

    def _submit_train_segment(iteration: int):
        nonlocal prev_job_id
        for tj in range(1, args.train_jobs_per_iter + 1):
            job_label = f"{base_name}_iter{iteration}_train{tj}"
            cmd = [
                "sbatch",
                f"--job-name={job_label}",
                f"--gres={gpu_str}",
                f"--cpus-per-task={train_cpus}",
                f"--mem={train_mem}G",
                f"--time={train_time}",
                f"--partition={train_partition}",
            ]
            if prev_job_id:
                cmd.append(f"--dependency=afterany:{prev_job_id}")

            cmd.append("train.slurm")
            cmd.extend(["--config", config_path, "--auto-resume"])

            prev_job_id = _sbatch(cmd)
            print(f"    → Train job {tj}/{args.train_jobs_per_iter}: {prev_job_id}")

    for it in range(1, args.num_iterations + 1):
        print(f"\n--- Iteration {it}/{args.num_iterations} ---")

        if start_with_eval:
            _submit_eval_revise(it)
            _submit_train_segment(it)
        else:
            _submit_train_segment(it)
            _submit_eval_revise(it)

    # Summary
    print("\n" + "=" * 70)
    print("SLURM Loop Submission Complete")
    print("=" * 70)
    total = len(all_job_ids)
    print(f"Total jobs submitted: {total}")
    if not args.no_submit:
        print(f"Job IDs: {', '.join(all_job_ids)}")
        print(f"\nMonitor: squeue -u $USER")
        print(f"Cancel all: scancel {all_job_ids[0]}  (or scancel {{{all_job_ids[0]}..{all_job_ids[-1]}}})")
    else:
        print("(dry-run mode — no jobs were actually submitted)")

    order = ("eval+revise → train" if start_with_eval else "train → eval+revise")
    print(f"\nWorkflow per iteration ({order}):")
    if start_with_eval:
        print(f"  1. Eval {args.num_eval_games} games → extract metrics (from existing checkpoint)")
        print(f"  2. LLM diagnoses + revises phi → overwrites {args.phi_dir}/phi_llm.py")
        print(f"  3. Train ({args.train_jobs_per_iter} jobs × {train_hours}h each) with updated phi")
    else:
        print(f"  1. Train ({args.train_jobs_per_iter} jobs × {train_hours}h each)")
        print(f"  2. Eval {args.num_eval_games} games → extract metrics")
        print(f"  3. LLM diagnoses + revises phi → overwrites {args.phi_dir}/phi_llm.py")
        print(f"  4. Next iteration trains with updated phi")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Iterative Eureka: LLM-based phi revision loop"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- revise ---
    rev = subparsers.add_parser("revise", help="Single phi revision from metrics")
    rev.add_argument("--metrics", required=True, help="Behavior metrics JSON file")
    rev.add_argument("--current-phi", required=True, help="Current phi .py file")
    rev.add_argument("--output", "-o", required=True, help="Output revised phi .py file")
    rev.add_argument("--iteration", type=int, default=1, help="Iteration number (for provenance)")
    rev.add_argument("--hint", type=str, default="", help="Extra diagnosis request for LLM")
    rev.add_argument("--scenario-name", default="GRF 11v11 full game")
    rev.add_argument("--temperature", type=float, default=0.3)

    # --- loop (local) ---
    lp = subparsers.add_parser("loop", help="Full eval→diagnose→revise loop (local, no training)")
    lp.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    lp.add_argument("--config", required=True, help="Training YAML config")
    lp.add_argument("--num-iterations", type=int, default=3)
    lp.add_argument("--num-eval-games", type=int, default=20)
    lp.add_argument("--num-eval-workers", type=int, default=4)
    lp.add_argument("--phi-dir", default="generated_phi", help="Directory containing phi_llm.py")
    lp.add_argument("--hint", type=str, default="")
    lp.add_argument("--scenario-name", default="GRF 11v11 full game")
    lp.add_argument("--temperature", type=float, default=0.3)
    lp.add_argument("--auto-train", action="store_true",
                     help="Automatically submit training jobs between iterations (not yet implemented)")

    # --- slurm-loop (SLURM cluster, full automation) ---
    sl = subparsers.add_parser(
        "slurm-loop",
        help="Submit fully automated SLURM chain: [train×N] → [eval+revise] → repeat"
    )
    sl.add_argument("--config", required=True, help="Training YAML config (must have slurm_config)")
    sl.add_argument("--num-iterations", type=int, default=3,
                    help="Number of Eureka iterations (each = train + eval + revise)")
    sl.add_argument("--train-jobs-per-iter", type=int, default=5,
                    help="Number of chained training SLURM jobs per iteration "
                         "(e.g. 5 × 1h on sharing = 5h of training)")
    sl.add_argument("--num-eval-games", type=int, default=20)
    sl.add_argument("--num-eval-workers", type=int, default=4)
    sl.add_argument("--phi-dir", default="generated_phi")
    sl.add_argument("--job-name", default=None, help="Base SLURM job name prefix")
    sl.add_argument("--hint", type=str, default="")
    sl.add_argument("--scenario-name", default="GRF 11v11 full game")
    sl.add_argument("--temperature", type=float, default=0.3)
    sl.add_argument("--start-with-eval", action="store_true",
                    help="Start each iteration with eval+revise instead of training "
                         "(useful when checkpoints already exist)")
    sl.add_argument("--no-submit", action="store_true",
                    help="Dry run: print commands without submitting")

    args = parser.parse_args()
    if args.command == "revise":
        cmd_revise(args)
    elif args.command == "loop":
        cmd_loop(args)
    elif args.command == "slurm-loop":
        cmd_slurm_loop(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
