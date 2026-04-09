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

from light_malib.llm import (
    chat_completions_text,
    ensure_phi_export,
    extract_python_code_block,
)
from light_malib.llm.prompts_phi import (
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

    # --- loop ---
    lp = subparsers.add_parser("loop", help="Full eval→diagnose→revise loop")
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

    args = parser.parse_args()
    if args.command == "revise":
        cmd_revise(args)
    elif args.command == "loop":
        cmd_loop(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
