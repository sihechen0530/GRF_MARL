#!/usr/bin/env python3
"""Submit one CPU SLURM render job per baseline/LLM experiment."""

import argparse
import csv
import os
import pathlib
import re
import subprocess
from typing import Dict, Iterable, List, Optional

import yaml

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATE_PREFIX = "20260419"
OUTPUT_ROOT = pathlib.Path("results") / f"{DATE_PREFIX}_rendered_win_episodes"


def has_checkpoint(run_dir: pathlib.Path) -> bool:
    return any((p / "desc.pkl").exists() for p in (run_dir / "agent_0").glob("*/*"))


def find_config_from_logs(run_dir: pathlib.Path) -> Optional[pathlib.Path]:
    for log_path in sorted(run_dir.glob("slurm_*.log")):
        try:
            text = log_path.read_text(errors="ignore")
        except OSError:
            continue
        match = re.search(r"Config:\s+(\S+\.ya?ml)", text)
        if not match:
            match = re.search(r"--config\s+(\S+\.ya?ml)", text)
        if match:
            path = pathlib.Path(match.group(1))
            if not path.is_absolute():
                path = BASE_DIR / path
            if path.exists():
                return path
    direct = run_dir / "config.yaml"
    if direct.exists():
        return direct
    return None


def config_for_expr(expr_name: str, run_dir: pathlib.Path) -> Optional[pathlib.Path]:
    direct = run_dir / "config.yaml"
    if direct.exists():
        return direct
    from_logs = find_config_from_logs(run_dir)
    if from_logs:
        return from_logs

    algorithm = expr_name.rsplit("_", 1)[-1]
    if expr_name.startswith("llm_mask_fix_full_game_11_vs_11_hard_"):
        return BASE_DIR / "expr_configs" / "llm_action_masking" / "full_game" / "11_vs_11_hard" / f"{algorithm}_fix_per_player.yaml"
    if expr_name == "llm_mask_fix_academy_counterattack_mappo_soft":
        return BASE_DIR / "expr_configs" / "llm_action_masking" / "counterattack" / "mappo_fix_per_player.yaml"
    if expr_name == "llm_mask_fix_academy_counterattack_mappo":
        return BASE_DIR / "expr_configs" / "llm_action_masking" / "counterattack" / "mappo_fix_per_player.yaml"
    if expr_name.startswith("llm_mask_full_game_11_vs_11_hard_"):
        return BASE_DIR / "expr_configs" / "llm_action_masking" / "full_game" / "11_vs_11_hard" / f"{algorithm}.yaml"
    if expr_name.startswith("llm_mask_academy_corner_"):
        return BASE_DIR / "expr_configs" / "llm_action_masking" / "corner" / f"{algorithm}.yaml"
    if expr_name.startswith("llm_mask_academy_counterattack_easy_"):
        return BASE_DIR / "expr_configs" / "llm_action_masking" / "counterattack_easy" / f"{algorithm}.yaml"
    if expr_name.startswith("llm_mask_academy_counterattack_"):
        return BASE_DIR / "expr_configs" / "llm_action_masking" / "counterattack" / f"{algorithm}.yaml"

    if expr_name.startswith("benchmark_academy_corner_"):
        return BASE_DIR / "expr_configs" / "cooperative_MARL_benchmark" / "academy" / "corner" / f"{algorithm}.yaml"
    if expr_name.startswith("benchmark_academy_counterattack_easy_"):
        return BASE_DIR / "expr_configs" / "cooperative_MARL_benchmark" / "academy" / "counterattack_easy" / f"{algorithm}.yaml"
    if expr_name.startswith("benchmark_academy_counterattack_"):
        return BASE_DIR / "expr_configs" / "cooperative_MARL_benchmark" / "academy" / "counterattack" / f"{algorithm}.yaml"
    if expr_name.startswith("benchmark_full_game_11_v_11_hard_"):
        return BASE_DIR / "expr_configs" / "cooperative_MARL_benchmark" / "full_game" / "11_vs_11_hard" / f"{algorithm}.yaml"
    if expr_name.startswith("full_game_11_v_11_hard_"):
        return BASE_DIR / "expr_configs" / "cooperative_MARL_benchmark" / "full_game" / "11_vs_11_hard" / f"{algorithm}.yaml"
    if expr_name.startswith("full_game_10_vs_10_hard_"):
        return BASE_DIR / "expr_configs" / "cooperative_MARL_benchmark" / "full_game" / "11_vs_11_hard" / f"{algorithm}.yaml"
    return None


def opponent_from_config(config_path: pathlib.Path) -> pathlib.Path:
    cfg = yaml.safe_load(config_path.read_text())
    init_cfg = cfg["populations"][0]["algorithm"]["policy_init_cfg"]
    agent_1 = init_cfg.get("agent_1", {})
    for item in agent_1.get("init_cfg", []) or []:
        policy_dir = item.get("policy_dir")
        if policy_dir:
            return BASE_DIR / policy_dir
    env_name = str(
        cfg["rollout_manager"]["worker"]["envs"][0]["scenario_config"]["env_name"]
    )
    if "11_vs_11" in env_name or "10_vs_10" in env_name:
        return BASE_DIR / "light_malib" / "trained_models" / "gr_football" / "11_vs_11" / "built_in"
    return BASE_DIR / "light_malib" / "trained_models" / "gr_football" / "5_vs_5" / "built_in"


def best_run_dir(exp_dir: pathlib.Path) -> Optional[pathlib.Path]:
    runs = [p.resolve() for p in exp_dir.iterdir() if p.is_dir()]
    runs = [p for p in runs if has_checkpoint(p)]
    if not runs:
        return None

    def score(run_dir: pathlib.Path):
        eval_csv = run_dir / "eval_results.csv"
        if not eval_csv.exists():
            return (-1.0, run_dir.name)
        with eval_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return (-1.0, run_dir.name)
        return (max(float(row.get("win_rate") or 0.0) for row in rows), run_dir.name)

    return max(runs, key=score)


def iter_experiments() -> Iterable[Dict[str, pathlib.Path]]:
    exps_root = BASE_DIR / "exps" / "gr_football"
    logs_root = BASE_DIR / "logs" / "gr_football"

    for exp_dir in sorted(p for p in exps_root.iterdir() if p.is_dir() and not p.name.startswith("llm")):
        run_dir = best_run_dir(exp_dir)
        if run_dir:
            yield {"group": "baseline", "expr": exp_dir.name, "run_dir": run_dir}

    for exp_dir in sorted(p for p in logs_root.iterdir() if p.is_dir() and p.name.startswith("llm")):
        run_dir = best_run_dir(exp_dir)
        if run_dir:
            yield {"group": "llm", "expr": exp_dir.name, "run_dir": run_dir}


def rel(path: pathlib.Path) -> str:
    return str(path.resolve().relative_to(BASE_DIR))


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)[-80:]


def build_job(exp: Dict[str, pathlib.Path], args) -> Optional[Dict]:
    config = config_for_expr(exp["expr"], exp["run_dir"])
    if config is None or not config.exists():
        return {"skip": "missing config", **exp}
    opponent = opponent_from_config(config)
    if not (opponent / "desc.pkl").exists():
        return {"skip": f"missing opponent {opponent}", **exp, "config": config}
    output_dir = OUTPUT_ROOT / exp["group"] / exp["expr"] / exp["run_dir"].name
    command = [
        "sbatch",
        "--job-name",
        f"rw_{sanitize(exp['expr'])}",
        "render_win_cpu.slurm",
        "--config",
        rel(config),
        "--run-dir",
        rel(exp["run_dir"]),
        "--opponent",
        rel(opponent),
        "--output-dir",
        str(output_dir),
        "--num-wins",
        str(args.num_wins),
        "--max-episodes",
        str(args.max_episodes),
        "--fps",
        str(args.fps),
        "--seed-base",
        str(args.seed_base),
    ]
    return {**exp, "config": config, "opponent": opponent, "output_dir": output_dir, "command": command}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-wins", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20260419)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    jobs = []
    skips = []
    for exp in iter_experiments():
        job = build_job(exp, args)
        if job and "skip" in job:
            skips.append(job)
        elif job:
            jobs.append(job)
    if args.skip_first:
        jobs = jobs[args.skip_first :]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    (BASE_DIR / OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Prepared jobs: {len(jobs)}")
    if skips:
        print(f"Skipped: {len(skips)}")
        for skip in skips:
            print(f"SKIP {skip['group']}/{skip['expr']}: {skip['skip']}")

    for job in jobs:
        print(" ".join(job["command"]))
        if not args.dry_run:
            result = subprocess.run(job["command"], cwd=BASE_DIR, text=True, capture_output=True)
            if result.returncode != 0:
                print(f"FAILED {job['group']}/{job['expr']}: {result.stderr.strip()}")
                raise SystemExit(result.returncode)
            print(result.stdout.strip())


if __name__ == "__main__":
    main()
