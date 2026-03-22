#!/usr/bin/env python3
"""
Call DeepSeek (or any OpenAI-compatible API) to generate a phi module, write to a .py file.

Usage:
  export LLM_API_KEY=sk-...
  python scripts/generate_phi_llm.py --output generated_phi/phi_llm_v1.py

Training: set yaml potential_shaping.phi_module to the import path for that file
(e.g. put package under PYTHONPATH or use a path that is importable).

Example import path if file is at repo root package:
  generated_phi.phi_llm_v1  -> requires generated_phi/__init__.py or run with PYTHONPATH=.
"""
from __future__ import annotations

import argparse
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
from light_malib.llm.prompts_phi import messages_as_openai_json, build_phi_messages


def _file_header_note() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"# Auto-generated potential Φ module (LLM). UTC: {ts}\n"
        "# Validate with: python scripts/validate_phi_poc.py --phi-module <import_path>\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate phi(state,role) via OpenAI-compatible API")
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_REPO_ROOT / "generated_phi" / "phi_llm.py",
        help="Output .py path (default: generated_phi/phi_llm.py)",
    )
    ap.add_argument(
        "--task-hint",
        type=str,
        default="",
        help="Extra user instructions (e.g. counterattack scenario, emphasis on through balls).",
    )
    ap.add_argument(
        "--scenario-name",
        type=str,
        default="GRF cooperative MARL (academy or full game)",
    )
    ap.add_argument(
        "--left-roles-example",
        type=str,
        default="ball_carrier, left_winger, right_winger, trailing_mid, default",
    )
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts only; do not call API or write file.",
    )
    ap.add_argument(
        "--no-header",
        action="store_true",
        help="Do not prepend auto-generation comment header.",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_phi on the generated module after write (requires numpy).",
    )
    args = ap.parse_args()

    pairs = build_phi_messages(
        extra_task_hint=args.task_hint,
        left_roles_example=args.left_roles_example,
        scenario_name=args.scenario_name,
    )
    messages = messages_as_openai_json(pairs)

    if args.dry_run:
        for m in messages:
            print(f"=== {m['role']} ===\n{m['content']}\n")
        return

    raw = chat_completions_text(
        messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    code = ensure_phi_export(extract_python_code_block(raw))
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    text = ("" if args.no_header else _file_header_note()) + code
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out.resolve()} ({len(text)} chars)")

    if args.validate:
        import numpy as np

        from light_malib.envs.gr_football.potential_shaping import validate_phi

        # Load from file path: importlib by path
        import importlib.util

        spec = importlib.util.spec_from_file_location("phi_llm_generated", out)
        if spec is None or spec.loader is None:
            raise RuntimeError("importlib spec failed")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        phi = getattr(mod, "phi")
        validate_phi(phi, np.random.default_rng(0), n_left=4, n_right=2, n_samples=128)
        print("validate_phi: OK")


if __name__ == "__main__":
    main()
