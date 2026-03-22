#!/usr/bin/env python3
"""
Minimal check that OpenAI-compatible chat API (e.g. DeepSeek) works from this repo.

  export LLM_API_KEY=sk-...   # or DEEPSEEK_API_KEY
  python scripts/smoke_test_llm.py

Optional:
  LLM_BASE_URL   default https://api.deepseek.com/v1
  LLM_MODEL      default deepseek-chat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test LLM HTTP (OpenAI-compatible)")
    ap.add_argument(
        "--prompt",
        type=str,
        default="Reply with exactly: OK",
        help="User message (keep short to save tokens)",
    )
    args = ap.parse_args()

    try:
        from light_malib.llm import LLMClientConfig, chat_completions_text
    except ImportError as e:
        print("Import failed:", e)
        print("Use the same interpreter you train with, e.g. python3 -m pip install -r requirements.txt")
        sys.exit(2)

    try:
        cfg = LLMClientConfig.from_env()
    except ValueError as e:
        print(e)
        print("Set LLM_API_KEY or DEEPSEEK_API_KEY in the environment.")
        sys.exit(1)

    print(f"base_url={cfg.base_url} model={cfg.model} timeout={cfg.timeout_s}s")
    text = chat_completions_text(
        [{"role": "user", "content": args.prompt}],
        temperature=0.0,
        max_tokens=64,
    )
    text = text.strip()
    print("assistant:", repr(text[:500]))
    print("smoke_test_llm: OK")


if __name__ == "__main__":
    main()
