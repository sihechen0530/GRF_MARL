# Copyright 2025 GRF_MARL contributors
"""
Reusable OpenAI-compatible HTTP helpers for LLM experiments (DeepSeek, OpenAI, local vLLM).

Other features (e.g. action masking) can import the same client and swap prompts only.

Environment (typical DeepSeek):
  LLM_API_KEY or DEEPSEEK_API_KEY — required
  LLM_BASE_URL — default https://api.deepseek.com/v1
  LLM_MODEL — default deepseek-chat
  LLM_TIMEOUT_S — default 120
"""
from __future__ import annotations

from light_malib.llm.code_parsing import ensure_phi_export, extract_python_code_block
from light_malib.llm.config import LLMClientConfig
from light_malib.llm.openai_compatible import (
    assistant_text,
    chat_completions,
    chat_completions_text,
)
from light_malib.llm.entropy_mask import EntropyGuidedMask

__all__ = [
    "LLMClientConfig",
    "assistant_text",
    "chat_completions",
    "chat_completions_text",
    "extract_python_code_block",
    "ensure_phi_export",
    "EntropyGuidedMask",
]
