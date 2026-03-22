# Copyright 2025 GRF_MARL contributors
"""HTTP JSON client for OpenAI-compatible chat completions (reusable for masking / RL experiments)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

from light_malib.llm.config import LLMClientConfig


def chat_completions(
    messages: Sequence[Dict[str, str]],
    *,
    config: Optional[LLMClientConfig] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    POST {base_url}/chat/completions with Bearer auth.

    Returns the full JSON body (choices, usage, id, ...).
    """
    cfg = config or LLMClientConfig.from_env()
    url = f"{cfg.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": cfg.model,
        "messages": list(messages),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if extra_body:
        body.update(extra_body)
    resp = requests.post(
        url,
        headers=headers,
        data=json.dumps(body),
        timeout=cfg.timeout_s,
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"chat_completions HTTP {resp.status_code}: {resp.text[:2000]}"
        )
    return resp.json()


def assistant_text(response: Dict[str, Any]) -> str:
    """Extract assistant message content from a chat_completions JSON body."""
    choices = response.get("choices") or []
    if not choices:
        raise ValueError(f"no choices in response: {response!r}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        raise ValueError(f"missing message.content: {response!r}")
    if isinstance(content, list):
        # Some APIs return structured content blocks
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def chat_completions_text(
    messages: Sequence[Dict[str, str]],
    *,
    config: Optional[LLMClientConfig] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience: return assistant string only."""
    data = chat_completions(
        messages,
        config=config,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return assistant_text(data)
