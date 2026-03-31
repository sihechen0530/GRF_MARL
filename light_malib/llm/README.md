# `light_malib.llm` — reusable OpenAI-compatible HTTP

Use this for **any** experiment that needs `POST /v1/chat/completions` (DeepSeek, OpenAI, many proxies).

## Environment

| Variable | Meaning |
|----------|---------|
| `LLM_API_KEY` | Preferred; Bearer token |
| `DEEPSEEK_API_KEY` | Fallback if `LLM_API_KEY` unset |
| `LLM_BASE_URL` | Default `https://api.deepseek.com/v1` |
| `LLM_MODEL` | Default `deepseek-chat` |
| `LLM_TIMEOUT_S` | Default `120` |

## Python API

```python
from light_malib.llm import LLMClientConfig, chat_completions_text

text = chat_completions_text(
    [{"role": "user", "content": "Say hi in one word."}],
    temperature=0.0,
)
```

For full JSON (usage, logprobs, etc.) use `chat_completions` and `assistant_text`.

## Φ generation

See `scripts/generate_phi_llm.py` and `light_malib.llm.prompts_phi`.

Action-masking or other tasks: copy the same HTTP layer; add new `prompts_*.py` modules without changing `openai_compatible.py`.
