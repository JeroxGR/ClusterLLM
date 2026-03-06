"""
Unified LLM client for ClusterLLM.
Supports: Groq (Llama, DeepSeek, Qwen, etc.), OpenAI, DeepSeek, Qwen, Together AI.

All providers use OpenAI-compatible APIs, so we use the openai SDK with different base_url.
Configure via environment variables or pass directly.

Usage:
    from llm_client import get_client, get_model_name, delayed_completion

Environment Variables:
    LLM_PROVIDER     - one of: groq, openai, deepseek, qwen, llama (default: groq)
    LLM_API_KEY      - API key for the chosen provider
    LLM_BASE_URL     - (optional) override the base URL
    LLM_MODEL        - (optional) override the model name

Provider defaults:
    groq:     model=llama-3.3-70b-versatile, base_url=https://api.groq.com/openai/v1
    openai:   model=gpt-4o-mini,             base_url=https://api.openai.com/v1
    deepseek: model=deepseek-chat,           base_url=https://api.deepseek.com
    qwen:     model=qwen-plus,               base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
    llama:    model=meta-llama/Llama-3.1-8B-Instruct-Turbo, base_url=https://api.together.xyz/v1

Groq available models (set LLM_MODEL to switch):
    llama-3.3-70b-versatile        (Llama 3.3 70B)
    deepseek-r1-distill-llama-70b  (DeepSeek R1 70B)
    qwen-qwq-32b                   (Qwen QwQ 32B)
    llama-3.1-8b-instant           (Llama 3.1 8B - fastest)
"""

import os
import time
from openai import OpenAI

# Auto-load .env file from project root
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

PROVIDER_DEFAULTS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "env_key": "QWEN_API_KEY",
    },
    "llama": {
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "env_key": "TOGETHER_API_KEY",
    },
}

_client = None


def get_provider():
    return os.getenv("LLM_PROVIDER", "groq").lower()


def get_client():
    global _client
    if _client is not None:
        return _client

    provider = get_provider()
    defaults = PROVIDER_DEFAULTS.get(provider)
    if defaults is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDER_DEFAULTS.keys())}")

    api_key = os.getenv("LLM_API_KEY") or os.getenv(defaults["env_key"])
    if not api_key:
        raise ValueError(
            f"No API key found. Set LLM_API_KEY or {defaults['env_key']} environment variable."
        )

    base_url = os.getenv("LLM_BASE_URL", defaults["base_url"])

    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def get_model_name():
    provider = get_provider()
    defaults = PROVIDER_DEFAULTS[provider]
    return os.getenv("LLM_MODEL", defaults["model"])


def delayed_completion(delay_in_seconds=1, max_trials=1, **kwargs):
    """Call the chat completions API with retry logic and exponential backoff for rate limits."""
    time.sleep(delay_in_seconds)

    client = get_client()
    output, error = None, None
    for attempt in range(max_trials):
        try:
            output = client.chat.completions.create(**kwargs)
            break
        except Exception as e:
            error = e
            # Exponential backoff: 2s, 4s, 8s, 16s, 32s...
            backoff = min(delay_in_seconds * (2 ** (attempt + 1)), 60)
            time.sleep(backoff)
    return output, error


def extract_content(completion):
    """Extract text content from a completion response."""
    return completion.choices[0].message.content.strip()
