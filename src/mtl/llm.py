"""Centralized LLM configuration for the MTL pipeline.

All retrieval, extraction, and rerank modules read their API settings from here
instead of hardcoding DeepSeek URLs or model names.

Configuration sources (highest priority first):
  1. Programmatic: `mtl.llm.configure(api_key=..., base_url=..., model=...)`
  2. Environment: `MTL_LLM_API_KEY`, `MTL_LLM_BASE_URL`, `MTL_LLM_MODEL`
  3. Fallback: `DEEPSEEK_API_KEY`, `https://api.deepseek.com`, `deepseek-chat`

Usage:
    from mtl.llm import get_client, get_model, configure

    # Default (env vars or DeepSeek fallback)
    client = get_client()

    # Override programmatically
    configure(api_key="sk-xxx", base_url="https://api.openai.com", model="gpt-4o")
    client = get_client()
    model = get_model()
"""
import os
import threading
from openai import OpenAI

# Module-level state
_config: dict = {}
_client: OpenAI | None = None
_client_lock = threading.Lock()
_configured = False


def configure(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> None:
    """Set LLM configuration programmatically. Overrides env vars.

    Call this once at pipeline startup. All downstream modules
    (retriever, rerank, synergy, extraction) will use these settings.

    Also sets DEEPSEEK_API_KEY env var for backward compatibility with
    original harbor/experiments extraction code.

    Args:
        api_key: API key. Defaults to MTL_LLM_API_KEY or DEEPSEEK_API_KEY env var.
        base_url: API base URL. Defaults to MTL_LLM_BASE_URL env var or https://api.deepseek.com.
        model: Model name. Defaults to MTL_LLM_MODEL env var or deepseek-chat.
    """
    global _config, _client, _configured

    _config["api_key"] = api_key or os.getenv("MTL_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    _config["base_url"] = base_url or os.getenv("MTL_LLM_BASE_URL") or "https://api.deepseek.com"
    _config["model"] = model or os.getenv("MTL_LLM_MODEL") or "deepseek-chat"

    # Propagate to env vars so original harbor/experiments code picks them up
    if _config["api_key"]:
        os.environ["DEEPSEEK_API_KEY"] = _config["api_key"]
        os.environ["MTL_LLM_API_KEY"] = _config["api_key"]
    if _config["base_url"]:
        os.environ["MTL_LLM_BASE_URL"] = _config["base_url"]
    if _config["model"]:
        os.environ["MTL_LLM_MODEL"] = _config["model"]

    # Reset cached client so it picks up new config
    with _client_lock:
        _client = None
    _configured = True


def _ensure_configured():
    """Lazy-init with defaults if configure() was never called."""
    if not _configured:
        configure()


def get_api_key() -> str:
    """Return the configured API key."""
    _ensure_configured()
    return _config.get("api_key", "")


def get_base_url() -> str:
    """Return the configured API base URL."""
    _ensure_configured()
    return _config.get("base_url", "https://api.deepseek.com")


def get_model() -> str:
    """Return the configured model name."""
    _ensure_configured()
    return _config.get("model", "deepseek-chat")


def get_client() -> OpenAI:
    """Return a cached OpenAI client configured with current settings (thread-safe)."""
    global _client
    _ensure_configured()
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = OpenAI(
                    api_key=_config["api_key"],
                    base_url=_config["base_url"],
                )
    return _client


def reset() -> None:
    """Reset all configuration (useful for testing)."""
    global _config, _client, _configured
    _config = {}
    _client = None
    _configured = False


def describe() -> dict:
    """Return current config for display."""
    _ensure_configured()
    key = _config.get("api_key") or ""
    return {
        "base_url": _config.get("base_url", ""),
        "model": _config.get("model", ""),
        "api_key": key[:8] + "..." + key[-4:] if len(key) > 12 else "(not set)",
    }
