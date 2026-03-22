from __future__ import annotations

import os
from typing import Any

from pepole.providers.anthropic import AnthropicClient
from pepole.providers.base import LLMClient
from pepole.providers.dry_run import DryRunClient
from pepole.providers.google_gemini import GeminiClient
from pepole.providers.openai_compat import OpenAICompatClient


def parse_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        return "openai", spec
    provider, model = spec.split(":", 1)
    return provider.strip().lower(), model.strip()


def _has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _has_google_key() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def get_client(spec: str, *, allow_dry: bool = True) -> LLMClient:
    """
    spec 形如:
      openai:gpt-4o
      anthropic:claude-sonnet-4-20250514
      google:gemini-2.0-flash
    """
    provider, model = parse_spec(spec)
    if provider == "openai":
        if not _has_openai_key():
            if allow_dry:
                return DryRunClient("openai-dry")
            raise RuntimeError("OPENAI_API_KEY 未设置")
        return OpenAICompatClient(model)
    if provider == "anthropic":
        if not _has_anthropic_key():
            if allow_dry:
                return DryRunClient("anthropic-dry")
            raise RuntimeError("ANTHROPIC_API_KEY 未设置")
        return AnthropicClient(model)
    if provider in ("google", "gemini"):
        if not _has_google_key():
            if allow_dry:
                return DryRunClient("gemini-dry")
            raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY 未设置")
        return GeminiClient(model)
    raise ValueError(f"未知 provider: {provider}")


def get_client_strict(spec: str) -> LLMClient:
    return get_client(spec, allow_dry=False)
