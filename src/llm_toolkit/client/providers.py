"""Provider specifications.

Only first-party endpoints are supported: OpenRouter, Google AI Studio (its OpenAI-compatible
endpoint) and the Anthropic Messages API. Model ids are configuration: each default can be overridden
with ``LLM_TOOLKIT_<NAME>_MODEL`` and the order with ``LLM_TOOLKIT_PROVIDERS``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    kind: Literal["openai", "anthropic"]
    base_url: str
    model: str
    key_env: str
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    write_timeout: float = 10.0
    max_retries: int = 2  # 0 is valid: exactly one attempt
    # A 429 whose Retry-After is longer than this is treated as "quota exhausted": the provider is
    # put in cooldown and the client fails over instead of waiting.
    cooldown_threshold_s: float = 30.0
    # (regex, cooldown seconds) checked against a 429 body when Retry-After does not decide it.
    quota_patterns: tuple[tuple[str, float], ...] = ()
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def with_model(self, model: str) -> ProviderSpec:
        return replace(self, model=model)


DEFAULT_MODELS: dict[str, str] = {
    "openrouter": "google/gemini-3.6-flash",
    "google": "gemini-3.6-flash",
    "anthropic": "claude-opus-5-5",
}


def _model(name: str, env: Mapping[str, str]) -> str:
    return env.get(f"LLM_TOOLKIT_{name.upper()}_MODEL") or DEFAULT_MODELS[name]


def openrouter(env: Mapping[str, str] | None = None) -> ProviderSpec:
    env = os.environ if env is None else env
    return ProviderSpec(
        name="openrouter",
        kind="openai",
        base_url="https://openrouter.ai/api/v1",
        model=_model("openrouter", env),
        key_env="OPENROUTER_API_KEY",
        # Ask OpenRouter to include the billed cost in `usage`.
        extra_body={"usage": {"include": True}},
    )


def google(env: Mapping[str, str] | None = None) -> ProviderSpec:
    env = os.environ if env is None else env
    return ProviderSpec(
        name="google",
        kind="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model=_model("google", env),
        key_env="GEMINI_API_KEY",
        # The free tier queues behind paid traffic: fail fast and let the next provider answer.
        read_timeout=20.0,
        max_retries=1,
        quota_patterns=((r"PerDay", 3600.0), (r"PerMinute", 60.0), (r"RESOURCE_EXHAUSTED", 60.0)),
    )


def anthropic(env: Mapping[str, str] | None = None) -> ProviderSpec:
    env = os.environ if env is None else env
    return ProviderSpec(
        name="anthropic",
        kind="anthropic",
        base_url="https://api.anthropic.com/v1",
        model=_model("anthropic", env),
        key_env="ANTHROPIC_API_KEY",
        read_timeout=60.0,
    )


BUILTIN = {"openrouter": openrouter, "google": google, "anthropic": anthropic}


def default_providers(env: Mapping[str, str] | None = None) -> list[ProviderSpec]:
    """Providers in failover order (``LLM_TOOLKIT_PROVIDERS``, default ``google,openrouter,anthropic``)."""
    env = os.environ if env is None else env
    order = env.get("LLM_TOOLKIT_PROVIDERS", "google,openrouter,anthropic")
    names = [n.strip() for n in order.split(",") if n.strip()]
    unknown = [n for n in names if n not in BUILTIN]
    if unknown:
        raise ValueError(f"unknown provider(s) in LLM_TOOLKIT_PROVIDERS: {', '.join(unknown)}")
    return [BUILTIN[n](env) for n in names]
