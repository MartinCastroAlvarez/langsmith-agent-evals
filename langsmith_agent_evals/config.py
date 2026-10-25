"""Settings from ``LSE_*`` environment variables. Every value has a local, free default."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import date

__all__ = ["DEFAULT_MODELS", "PROVIDERS", "Settings"]

PROVIDERS = ("ollama", "anthropic", "fake")
DEFAULT_MODELS = {"ollama": "qwen3:4b", "anthropic": "claude-sonnet-5-5", "fake": "fake-support"}
DEFAULT_JUDGE_MODELS = {
    "ollama": "qwen3:4b",
    "anthropic": "claude-haiku-4-5-20251001",
    "fake": "fake-judge",
}


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. ``Settings.from_env()`` reads the environment."""

    provider: str = "ollama"
    model: str = ""
    judge_provider: str = ""
    judge_model: str = ""
    ollama_host: str = "http://127.0.0.1:11434"
    allow_remote_ollama: bool = False
    temperature: float = 0.0
    max_steps: int = 8
    #: USD per 1M input / output tokens; 0 for a local model.
    price_in: float = 0.0
    price_out: float = 0.0
    #: The day the agent believes it is (``days_since_delivery``); the dataset pins it.
    today: date = field(default_factory=date.today)

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}, got {self.provider!r}")
        judge = self.judge_provider or self.provider
        if judge not in PROVIDERS:
            raise ValueError(f"judge provider must be one of {PROVIDERS}, got {judge!r}")
        object.__setattr__(self, "judge_provider", judge)
        if not self.model:
            object.__setattr__(self, "model", DEFAULT_MODELS[self.provider])
        if not self.judge_model:
            object.__setattr__(self, "judge_model", DEFAULT_JUDGE_MODELS[judge])
        if not 1 <= self.max_steps <= 25:
            raise ValueError("max_steps must be between 1 and 25")
        if self.price_in < 0 or self.price_out < 0:
            raise ValueError("prices must be >= 0")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        return cls(
            provider=env.get("LSE_PROVIDER", "ollama").strip() or "ollama",
            model=env.get("LSE_MODEL", "").strip(),
            judge_provider=env.get("LSE_JUDGE_PROVIDER", "").strip(),
            judge_model=env.get("LSE_JUDGE_MODEL", "").strip(),
            ollama_host=env.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip(),
            allow_remote_ollama=env.get("LSE_ALLOW_REMOTE_OLLAMA", "") == "1",
            temperature=_float(env, "LSE_TEMPERATURE", 0.0),
            max_steps=int(env.get("LSE_MAX_STEPS", "8") or 8),
            price_in=_float(env, "LSE_PRICE_IN", 0.0),
            price_out=_float(env, "LSE_PRICE_OUT", 0.0),
        )
