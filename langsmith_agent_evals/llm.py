"""Chat models through ``init_chat_model``: local ollama by default, Anthropic optional, the fake
for offline runs. The same factory builds the agent model and the judge model."""

from __future__ import annotations

import ipaddress
import os
from typing import Literal
from urllib.parse import urlparse

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from langsmith_agent_evals.config import Settings
from langsmith_agent_evals.fake import FakeSupportModel

__all__ = ["ModelConfigError", "is_loopback_url", "make_model"]


class ModelConfigError(RuntimeError):
    """The configured provider can't be used as configured."""


def is_loopback_url(url: str) -> bool:
    host = urlparse(url if "://" in url else f"http://{url}").hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_model(settings: Settings, role: Literal["agent", "judge"] = "agent") -> BaseChatModel:
    provider = settings.provider if role == "agent" else settings.judge_provider
    name = settings.model if role == "agent" else settings.judge_model
    if provider == "fake":
        return FakeSupportModel(model_name=name)
    if provider == "ollama":
        if not settings.allow_remote_ollama and not is_loopback_url(settings.ollama_host):
            raise ModelConfigError(
                f"OLLAMA_HOST={settings.ollama_host!r} is not loopback; set "
                "LSE_ALLOW_REMOTE_OLLAMA=1 to send prompts to another machine"
            )
        model = init_chat_model(
            name,
            model_provider="ollama",
            base_url=settings.ollama_host,
            temperature=settings.temperature,
            reasoning=False,
        )
    elif provider == "anthropic":  # pragma: no cover - optional extra + the reader's own key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ModelConfigError("ANTHROPIC_API_KEY is not set")
        try:
            model = init_chat_model(
                name, model_provider="anthropic", temperature=settings.temperature
            )
        except ImportError as exc:
            raise ModelConfigError(
                "pip install 'langsmith-agent-evals[anthropic]' to use Anthropic"
            ) from exc
    else:  # pragma: no cover - Settings validates the provider
        raise ModelConfigError(f"unknown provider {provider!r}")
    if not isinstance(model, BaseChatModel):  # pragma: no cover - init_chat_model contract
        raise ModelConfigError("init_chat_model did not return a chat model")
    return model
