"""Offline by construction: LangSmith env vars are cleared, test tracking is off, and every
socket connection and DNS lookup is refused and recorded. The session fails if anything tried."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest

for _key in list(os.environ):
    if _key.startswith(("LANGSMITH_", "LANGCHAIN_", "ANTHROPIC_", "OPENAI_", "LSE_")):
        del os.environ[_key]
#: The langsmith pytest plugin and ``expect`` run locally and never send results.
os.environ["LANGSMITH_TEST_TRACKING"] = "false"

from langsmith_agent_evals.config import Settings
from langsmith_agent_evals.data import Dataset
from langsmith_agent_evals.data import load_dataset
from langsmith_agent_evals.fake import FakeSupportModel
from langsmith_agent_evals.harness import Experiment
from langsmith_agent_evals.harness import run_experiment

NETWORK_ATTEMPTS: list[Any] = []


class NetworkBlocked(OSError):
    pass


def _refuse_connect(self: socket.socket, address: Any) -> None:
    NETWORK_ATTEMPTS.append(("connect", address))
    raise NetworkBlocked(f"network disabled in tests: connect {address!r}")


def _refuse_connect_ex(self: socket.socket, address: Any) -> int:
    NETWORK_ATTEMPTS.append(("connect_ex", address))
    raise NetworkBlocked(f"network disabled in tests: connect_ex {address!r}")


def _refuse_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    NETWORK_ATTEMPTS.append(("getaddrinfo", host))
    raise NetworkBlocked(f"network disabled in tests: resolve {host!r}")


@pytest.fixture(scope="session", autouse=True)
def no_network() -> Iterator[list[Any]]:
    saved = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo)
    socket.socket.connect = _refuse_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _refuse_connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = _refuse_getaddrinfo
    try:
        yield NETWORK_ATTEMPTS
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = saved  # type: ignore[method-assign]
    assert NETWORK_ATTEMPTS == [], f"tests tried the network: {NETWORK_ATTEMPTS}"


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    return load_dataset()


@pytest.fixture
def settings() -> Settings:
    return Settings(provider="fake")


@pytest.fixture
def fake() -> FakeSupportModel:
    return FakeSupportModel()


@pytest.fixture(scope="session")
def experiments(no_network: list[Any]) -> dict[str, Experiment]:
    """v1, v2 and degraded over the golden set, once per session."""
    s = Settings(provider="fake")
    return {p: run_experiment(p, settings=s) for p in ("v1", "v2", "degraded")}
