"""Static checks on the package source: no dangerous calls, no suppressed findings."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "langsmith_agent_evals"
FILES = sorted(PKG.glob("*.py"))
BANNED_CALLS = {"eval", "exec", "compile", "__import__"}
BANNED_MODULES = {"pickle", "marshal", "shelve", "subprocess", "yaml", "requests", "os.system"}


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_no_dangerous_code(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for marker in ("# noqa", "# nosec", "type: ignore", "shell=True", "verify=False"):
        assert marker not in text, f"{path.name}: {marker}"
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in BANNED_CALLS, f"{path.name}: {node.func.id}()"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in BANNED_MODULES, (
                    f"{path.name}: import {alias.name}"
                )
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in BANNED_MODULES, (
                f"{path.name}: from {node.module}"
            )
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "execute" and node.args:
            assert not isinstance(node.args[0], ast.JoinedStr), f"{path.name}: f-string SQL"


def test_no_secrets_or_private_urls_in_package() -> None:
    for path in PKG.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".json", ".md"}:
            text = path.read_text(encoding="utf-8")
            assert "sk-" not in text and "lsv2_" not in text, path.name
