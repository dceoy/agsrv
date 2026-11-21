"""Shared test setup for existing query and job tests."""

from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse]
def healthy_claude_for_existing_tests(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep backend behavior tests independent of Claude availability."""
    if request.node.path.name != "test_claude_probe.py":
        monkeypatch.setattr("agsrv.claude.probe", AsyncMock(return_value=True))
