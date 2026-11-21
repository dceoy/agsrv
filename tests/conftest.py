"""Shared test setup for existing query and job tests."""

from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse]
def healthy_claude_for_existing_tests(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep backend behavior tests independent of Claude availability.

    Queries and jobs now run the Claude availability probe (a ``/usage`` run plus
    ``auth status``) before the real turn. Existing tests use fake runners that
    count calls, hold slots, and rely on timing, so a real probe would consume
    stream calls and slot time and make them depend on CLI login state. Stubbing
    ``claude.probe`` keeps them deterministic; ``test_claude_probe.py`` opts out
    and exercises the real probe.
    """
    if request.node.path.name != "test_claude_probe.py":
        monkeypatch.setattr("agsrv.claude.probe", AsyncMock(return_value=True))
