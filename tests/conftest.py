"""Shared test isolation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from agsrv import tmpguard
from agsrv.config import Settings

if TYPE_CHECKING:
    from collections.abc import Callable


def vfs(free: int) -> os.statvfs_result:
    """Return a ``statvfs`` result with ``free`` available bytes."""
    return os.statvfs_result((1, 1, free, free, free, 0, 0, 0, 0, 255))


def statvfs_stub(free: Callable[[], int]) -> Callable[[object], os.statvfs_result]:
    """Build a ``statvfs``/``fstatvfs`` replacement reporting ``free()`` bytes.

    Returns:
        The replacement function.
    """

    def stub(_target: object) -> os.statvfs_result:
        return vfs(free())

    return stub


@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse]
def free_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep disk guard tests independent of host free space."""
    huge = statvfs_stub(lambda: 10**12)
    monkeypatch.setattr(os, "statvfs", huge)
    monkeypatch.setattr(os, "fstatvfs", huge)


@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse]
def staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Isolate test Settings: private staging dir, jobs DB and staging policy."""
    base = tmp_path_factory.getbasetemp()
    original = Settings.__init__
    # pytest's own tmp dirs live under /tmp, which real deployments may not use.
    monkeypatch.setattr(tmpguard, "_forbidden_stage_roots", tuple)

    def init(self: Settings, *args: Any, **kwargs: Any) -> None:  # ruff: ignore[any-type]
        if "jobs_db_path" not in kwargs:  # never touch the real ~/.local/state DB
            kwargs["jobs_db_path"] = Path(tempfile.mkdtemp(dir=base)) / "jobs.sqlite3"
        root = kwargs.get("workspace_root")
        if root is not None and "tmp_staging_dirs" not in kwargs:
            root = Path(root).resolve()
            if root.is_relative_to(base):
                stage_base = base / "staging"
                stage_base.mkdir(exist_ok=True)
                stage = Path(tempfile.mkdtemp(dir=stage_base))
                kwargs["tmp_staging_dirs"] = ((root, stage),)
        original(self, *args, **kwargs)

    monkeypatch.setattr(Settings, "__init__", init)


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
