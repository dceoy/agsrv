"""Shared test isolation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

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
    """Give every test Settings rooted under pytest's tmp dir a private staging dir."""
    base = tmp_path_factory.getbasetemp()
    original = Settings.__init__

    def init(self: Settings, *args: Any, **kwargs: Any) -> None:  # ruff: ignore[any-type]
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
