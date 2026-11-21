"""Shared test isolation."""

from __future__ import annotations

import shutil

import pytest


@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse]
def free_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep disk guard tests independent of host free space."""

    def usage(_path: object) -> object:
        return shutil._ntuple_diskusage(10**12, 0, 10**12)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(shutil, "disk_usage", usage)
