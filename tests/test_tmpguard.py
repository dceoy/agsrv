"""Disk admission and per-turn temporary directory tests."""

from __future__ import annotations

import shutil
import stat
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agsrv.agent import build_options
from agsrv.config import Settings
from agsrv.main import create_app
from agsrv.schemas import QueryRequest
from agsrv.tmpguard import check_space, turn_tmpdir
from tests.test_main import FakeRunner


def test_directory_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Private directories are removed on success and error; cleanup errors log."""
    with turn_tmpdir(tmp_path) as directory:
        assert directory.parent == tmp_path
        assert directory.name.startswith(".agsrv-tmp-")
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        (directory / "file").write_text("test")
    assert not directory.exists()
    msg = "turn"
    with pytest.raises(ValueError, match="turn"), turn_tmpdir(tmp_path) as directory:
        raise ValueError(msg)
    assert not directory.exists()
    remove = shutil.rmtree
    monkeypatch.setattr(shutil, "rmtree", MagicMock(side_effect=OSError("cleanup")))
    with turn_tmpdir(tmp_path) as directory:
        pass
    assert "Failed to remove" in caplog.text
    remove(directory)


@pytest.mark.parametrize("sandbox", [False, True])
def test_options_tmpdir(
    tmp_path: Path, sandbox: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both Claude configurations carry the turn directory."""
    monkeypatch.setattr(
        "agsrv.agent.prepare_sandbox_cache", MagicMock(return_value=tmp_path)
    )
    monkeypatch.setattr("agsrv.agent.bundled_cli_path", lambda: tmp_path)
    request = QueryRequest(agent="claude", prompt="test", sandbox=sandbox)
    options = build_options(
        request,
        Settings(workspace_root=tmp_path),
        tmp_path,
        tmp_path,
        tmp_path / "private",
    )
    assert options.env["TMPDIR"] == str(tmp_path / "private")
    if sandbox:
        assert options.env["XDG_CACHE_HOME"] == str(tmp_path / ".cache/agsrv-sandbox")


@pytest.mark.parametrize("endpoint", ["/v1/query", "/v1/jobs"])
@pytest.mark.parametrize("agent", ["claude", "codex"])
@pytest.mark.parametrize("low", ["tmp", "workspace"])
def test_reject_low_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str, agent: str, low: str
) -> None:
    """Both APIs and agents reject either low filesystem before starting."""
    settings = Settings(
        api_key="secret",
        workspace_root=tmp_path,
        jobs_db_path=tmp_path / "jobs.db",
        tmp_min_free_bytes=100,
        workspace_min_free_bytes=200,
    )

    def usage(path: Path) -> shutil._ntuple_diskusage:  # pyright: ignore[reportPrivateUsage]
        free = 1 if (str(path) == "/tmp") == (low == "tmp") else 1000  # ruff: ignore[hardcoded-temp-file]
        return shutil._ntuple_diskusage(1000, 1000 - free, free)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(shutil, "disk_usage", usage)
    runner = FakeRunner()
    with TestClient(create_app(settings, runner)) as client:
        response = client.post(
            endpoint,
            headers={"Authorization": "Bearer secret"},
            json={
                "agent": agent,
                "prompt": "test",
                **({"sandbox": False} if agent == "claude" else {}),
            },
        )
        assert response.status_code == 503
        expected = "/tmp" if low == "tmp" else str(tmp_path)  # ruff: ignore[hardcoded-temp-file]
        assert expected in response.json()["detail"]
        assert "1 bytes free" in response.json()["detail"]
        assert runner.stream_calls == 0
        assert not list(tmp_path.glob(".agsrv-tmp-*"))


def test_check_disabled_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero disables checks, and an unavailable filesystem fails closed."""
    settings = Settings(
        workspace_root=tmp_path, tmp_min_free_bytes=0, workspace_min_free_bytes=0
    )
    mock = MagicMock(side_effect=OSError("unavailable"))
    monkeypatch.setattr(shutil, "disk_usage", mock)
    check_space(settings)
    mock.assert_not_called()
    with pytest.raises(HTTPException) as exc:
        check_space(replace(settings, tmp_min_free_bytes=1))
    assert exc.value.status_code == 503


def test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Thresholds parse their explicit values including disabled checks."""
    monkeypatch.setenv("AGSRV_TMP_MIN_FREE_BYTES", "0")
    monkeypatch.setenv("AGSRV_WORKSPACE_MIN_FREE_BYTES", "123")
    settings = Settings.from_env()
    assert settings.tmp_min_free_bytes == 0
    assert settings.workspace_min_free_bytes == 123


@pytest.mark.parametrize("endpoint", ["/v1/query", "/v1/jobs"])
@pytest.mark.parametrize("fail", [False, True])
def test_turn_directory_lifecycle(tmp_path: Path, endpoint: str, fail: bool) -> None:
    """Claude receives a private directory that is removed after either outcome."""
    from tests.test_jobs import _wait  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[import-outside-top-level]

    settings = Settings(
        api_key="secret", workspace_root=tmp_path, jobs_db_path=tmp_path / "jobs.db"
    )
    runner = FakeRunner(fail_after=0 if fail else None)
    with TestClient(create_app(settings, runner)) as client:
        response = client.post(
            endpoint,
            headers={"Authorization": "Bearer secret"},
            json={"agent": "claude", "prompt": "test", "sandbox": False},
        )
        if endpoint == "/v1/jobs":
            outcome = _wait(client, response.json()["job_id"])
            assert outcome["status"] == ("failed" if fail else "succeeded")
        else:
            assert response.status_code == (502 if fail else 200)
        assert runner.options is not None
        directory = Path(runner.options.env["TMPDIR"])
        assert directory.parent == tmp_path
        assert not directory.exists()
