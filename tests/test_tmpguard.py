"""Disk admission and per-turn temporary directory tests."""

from __future__ import annotations

import shutil
import stat
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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
        (directory / "nested").mkdir()
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


def test_replaced_directory_is_preserved(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Cleanup never removes a substitute directory at the original path."""
    with turn_tmpdir(tmp_path) as directory:
        moved = tmp_path / "moved"
        directory.rename(moved)
        directory.mkdir()
        (directory / "keep").write_text("safe")
    assert (directory / "keep").read_text() == "safe"
    assert "changed; skipping cleanup" in caplog.text


def test_restrictive_umask(tmp_path: Path) -> None:
    """Creation succeeds and is usable even with every mode bit masked."""
    import os  # ruff: ignore[import-outside-top-level]

    previous = os.umask(0o777)
    try:
        with turn_tmpdir(tmp_path) as directory:
            assert stat.S_IMODE(directory.stat().st_mode) == 0o700
            (directory / "file").write_text("usable")
    finally:
        os.umask(previous)
    assert not directory.exists()


@pytest.mark.parametrize("name", ["tmp_min_free_bytes", "workspace_min_free_bytes"])
def test_negative_threshold(name: str) -> None:
    """Negative thresholds name the offending setting."""
    with pytest.raises(ValueError, match=f"AGSRV_{name.upper()}"):
        Settings(tmp_min_free_bytes=-1) if name == "tmp_min_free_bytes" else Settings(
            workspace_min_free_bytes=-1
        )


def test_cwd_separate_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admission checks the actual cwd even if its mount differs from the root."""
    cwd = tmp_path / "mounted"
    cwd.mkdir()
    settings = Settings(
        workspace_root=tmp_path, tmp_min_free_bytes=0, workspace_min_free_bytes=100
    )

    def usage(path: Path) -> shutil._ntuple_diskusage:  # pyright: ignore[reportPrivateUsage]
        free = 1 if path == cwd else 1000
        return shutil._ntuple_diskusage(1000, 1000 - free, free)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(shutil, "disk_usage", usage)
    with pytest.raises(HTTPException, match=str(cwd)):
        check_space(settings, cwd)


def test_job_replay_and_low_space_leave_no_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing keys replay under low space; new jobs leave no rows or capacity held."""
    import sqlite3  # ruff: ignore[import-outside-top-level]

    from tests.test_jobs import _wait  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[import-outside-top-level]

    settings = Settings(
        api_key="secret",
        workspace_root=tmp_path,
        jobs_db_path=tmp_path / "jobs.db",
        tmp_min_free_bytes=1,
    )
    app = create_app(settings, FakeRunner())
    headers = {"Authorization": "Bearer secret", "Idempotency-Key": "replay"}
    body = {"agent": "claude", "prompt": "test", "sandbox": False}
    with TestClient(app) as client:
        first = client.post("/v1/jobs", headers=headers, json=body)
        assert _wait(client, first.json()["job_id"])["status"] == "succeeded"
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            MagicMock(return_value=shutil.disk_usage(tmp_path)._replace(free=0)),
        )
        assert (
            client.post("/v1/jobs", headers=headers, json=body).json()["job_id"]
            == first.json()["job_id"]
        )
        assert (
            client.post(
                "/v1/jobs", headers=headers, json={**body, "prompt": "other"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/v1/jobs", headers={"Authorization": "Bearer secret"}, json=body
            ).status_code
            == 503
        )
        with sqlite3.connect(settings.jobs_db_path) as db:
            assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert app.state.semaphore._value == settings.max_concurrency
        assert app.state.codex_semaphore._value == settings.codex_max_concurrency
        assert not app.state.jobs.workspaces


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude", "codex"])
async def test_query_rechecks_after_workspace_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str
) -> None:
    """Queued queries create no directory and reject newly exhausted space."""
    import asyncio  # ruff: ignore[import-outside-top-level]

    import httpx  # ruff: ignore[import-outside-top-level]

    settings = Settings(
        api_key="secret",
        workspace_root=tmp_path,
        tmp_min_free_bytes=1,
        workspace_min_free_bytes=1,
        max_concurrency=1,
    )
    app = create_app(settings, FakeRunner())
    monkeypatch.setattr("agsrv.main._codex_available", AsyncMock(return_value=True))
    low = False

    def usage(_path: Path) -> shutil._ntuple_diskusage:  # pyright: ignore[reportPrivateUsage]
        return shutil._ntuple_diskusage(100, 100 if low else 0, 0 if low else 100)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(shutil, "disk_usage", usage)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with app.state.jobs.workspace(tmp_path):
            task = asyncio.create_task(
                client.post(
                    "/v1/query",
                    headers={"Authorization": "Bearer secret"},
                    json={
                        "agent": agent,
                        "prompt": "test",
                        **({"sandbox": False} if agent == "claude" else {}),
                    },
                )
            )
            for _ in range(100):
                if (
                    app.state.semaphore.locked()
                    or app.state.codex_semaphore._value < settings.codex_max_concurrency
                ):
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.02)
            assert not task.done()
            assert not await asyncio.to_thread(
                lambda: list(tmp_path.glob(".agsrv-tmp-*"))
            )
            low = True
        assert (await asyncio.wait_for(task, 1)).status_code == 503
        assert not app.state.jobs.workspaces
        assert not await asyncio.to_thread(lambda: list(tmp_path.glob(".agsrv-tmp-*")))


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude", "codex"])
async def test_job_rechecks_after_slot_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str
) -> None:
    """A job waiting for an agent slot rechecks free space before execution."""
    import asyncio  # ruff: ignore[import-outside-top-level]

    from agsrv.schemas import CodexQueryRequest  # ruff: ignore[import-outside-top-level]

    settings = Settings(
        workspace_root=tmp_path, tmp_min_free_bytes=1, workspace_min_free_bytes=0
    )
    app = create_app(settings, FakeRunner())
    semaphore = app.state.semaphore if agent == "claude" else app.state.codex_semaphore
    for _ in range(semaphore._value):
        await semaphore.acquire()
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        MagicMock(return_value=shutil.disk_usage(tmp_path)._replace(free=0)),
    )
    body = (
        QueryRequest(agent="claude", prompt="test", sandbox=False)
        if agent == "claude"
        else CodexQueryRequest(agent="codex", prompt="test")
    )
    task = asyncio.create_task(app.state.jobs._execute(body, tmp_path))
    await asyncio.sleep(0)
    assert not task.done()
    semaphore.release()
    with pytest.raises(HTTPException) as error:
        await task
    assert error.value.status_code == 503
    assert not await asyncio.to_thread(lambda: list(tmp_path.glob(".agsrv-tmp-*")))


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", [False, True])
async def test_codex_detached_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disconnect: bool
) -> None:
    """A cancelled Codex turn retains its directory until detached cleanup ends."""
    import asyncio  # ruff: ignore[import-outside-top-level]

    from starlette.requests import Request  # ruff: ignore[import-outside-top-level]

    from agsrv import main  # ruff: ignore[import-outside-top-level]
    from agsrv.schemas import CodexQueryRequest  # ruff: ignore[import-outside-top-level]

    app = create_app(Settings(workspace_root=tmp_path))
    started, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    gone = False
    directory: Path | None = None

    async def receive() -> dict[str, str]:
        if gone:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def run(**kwargs: object) -> dict[str, object]:
        nonlocal directory
        directory = Path(str(kwargs["tmpdir"]))
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()
        return {}

    monkeypatch.setattr(main, "_codex_available", AsyncMock(return_value=True))
    monkeypatch.setattr(main.codex_agent, "run_codex", run)
    monkeypatch.setattr(main, "DISCONNECT_POLL_INTERVAL", 0.01)
    request = Request({"type": "http", "app": app}, receive)
    body = CodexQueryRequest(
        agent="codex", prompt="test", timeout_seconds=0.05 if not disconnect else 10
    )
    task = asyncio.create_task(main._codex_query_with_disconnect(body, request))  # pyright: ignore[reportPrivateUsage]
    await asyncio.wait_for(started.wait(), 1)
    assert directory is not None
    assert await asyncio.to_thread(directory.exists)
    gone = disconnect
    if disconnect:
        assert (await asyncio.wait_for(task, 1)).status_code == 499
    else:
        with pytest.raises(HTTPException) as error:
            await asyncio.wait_for(task, 1)
        assert error.value.status_code == 504
    await asyncio.wait_for(closing.wait(), 1)
    assert await asyncio.to_thread(directory.exists)
    release.set()
    for _ in range(100):
        if not main._pump_tasks:
            break
        await asyncio.sleep(0.01)
    assert not main._pump_tasks
    assert not await asyncio.to_thread(directory.exists)
    assert not app.state.jobs.workspaces
