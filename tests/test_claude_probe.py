"""Claude availability, caching, and cancellation coverage."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from agsrv import claude, main
from agsrv.config import Settings
from tests.test_jobs import AUTH, BODY, _wait
from tests.test_main import FakeRunner

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_probe_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe results are cached until expiry."""
    runner = FakeRunner()
    settings = Settings(workspace_root=tmp_path)
    assert await claude.probe(runner, settings)
    assert runner.stream_calls == 1
    assert not await claude.probe(FakeRunner(fail_after=0), settings)
    assert not await claude.probe(FakeRunner(messages=[]), settings)
    app = main.create_app(settings, runner)
    probe = AsyncMock(side_effect=[True, False])
    monkeypatch.setattr(claude, "probe", probe)
    assert await main._claude_available(app)
    assert await main._claude_available(app)
    assert probe.await_count == 1
    app.state.claude_probe_expires = 0
    assert not await main._claude_available(app)
    assert probe.await_count == 2


def test_endpoints_and_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Availability controls health, queries, and jobs."""
    settings = replace(
        Settings(workspace_root=tmp_path),
        api_key="secret",
        jobs_db_path=tmp_path / "jobs.db",
    )
    runner = FakeRunner()
    probe = AsyncMock(return_value=False)
    monkeypatch.setattr(claude, "probe", probe)
    app = main.create_app(settings, runner)
    with TestClient(app) as client:
        assert (
            client.get("/v1/agents", headers=AUTH).json()["claude"]["healthy"] is False
        )
        response = client.post("/v1/query", headers=AUTH, json=BODY)
        assert response.status_code == 503
        assert response.json()["detail"] == "Claude is disabled or unavailable"
        job = client.post("/v1/jobs", headers=AUTH, json=BODY).json()
        assert _wait(client, job["job_id"])["status"] == "failed"
        assert runner.stream_calls == 0
        probe.return_value = True
        app.state.claude_probe_expires = 0
        assert (
            client.get("/v1/agents", headers=AUTH).json()["claude"]["healthy"] is True
        )


@pytest.mark.asyncio
async def test_cancelled_probe_releases_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation retains capacity until cleanup completes."""
    app = main.create_app(Settings(workspace_root=tmp_path, max_concurrency=1))
    started = asyncio.Event()
    cleanup = asyncio.Event()

    async def slow(*_args: object) -> bool:
        started.set()
        try:
            await asyncio.Future[None]()
        finally:
            await cleanup.wait()
        return True

    monkeypatch.setattr(claude, "probe", slow)
    task = asyncio.create_task(main._claude_available(app))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert app.state.semaphore.locked()
    assert not app.state.claude_healthy
    cleanup.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not app.state.semaphore.locked()
    assert app.state.claude_probe_expires == 0


@pytest.mark.asyncio
async def test_probe_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A timed out probe reports unavailable and releases its slot."""
    app = main.create_app(Settings(workspace_root=tmp_path, max_concurrency=1))
    monkeypatch.setattr(main, "CODEX_PROBE_TIMEOUT", 0.001)

    async def slow(*_args: object) -> bool:
        await asyncio.sleep(60)
        return True

    monkeypatch.setattr(claude, "probe", slow)
    assert not await main._claude_available(app)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not app.state.semaphore.locked()


@pytest.mark.asyncio
async def test_concurrent_probe_and_busy_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent callers share a probe and a busy slot returns 429."""
    app = main.create_app(Settings(workspace_root=tmp_path, max_concurrency=1))
    started = asyncio.Event()
    release = asyncio.Event()

    async def probe(*_args: object) -> bool:
        started.set()
        await release.wait()
        return True

    monkeypatch.setattr(claude, "probe", probe)
    first = asyncio.create_task(main._claude_available(app))
    await started.wait()
    second = asyncio.create_task(main._claude_available(app))
    await asyncio.sleep(0)
    release.set()
    assert await first
    assert await second
    app.state.claude_probe_expires = 0
    await app.state.semaphore.acquire()
    with pytest.raises(main.HTTPException) as error:
        await main._claude_available(app)
    assert error.value.status_code == 429
    app.state.semaphore.release()


@pytest.mark.asyncio
async def test_cancel_does_not_invalidate_new_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late cleanup does not invalidate a newer cache entry."""
    app = main.create_app(Settings(workspace_root=tmp_path))
    started = asyncio.Event()
    cleanup = asyncio.Event()

    async def probe(*_args: object) -> bool:
        started.set()
        try:
            await asyncio.Future[None]()
        finally:
            await cleanup.wait()
        return True

    monkeypatch.setattr(claude, "probe", probe)
    task = asyncio.create_task(main._claude_available(app))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    app.state.claude_probe_expires = 123.0
    cleanup.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert app.state.claude_probe_expires == pytest.approx(123.0)
