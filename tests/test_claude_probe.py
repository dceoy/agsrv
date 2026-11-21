"""Claude availability, caching, and cancellation coverage."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import pytest
from claude_agent_sdk import ResultMessage
from fastapi.testclient import TestClient

from agsrv import claude, main
from agsrv.config import Settings
from agsrv.sandbox import SandboxSetupError
from tests.test_jobs import AUTH, BODY, _wait
from tests.test_main import FakeRunner, canned

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


LOGGED_IN = '{"loggedIn": true}'


def fake_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: str = LOGGED_IN,
    *,
    code: int = 0,
    body: str = "",
) -> None:
    """Install a fake bundled CLI whose ``auth status`` prints ``output``."""
    cli = tmp_path / "fake-claude"
    cli.write_text(
        "#!/bin/sh\n"
        f"{body}\n"
        '[ "$1 $2" = "auth status" ] || exit 9\n'
        f"printf '%s' '{output}'\nexit {code}\n"
    )
    cli.chmod(0o755)
    monkeypatch.setattr(claude, "bundled_cli_path", lambda: cli)


@pytest.mark.asyncio
async def test_probe_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe results are cached until expiry."""
    fake_cli(tmp_path, monkeypatch)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "code"),
    [
        ('{"loggedIn": false}', 0),
        ('{"loggedIn": "true"}', 0),
        ('{"loggedIn": 1}', 0),
        ("{}", 0),
        ("[true]", 0),
        ("not json", 0),
        ("", 0),
        (LOGGED_IN, 1),
    ],
)
async def test_auth_status_gates_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str, code: int
) -> None:
    """Anything but exit 0 and loggedIn true is unhealthy despite /usage."""
    fake_cli(tmp_path, monkeypatch, output, code=code)
    runner = FakeRunner()
    assert not await claude.probe(runner, Settings(workspace_root=tmp_path))
    assert runner.stream_calls == 1


@pytest.mark.asyncio
async def test_auth_status_uses_query_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``auth status`` runs with the same environment as the query path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    fake_cli(
        tmp_path,
        monkeypatch,
        body=(
            '[ "$CLAUDE_CODE_DISABLE_AUTO_MEMORY" = 1 ] || exit 3\n'
            f'[ "$HOME" = "{tmp_path}" ] || exit 4'
        ),
    )
    assert await claude.probe(FakeRunner(), Settings(workspace_root=tmp_path))


@pytest.mark.asyncio
async def test_auth_status_matches_sdk_turn_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``auth status`` has no CLAUDECODE and runs in the workspace."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("CLAUDECODE", "1")
    fake_cli(
        tmp_path,
        monkeypatch,
        body=(
            '[ -z "${CLAUDECODE+x}" ] || exit 3\n'
            f'[ "$(pwd -P)" = "{workspace.resolve()}" ] || exit 4'
        ),
    )
    assert await claude.probe(FakeRunner(), Settings(workspace_root=workspace))


@pytest.mark.asyncio
async def test_auth_status_tolerates_leading_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warning line before the JSON object still parses."""
    fake_cli(tmp_path, monkeypatch, body="echo 'warning: something odd'")
    assert await claude.probe(FakeRunner(), Settings(workspace_root=tmp_path))


@pytest.mark.asyncio
async def test_auth_status_spawn_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing or unspawnable CLI is unhealthy."""
    settings = Settings(workspace_root=tmp_path)

    def missing() -> Path:
        msg = "missing"
        raise SandboxSetupError(msg)

    monkeypatch.setattr(claude, "bundled_cli_path", missing)
    assert not await claude.probe(FakeRunner(), settings)
    monkeypatch.setattr(claude, "bundled_cli_path", lambda: tmp_path / "nope")
    assert not await claude.probe(FakeRunner(), settings)


@pytest.mark.asyncio
async def test_auth_status_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hanging ``auth status`` is killed and reported unhealthy."""
    fake_cli(tmp_path, monkeypatch, body="exec sleep 30")
    monkeypatch.setattr(claude, "AUTH_STATUS_TIMEOUT", 0.2)
    spawned: list[asyncio.subprocess.Process] = []
    real = cast(
        "Callable[..., Awaitable[asyncio.subprocess.Process]]",
        asyncio.create_subprocess_exec,
    )

    async def spy(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await real(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    assert not await claude.probe(FakeRunner(), Settings(workspace_root=tmp_path))
    assert len(spawned) == 1
    assert spawned[0].returncode is not None


@pytest.mark.asyncio
async def test_error_result_is_unhealthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``is_error`` result such as "Not logged in" is unhealthy."""
    fake_cli(tmp_path, monkeypatch)
    failed = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=True,
        num_turns=1,
        session_id="s",
        result="Not logged in",
    )
    runner = FakeRunner(messages=[*canned()[:-1], failed])
    assert not await claude.probe(runner, Settings(workspace_root=tmp_path))


@pytest.mark.asyncio
async def test_probe_never_uses_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sandbox prerequisites neither fail the probe nor touch the cache."""
    fake_cli(tmp_path, monkeypatch)

    def broken(*_args: object) -> None:
        msg = "sandbox prerequisite failed"
        raise SandboxSetupError(msg)

    for name in ("service_home", "prepare_sandbox_cache", "build_sandbox_settings"):
        monkeypatch.setattr(claude, name, broken)
    runner = FakeRunner()
    assert await claude.probe(runner, Settings(workspace_root=tmp_path))
    assert runner.options is not None
    assert runner.options.sandbox is None


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


def test_uncached_query_probes_then_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no cached value a query runs the probe and reports 503."""
    settings = replace(
        Settings(workspace_root=tmp_path),
        api_key="secret",
        jobs_db_path=tmp_path / "jobs.db",
    )
    runner = FakeRunner()
    probe = AsyncMock(return_value=False)
    monkeypatch.setattr(claude, "probe", probe)
    with TestClient(main.create_app(settings, runner)) as client:
        response = client.post("/v1/query", headers=AUTH, json=BODY)
        assert response.status_code == 503
        assert probe.await_count == 1
        assert runner.stream_calls == 0


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
    monkeypatch.setattr(main, "PROBE_TIMEOUT", 0.001)

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
    """Concurrent callers share a probe; a busy slot serves the cached value."""
    app = main.create_app(Settings(workspace_root=tmp_path, max_concurrency=1))
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def probe(*_args: object) -> bool:
        nonlocal calls
        calls += 1
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
    assert calls == 1
    # Expired cache plus busy slot: cached healthy value, no 429, no probe.
    app.state.claude_probe_expires = 0
    await app.state.semaphore.acquire()
    assert await main._claude_available(app)
    assert calls == 1
    app.state.claude_healthy = False
    assert not await main._claude_available(app)
    app.state.semaphore.release()


def test_agents_listing_never_429_when_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /v1/agents returns the cached healthy value while the slot is busy."""
    settings = replace(
        Settings(workspace_root=tmp_path, max_concurrency=1),
        api_key="secret",
        jobs_db_path=tmp_path / "jobs.db",
    )
    probe = AsyncMock(return_value=True)
    monkeypatch.setattr(claude, "probe", probe)
    app = main.create_app(settings, FakeRunner())
    with TestClient(app) as client:
        assert client.get("/v1/agents", headers=AUTH).json()["claude"]["healthy"]
        app.state.claude_probe_expires = 0
        app.state.semaphore._value = 0  # slot busy
        response = client.get("/v1/agents", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["claude"]["healthy"] is True
        assert probe.await_count == 1


def test_cached_unhealthy_query_returns_503_before_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh unhealthy cache gives 503 even when the slot is busy (no 429)."""
    settings = replace(
        Settings(workspace_root=tmp_path, max_concurrency=1),
        api_key="secret",
        jobs_db_path=tmp_path / "jobs.db",
    )
    probe = AsyncMock(return_value=False)
    monkeypatch.setattr(claude, "probe", probe)
    app = main.create_app(settings, FakeRunner())
    with TestClient(app) as client:
        assert not client.get("/v1/agents", headers=AUTH).json()["claude"]["healthy"]
        app.state.semaphore._value = 0  # slot busy
        response = client.post("/v1/query", headers=AUTH, json=BODY)
        assert response.status_code == 503
        assert response.json()["detail"] == "Claude is disabled or unavailable"
        # Expired cache: the busy slot yields the usual 429 again.
        app.state.claude_probe_expires = 0
        assert client.post("/v1/query", headers=AUTH, json=BODY).status_code == 429


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
