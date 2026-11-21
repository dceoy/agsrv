"""Entry-point tests: every agent path uses admission, preflight and the guard."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from agsrv import main, muse
from agsrv.claude import build_options
from agsrv.config import Settings
from agsrv.main import create_app
from agsrv.sandbox import SANDBOX_DENIED_ENV_VARS, SandboxSetupError
from agsrv.schemas import CodexQueryRequest, QueryRequest
from agsrv.tmpguard import TmpdirError
from tests.conftest import statvfs_stub
from tests.test_jobs import _wait  # pyright: ignore[reportPrivateUsage]
from tests.test_main import FakeRunner
from tests.test_tmpguard import fds

AUTH = {"Authorization": "Bearer secret"}
CLAUDE = {"agent": "claude", "prompt": "test", "sandbox": False}
CODEX = {"agent": "codex", "prompt": "test"}


def _staging(app: Any) -> Path:  # ruff: ignore[any-type]
    return Path(app.state.settings.tmp_staging_dirs[0][1])


def _clean(app: Any, root: Path) -> None:  # ruff: ignore[any-type]
    assert not list(root.glob(".agsrv-*"))
    assert not list(_staging(app).iterdir())
    assert app.state.anchors._live == 0  # pyright: ignore[reportPrivateUsage]


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:  # ruff: ignore[any-type]
    return Settings(
        api_key="secret",
        workspace_root=tmp_path,
        jobs_db_path=tmp_path.parent / f"{tmp_path.name}-jobs.db",
        **kwargs,
    )


@pytest.mark.parametrize("sandbox", [False, True])
def test_options_tmpdir(
    tmp_path: Path, sandbox: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both Claude configurations carry the turn directory."""
    monkeypatch.setattr(
        "agsrv.claude.prepare_sandbox_cache", MagicMock(return_value=tmp_path)
    )
    monkeypatch.setattr("agsrv.claude.bundled_cli_path", lambda: tmp_path)
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
@pytest.mark.parametrize("fail", [False, True])
def test_claude_turn_directory_lifecycle(
    tmp_path: Path, endpoint: str, fail: bool
) -> None:
    """Claude receives a private directory that is removed after either outcome."""
    runner = FakeRunner(fail_after=0 if fail else None)
    app = create_app(_settings(tmp_path), runner)
    with TestClient(app) as client:
        response = client.post(endpoint, headers=AUTH, json=CLAUDE)
        if endpoint == "/v1/jobs":
            outcome = _wait(client, response.json()["job_id"])
            assert outcome["status"] == ("failed" if fail else "succeeded")
        else:
            assert response.status_code == (502 if fail else 200)
        assert runner.options is not None
        directory = Path(runner.options.env["TMPDIR"])
        assert directory.parent == tmp_path
        assert directory.name.startswith(".agsrv-tmp-")
        _clean(app, tmp_path)


@pytest.mark.parametrize("endpoint", ["/v1/query", "/v1/jobs"])
def test_codex_payload_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    """Codex runs receive the canonical turn directory, removed afterwards."""
    received: list[str] = []

    async def run(**kwargs: Any) -> dict[str, Any]:  # ruff: ignore[any-type, unused-async]
        received.append(kwargs["tmpdir"])
        assert Path(kwargs["tmpdir"]).is_dir()  # ruff: ignore[blocking-path-method-in-async-function]
        return {
            "agent": "codex",
            "result": "ok",
            "model": None,
            "is_error": False,
            "duration_ms": 1,
            "usage": None,
            "backend": {},
        }

    monkeypatch.setattr(main, "_codex_available", AsyncMock(return_value=True))
    monkeypatch.setattr(main.codex, "run_codex", run)
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post(endpoint, headers=AUTH, json=CODEX)
        if endpoint == "/v1/jobs":
            assert _wait(client, response.json()["job_id"])["status"] == "succeeded"
        else:
            assert response.status_code == 200
        assert Path(received[0]).parent == tmp_path
        _clean(app, tmp_path)


@pytest.mark.parametrize("endpoint", ["/v1/query", "/v1/jobs"])
@pytest.mark.parametrize("agent", ["claude", "codex"])
@pytest.mark.parametrize("low", ["tmp", "workspace"])
def test_reject_low_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str, agent: str, low: str
) -> None:
    """Both APIs and agents reject either low filesystem before starting."""
    settings = _settings(tmp_path, tmp_min_free_bytes=100, workspace_min_free_bytes=200)
    monkeypatch.setattr(
        os, "statvfs", statvfs_stub(lambda: 1 if low == "tmp" else 1000)
    )
    monkeypatch.setattr(
        os, "fstatvfs", statvfs_stub(lambda: 1 if low != "tmp" else 1000)
    )
    runner = FakeRunner()
    app = create_app(settings, runner)
    with TestClient(app) as client:
        response = client.post(
            endpoint, headers=AUTH, json=CLAUDE if agent == "claude" else CODEX
        )
        assert response.status_code == 503
        expected = "/tmp" if low == "tmp" else str(tmp_path)  # ruff: ignore[hardcoded-temp-file]
        assert expected in response.json()["detail"]
        assert "1 bytes free" in response.json()["detail"]
        assert runner.stream_calls == 0
        _clean(app, tmp_path)


def test_no_staging_is_fatal_at_startup(tmp_path: Path) -> None:
    """A workspace root whose mount has no staging directory refuses to start."""
    settings = Settings(
        api_key="secret",
        workspace_root=tmp_path,
        jobs_db_path=tmp_path.parent / "j.db",
        tmp_staging_dirs=(),
    )
    app = create_app(settings, FakeRunner())
    base = fds()
    with (
        pytest.raises(TmpdirError, match="AGSRV_TMP_STAGING_DIRS"),
        TestClient(app),
    ):
        pytest.fail("startup must not succeed")
    assert fds() == base
    assert not app.state.jobs.started


def test_default_environment_configuration_is_fatal_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default (unset) ``AGSRV_TMP_STAGING_DIRS`` fails, naming the setting."""
    monkeypatch.delenv("AGSRV_TMP_STAGING_DIRS", raising=False)
    monkeypatch.setenv("AGSRV_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGSRV_JOBS_DB_PATH", str(tmp_path.parent / "env-j.db"))
    app = create_app()  # Settings.from_env(), no overrides
    with (
        pytest.raises(TmpdirError, match=r"workspace root .*AGSRV_TMP_STAGING_DIRS"),
        TestClient(app),
    ):
        pytest.fail("startup must not succeed")


def test_startup_rejects_unsafe_staging_and_closes_anchors(tmp_path: Path) -> None:
    """Lifespan validates trusted anchors up front and closes them on shutdown."""
    loose = tmp_path.parent / f"{tmp_path.name}-loose"
    loose.mkdir()
    loose.chmod(0o755)
    bad = create_app(
        _settings(tmp_path, tmp_staging_dirs=((tmp_path, loose),)), FakeRunner()
    )
    with pytest.raises(Exception, match="mode 0700"), TestClient(bad):
        pass  # pragma: no cover
    good = create_app(_settings(tmp_path), FakeRunner())
    with TestClient(good) as client:
        assert good.state.anchors.is_open
        assert client.get("/health").status_code == 200
    assert not good.state.anchors.is_open


def test_job_replay_and_low_space_leave_no_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing keys replay under low space; new jobs leave no rows or capacity held."""
    settings = _settings(tmp_path, tmp_min_free_bytes=1)
    app = create_app(settings, FakeRunner())
    headers = {**AUTH, "Idempotency-Key": "replay"}
    with TestClient(app) as client:
        first = client.post("/v1/jobs", headers=headers, json=CLAUDE)
        assert _wait(client, first.json()["job_id"])["status"] == "succeeded"
        monkeypatch.setattr(os, "statvfs", statvfs_stub(lambda: 0))
        assert (
            client.post("/v1/jobs", headers=headers, json=CLAUDE).json()["job_id"]
            == first.json()["job_id"]
        )
        assert app.state.anchors._live == 0  # replay released its transient fd
        mismatch = client.post(
            "/v1/jobs", headers=headers, json={**CLAUDE, "prompt": "other"}
        )
        assert mismatch.status_code == 409
        assert client.post("/v1/jobs", headers=AUTH, json=CLAUDE).status_code == 503
        with sqlite3.connect(settings.jobs_db_path) as db:
            assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert app.state.semaphore._value == settings.max_concurrency
        assert app.state.codex_semaphore._value == settings.codex_max_concurrency
        assert not app.state.jobs.workspaces
        _clean(app, tmp_path)


def test_job_queue_full_releases_admission(tmp_path: Path) -> None:
    """A rejected submission (capacity) never keeps its admission descriptor."""
    app = create_app(_settings(tmp_path, jobs_max_pending=1), FakeRunner(delay=0.5))
    with TestClient(app) as client:
        first = client.post("/v1/jobs", headers=AUTH, json=CLAUDE)
        assert first.status_code == 202
        second = client.post("/v1/jobs", headers=AUTH, json={**CLAUDE, "prompt": "b"})
        assert second.status_code == 429
        assert app.state.anchors._live == 1  # only the accepted job
        assert _wait(client, first.json()["job_id"])["status"] == "succeeded"
    assert app.state.anchors._live == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_job_cancelled_before_it_starts_releases_admission(
    tmp_path: Path,
) -> None:
    """A job task cancelled before its first step still closes its descriptor."""
    app = create_app(_settings(tmp_path), FakeRunner())
    manager = app.state.jobs
    manager.start()
    manager.submit(QueryRequest.model_validate(CLAUDE), "Bearer secret", None)
    assert app.state.anchors._live == 1
    for task in list(manager.tasks.values()):
        task.cancel()
    await manager.stop()
    assert app.state.anchors._live == 0
    await app.state.anchors.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude", "codex"])
async def test_query_rechecks_after_workspace_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str
) -> None:
    """Queued queries create no directory and reject newly exhausted space."""
    settings = _settings(
        tmp_path, tmp_min_free_bytes=1, workspace_min_free_bytes=1, max_concurrency=1
    )
    app = create_app(settings, FakeRunner())
    monkeypatch.setattr("agsrv.main._codex_available", AsyncMock(return_value=True))
    low = False
    monkeypatch.setattr(os, "statvfs", statvfs_stub(lambda: 0 if low else 100))
    monkeypatch.setattr(os, "fstatvfs", statvfs_stub(lambda: 0 if low else 100))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with app.state.jobs.workspace(tmp_path):
            task = asyncio.create_task(
                client.post(
                    "/v1/query",
                    headers=AUTH,
                    json=CLAUDE if agent == "claude" else CODEX,
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
            assert not list(tmp_path.glob(".agsrv-*"))  # ruff: ignore[blocking-path-method-in-async-function]
            low = True
        assert (await asyncio.wait_for(task, 1)).status_code == 503
        assert not app.state.jobs.workspaces
    _clean(app, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude", "codex"])
async def test_job_rechecks_after_slot_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str
) -> None:
    """A job waiting for an agent slot rechecks free space before execution."""
    app = create_app(
        _settings(tmp_path, tmp_min_free_bytes=1, workspace_min_free_bytes=0),
        FakeRunner(),
    )
    semaphore = app.state.semaphore if agent == "claude" else app.state.codex_semaphore
    for _ in range(semaphore._value):
        await semaphore.acquire()
    monkeypatch.setattr(os, "statvfs", statvfs_stub(lambda: 0))
    body = (
        QueryRequest.model_validate(CLAUDE)
        if agent == "claude"
        else CodexQueryRequest.model_validate(CODEX)
    )
    admitted = app.state.anchors.admit(tmp_path)
    task = asyncio.create_task(app.state.jobs._execute(body, admitted))
    await asyncio.sleep(0)
    assert not task.done()
    semaphore.release()
    with pytest.raises(HTTPException) as error:
        await task
    assert error.value.status_code == 503
    admitted.close()
    _clean(app, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", [False, True])
async def test_codex_detached_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disconnect: bool
) -> None:
    """A cancelled Codex turn keeps its directory and descriptors until cleanup ends."""
    app = create_app(_settings(tmp_path))
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
    monkeypatch.setattr(main.codex, "run_codex", run)
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
    assert app.state.anchors._live == 1  # still owned by the detached task
    release.set()
    for _ in range(100):
        if not main._pump_tasks:  # pyright: ignore[reportPrivateUsage]
            break
        await asyncio.sleep(0.01)
    assert not main._pump_tasks  # pyright: ignore[reportPrivateUsage]
    await asyncio.sleep(0.05)
    assert not await asyncio.to_thread(directory.exists)
    assert not app.state.jobs.workspaces
    _clean(app, tmp_path)


@pytest.mark.asyncio
async def test_codex_task_cancelled_before_start_releases_descriptor() -> None:
    """The detachable task's done-callback owns the descriptor even if never run."""
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    lease = main._SlotLease(semaphore)  # pyright: ignore[reportPrivateUsage]
    done: list[bool] = []

    async def never() -> None:  # ruff: ignore[unused-async]
        pytest.fail("must not start")  # pragma: no cover

    outer = asyncio.create_task(
        main._await_codex(never(), lease, on_done=lambda: done.append(True))  # pyright: ignore[reportPrivateUsage]
    )
    await asyncio.sleep(0)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    await asyncio.sleep(0.01)
    assert done == [True]
    assert not semaphore.locked() or lease.released


@pytest.mark.parametrize(
    ("path", "body", "mutate", "code"),
    [
        ("/v1/query", CLAUDE, {"permission_mode": "nonsense"}, 400),
        ("/v1/query", CLAUDE, {"timeout_seconds": 10**9}, 400),
        ("/v1/query", CODEX, {"timeout_seconds": 10**9}, 400),
    ],
)
def test_rejections_release_admission(
    tmp_path: Path, path: str, body: dict[str, Any], mutate: dict[str, Any], code: int
) -> None:
    """Validation failures after admission close the descriptor."""
    app = create_app(_settings(tmp_path), FakeRunner())
    with TestClient(app) as client:
        response = client.post(path, headers=AUTH, json={**body, **mutate})
        assert response.status_code in {code, 422}
        _clean(app, tmp_path)


def test_busy_claude_and_codex_release_admission(tmp_path: Path) -> None:
    """A 429 for either agent keeps nothing."""
    app = create_app(
        _settings(tmp_path, max_concurrency=1, codex_max_concurrency=1), FakeRunner()
    )

    async def fill() -> None:
        await app.state.semaphore.acquire()
        await app.state.codex_semaphore.acquire()

    asyncio.run(fill())
    with TestClient(app) as client:
        for body in (CLAUDE, CODEX):
            assert client.post("/v1/query", headers=AUTH, json=body).status_code == 429
        _clean(app, tmp_path)


def _rate_client(tmp_path: Path, runner: FakeRunner) -> tuple[TestClient, Any]:
    app = create_app(
        Settings(api_key="k", workspace_root=tmp_path, jobs_db_path=tmp_path / "j.db"),
        runner,
    )
    return TestClient(app, raise_server_exceptions=False), app


def test_rate_limits_use_private_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The /usage child gets a private TMPDIR that is removed, and is gated."""
    from claude_agent_sdk import ResultMessage  # ruff: ignore[import-outside-top-level]

    message = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=0,
        session_id="s",
        result="Current session: 5% used",
        total_cost_usd=0.0,
    )
    runner = FakeRunner(messages=[message])
    client, app = _rate_client(tmp_path, runner)
    headers = {"Authorization": "Bearer k"}
    path = "/v1/agents/claude/rate-limits"
    with client:
        assert client.get(path, headers=headers).status_code == 200
        assert runner.options is not None
        directory = Path(runner.options.env["TMPDIR"])
        assert directory.parent == tmp_path
        assert not directory.exists()
        monkeypatch.setattr(os, "statvfs", statvfs_stub(lambda: 0))
        runner.stream_calls = 0
        assert client.get(path, headers=headers).status_code == 503
        assert runner.stream_calls == 0
        _clean(app, tmp_path)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (HTTPException(503, "tmp"), 503),
        (SandboxSetupError("sandbox"), 503),
        (RuntimeError("bug"), 500),
    ],
)
def test_rate_limits_setup_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, code: int
) -> None:
    """Errors before the agent starts keep their status and release the slot."""
    monkeypatch.setattr(main, "build_options", MagicMock(side_effect=error))
    runner = FakeRunner()
    client, app = _rate_client(tmp_path, runner)
    with client:
        response = client.get(
            "/v1/agents/claude/rate-limits", headers={"Authorization": "Bearer k"}
        )
        assert response.status_code == code
        assert runner.stream_calls == 0
        _clean(app, tmp_path)


@pytest.mark.asyncio
async def test_shutdown_drains_detached_turns_before_closing_anchors(
    tmp_path: Path,
) -> None:
    """Lifespan exit waits for detached Codex work, then closes shared fds."""
    app = create_app(_settings(tmp_path), FakeRunner())
    finished = asyncio.Event()

    async def detached() -> None:
        admitted = app.state.anchors.admit(tmp_path)
        await asyncio.sleep(0.05)
        admitted.close()
        finished.set()

    async with main._lifespan(app):  # pyright: ignore[reportPrivateUsage]
        task = asyncio.create_task(detached())
        await asyncio.sleep(0)
        main._pump_tasks.add(task)  # pyright: ignore[reportPrivateUsage]
        task.add_done_callback(main._pump_tasks.discard)  # pyright: ignore[reportPrivateUsage]
    assert finished.is_set()
    assert not app.state.anchors.is_open


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({**CLAUDE, "permission_mode": "bypassPermissions"}, 403),
        ({**CLAUDE, "sandbox": True, "permission_mode": "auto"}, 403),
        ({**CLAUDE, "timeout_seconds": 10**9}, 400),
    ],
)
def test_cheap_validation_precedes_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any], code: int
) -> None:
    """Invalid requests get their 4xx even when admission would fail with 503."""
    app = create_app(_settings(tmp_path), FakeRunner())
    client = TestClient(app)
    admit = MagicMock(side_effect=HTTPException(503, "admission is broken"))
    monkeypatch.setattr(app.state.anchors, "admit", admit)
    response = client.post("/v1/query", headers=AUTH, json=body)
    assert response.status_code == code
    admit.assert_not_called()
    good = client.post("/v1/query", headers=AUTH, json=CLAUDE)
    assert good.status_code == 503  # a valid request still reaches admission
    admit.assert_called_once()


MUSE = {"agent": "muse", "prompt": "test"}


@pytest.mark.parametrize("endpoint", ["/v1/query", "/v1/jobs"])
def test_muse_payload_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    """Muse runs receive the canonical turn directory, removed afterwards."""
    received: list[str] = []

    async def run(**kwargs: Any) -> dict[str, Any]:  # ruff: ignore[any-type, unused-async]
        received.append(kwargs["tmpdir"])
        assert Path(kwargs["tmpdir"]).is_dir()  # ruff: ignore[blocking-path-method-in-async-function]
        return {
            "agent": "muse",
            "result": "ok",
            "model": None,
            "is_error": False,
            "duration_ms": 1,
            "usage": None,
            "backend": {},
        }

    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
    monkeypatch.setattr(muse, "run_muse", run)
    app = create_app(_settings(tmp_path, muse_enabled=True))
    with TestClient(app) as client:
        response = client.post(endpoint, headers=AUTH, json=MUSE)
        if endpoint == "/v1/jobs":
            assert _wait(client, response.json()["job_id"])["status"] == "succeeded"
        else:
            assert response.status_code == 200
        assert Path(received[0]).parent == tmp_path
        _clean(app, tmp_path)


@pytest.mark.parametrize("endpoint", ["/v1/query", "/v1/jobs"])
def test_muse_rejects_low_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    """Muse is preflighted like the other agents and starts nothing."""
    monkeypatch.setattr(os, "statvfs", statvfs_stub(lambda: 1))
    monkeypatch.setattr(os, "fstatvfs", statvfs_stub(lambda: 1000))
    run = AsyncMock()
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
    monkeypatch.setattr(muse, "run_muse", run)
    app = create_app(_settings(tmp_path, muse_enabled=True, tmp_min_free_bytes=100))
    with TestClient(app) as client:
        response = client.post(endpoint, headers=AUTH, json=MUSE)
        assert response.status_code == 503
        assert "1 bytes free" in response.json()["detail"]
        run.assert_not_awaited()
        _clean(app, tmp_path)


@pytest.mark.asyncio
async def test_muse_spawn_exports_tmpdir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every host gets a filtered environment; turn directories override TMPDIR."""
    for name in SANDBOX_DENIED_ENV_VARS:
        monkeypatch.setenv(name, "sentinel")
    monkeypatch.setenv("TMPDIR", "/inherited")
    monkeypatch.setenv("AGSRV_KEEP_ME", "keep")
    seen: list[dict[str, str]] = []
    client = MagicMock()
    client.close = AsyncMock()

    async def spawn(options: Any) -> MagicMock:  # ruff: ignore[any-type, unused-async]
        seen.append(options.env)
        return client

    monkeypatch.setattr(muse.MuseClient, "spawn", spawn)
    await muse._close(await muse._spawn(None, None, "/w/t"))  # pyright: ignore[reportPrivateUsage]
    await muse._close(await muse._spawn(None, None))  # pyright: ignore[reportPrivateUsage]
    assert seen[0]["TMPDIR"] == "/w/t"
    assert seen[1]["TMPDIR"] == "/inherited"
    for env in seen:
        assert set(SANDBOX_DENIED_ENV_VARS).isdisjoint(env)
        assert env["AGSRV_KEEP_ME"] == "keep"
