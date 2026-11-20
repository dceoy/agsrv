"""Tests for Codex integration without a live backend."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import Response
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from openai_codex.models import InitializeResponse
from starlette.requests import Request

from agsrv import codex_agent, main
from agsrv.config import Settings
from agsrv.main import create_app
from agsrv.schemas import CodexQueryRequest

if TYPE_CHECKING:
    import subprocess
    from pathlib import Path


class FakeClient:
    """Fake SDK context and wire client."""

    def __init__(self) -> None:
        """Create mocked operations."""
        self.thread_start = AsyncMock(
            return_value=SimpleNamespace(thread=SimpleNamespace(id="t", model="actual"))
        )
        self._start_turn = AsyncMock(
            return_value=(SimpleNamespace(turn=SimpleNamespace(id="turn")), object())
        )
        self.request = AsyncMock()
        self._client = self
        self._sync = self
        self.start = MagicMock()
        self._proc = None
        self.initialize = AsyncMock(return_value=InitializeResponse(userAgent="test/1"))

    def close(self) -> None:
        """Close the fake client."""


def test_client_declines() -> None:
    """The real SDK client installs the fail-closed handler."""
    client = codex_agent.make_client(None)
    handler = client._client._sync._approval_handler  # pyright: ignore[reportPrivateUsage]
    assert handler("anything", None) == {"decision": "decline"}


@pytest.mark.asyncio
async def test_probe_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize and normalize rate limits, preserving raw data."""
    client = FakeClient()
    monkeypatch.setattr(codex_agent, "make_client", MagicMock(return_value=client))
    assert await codex_agent.probe(None)
    client.request.return_value = SimpleNamespace(
        model_dump=MagicMock(
            return_value={
                "rateLimits": {
                    "planType": "plus",
                    "primary": {"usedPercent": 12, "resetsAt": 123},
                    "secondary": None,
                    "rateLimitReachedType": "x",
                },
                "rateLimitResetCredits": {"availableCount": 1},
            }
        )
    )
    result = await codex_agent.read_rate_limits(None)
    assert result["primary"] == {"used_percent": 12, "resets_at": 123}
    assert result["secondary"] is None
    assert result["plan"] == "plus"
    assert result["reached_type"] == "x"
    assert result["reset_credits"] == {"availableCount": 1}
    client.request.assert_awaited_once()

    def fail(_: object) -> NoReturn:
        raise RuntimeError

    monkeypatch.setattr(codex_agent, "make_client", fail)
    assert not await codex_agent.probe(None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("network", "sandbox"),
    [(False, "read-only"), (True, "read-only"), (True, "workspace-write")],
)
async def test_run(
    monkeypatch: pytest.MonkeyPatch, network: bool, sandbox: str
) -> None:
    """Every turn is ephemeral and denies approvals with the requested sandbox."""
    client = FakeClient()
    monkeypatch.setattr(codex_agent, "make_client", MagicMock(return_value=client))
    item = MagicMock()
    item.model_dump.return_value = {"type": "commandExecution", "command": "pwd"}
    result = SimpleNamespace(
        items=[item],
        final_response="done",
        error=None,
        duration_ms=10,
        usage=None,
        id="turn",
        status=SimpleNamespace(value="completed"),
    )
    run = AsyncMock(return_value=result)
    monkeypatch.setattr(
        codex_agent, "AsyncTurnHandle", MagicMock(return_value=SimpleNamespace(run=run))
    )
    monkeypatch.setattr(
        codex_agent, "AsyncThread", MagicMock(return_value=SimpleNamespace(run=run))
    )
    response = await codex_agent.run_codex(
        prompt="hi",
        cwd="/workspace",
        model="requested",
        effort=None,
        sandbox=sandbox,
        network_access=network,
        binary=None,
    )
    assert response["model"] == "actual"
    assert response["backend"]["commands"] == [
        {"type": "commandExecution", "command": "pwd"}
    ]
    assert client.thread_start.call_args.args[0]["ephemeral"] is True
    assert client.thread_start.call_args.args[0]["approvalPolicy"] == "never"
    if network:
        policy = client._start_turn.call_args.args[2]["sandboxPolicy"]  # pyright: ignore[reportPrivateUsage]
        assert policy == {
            "type": "readOnly" if sandbox == "read-only" else "workspaceWrite",
            "networkAccess": True,
        }
    else:
        assert run.call_args.kwargs["approval_mode"].value == "deny_all"


def test_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise routing, authentication, validation, and backend failure mapping."""
    probe = AsyncMock(return_value=True)
    run = AsyncMock(
        return_value={
            "agent": "codex",
            "result": "ok",
            "model": None,
            "is_error": False,
            "duration_ms": 0,
            "usage": None,
            "backend": {},
        }
    )
    limits = AsyncMock(return_value={"plan": "plus"})
    monkeypatch.setattr(codex_agent, "probe", probe)
    monkeypatch.setattr(codex_agent, "run_codex", run)
    monkeypatch.setattr(codex_agent, "read_rate_limits", limits)
    app = create_app(
        Settings(api_key="key", workspace_root=tmp_path, max_timeout_seconds=2)
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer key"}
    body = {"agent": "codex", "prompt": "hi"}
    assert client.get("/v1/agents").status_code == 401
    assert client.get("/v1/agents", headers=headers).json()["codex"]["healthy"] is True
    assert client.get("/v1/agents", headers=headers).json()["claude"]["healthy"] is True
    probe.assert_awaited_once()
    assert client.get("/v1/agents/codex/rate-limits", headers=headers).json() == {
        "plan": "plus"
    }
    assert client.post("/v1/query", headers=headers, json=body).json()["result"] == "ok"
    assert run.call_args.kwargs["sandbox"] == "read-only"
    assert (
        client.post("/v1/query", headers=headers, json={"prompt": "hi"}).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/query", headers=headers, json={**body, "agent": "other"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/query",
            headers=headers,
            json={**body, "options": {"sandbox": "full-access"}},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/query", headers=headers, json={**body, "cwd": "../outside"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/v1/query", headers=headers, json={**body, "timeout_seconds": 3}
        ).status_code
        == 400
    )
    app.state.codex_semaphore = asyncio.Semaphore(0)
    assert client.post("/v1/query", headers=headers, json=body).status_code == 429
    app.state.codex_semaphore = asyncio.Semaphore(1)
    run.side_effect = TimeoutError()
    assert client.post("/v1/query", headers=headers, json=body).status_code == 504
    run.side_effect = RuntimeError()
    assert client.post("/v1/query", headers=headers, json=body).status_code == 502
    limits.side_effect = RuntimeError()
    assert (
        client.get("/v1/agents/codex/rate-limits", headers=headers).status_code == 502
    )
    app.state.settings = Settings(
        api_key="key", workspace_root=tmp_path, codex_enabled=False
    )
    assert client.get("/v1/agents", headers=headers).json()["codex"]["healthy"] is False
    assert client.post("/v1/query", headers=headers, json=body).status_code == 503
    assert (
        client.get("/v1/agents/codex/rate-limits", headers=headers).status_code == 503
    )
    app.state.settings = Settings(
        api_key="key", workspace_root=tmp_path, codex_bin=str(tmp_path / "missing")
    )
    monkeypatch.undo()
    app.state.codex_probe_expires = 0
    assert client.get("/v1/agents", headers=headers).json()["codex"]["healthy"] is False


@pytest.mark.asyncio
async def test_probe_coalesces_and_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent checks share one bounded probe."""
    app = create_app(Settings(workspace_root=tmp_path))
    started = asyncio.Event()
    release = asyncio.Event()

    async def probe(_binary: str | None) -> bool:
        started.set()
        await release.wait()
        return True

    mock = AsyncMock(side_effect=probe)
    monkeypatch.setattr(codex_agent, "probe", mock)
    first = asyncio.create_task(main._codex_available(app))  # pyright: ignore[reportPrivateUsage]
    await started.wait()
    second = asyncio.create_task(main._codex_available(app))  # pyright: ignore[reportPrivateUsage]
    release.set()
    assert await asyncio.gather(first, second) == [True, True]
    mock.assert_awaited_once()
    app.state.codex_probe_expires = 0
    release.clear()
    monkeypatch.setattr(main, "CODEX_PROBE_TIMEOUT", 0.01)
    assert not await main._codex_available(app)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_cached_agents_with_busy_executor_and_probe_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached health responds while the executor and probe lock remain occupied."""
    app = create_app(
        Settings(
            api_key="key", workspace_root=tmp_path, codex_bin=str(tmp_path / "codex")
        )
    )
    probe = AsyncMock(return_value=True)
    monkeypatch.setattr(codex_agent, "probe", probe)
    monkeypatch.setattr(main, "CODEX_PROBE_TIMEOUT", 0.05)
    release = threading.Event()
    occupied = threading.Event()

    def block() -> None:
        occupied.set()
        release.wait()

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop.set_default_executor(executor)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer key"},
        ) as client:
            assert (await client.get("/v1/agents")).json()["codex"]["healthy"] is True
            loop.run_in_executor(None, block)
            assert occupied.wait(1)
            try:
                async with app.state.codex_probe_lock:
                    response = await asyncio.wait_for(client.get("/v1/agents"), 0.5)
                    assert response.json()["codex"]["healthy"] is True
                    assert not release.is_set()
                probe.assert_awaited_once()
            finally:
                release.set()


@pytest.mark.asyncio
async def test_codex_deadline_and_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probe time counts toward the deadline and disconnect cancels the run."""
    app = create_app(Settings(workspace_root=tmp_path, codex_max_concurrency=1))
    started = asyncio.Event()
    closed = asyncio.Event()
    disconnected = False

    async def receive() -> dict[str, str]:
        if disconnected:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def slow(*_args: object, **_kwargs: object) -> bool:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
        return True

    request = Request({"type": "http", "app": app}, receive)
    monkeypatch.setattr(codex_agent, "probe", slow)
    with pytest.raises(main.HTTPException) as error:
        await main._codex_query(  # pyright: ignore[reportPrivateUsage]
            CodexQueryRequest(agent="codex", prompt="x", timeout_seconds=0.01), request
        )
    assert error.value.status_code == 504
    assert closed.is_set()
    started.clear()
    closed.clear()
    monkeypatch.setattr(main, "DISCONNECT_POLL_INTERVAL", 0.01)
    task = asyncio.create_task(
        main._codex_query_with_disconnect(  # pyright: ignore[reportPrivateUsage]
            CodexQueryRequest(agent="codex", prompt="x"), request
        )
    )
    await started.wait()
    disconnected = True
    response = await asyncio.wait_for(task, 1)
    assert isinstance(response, Response)
    assert response.status_code == 499
    assert closed.is_set()
    assert not app.state.codex_semaphore.locked()
    disconnected = False
    started.clear()
    task = asyncio.create_task(
        main._codex_query_with_disconnect(  # pyright: ignore[reportPrivateUsage]
            CodexQueryRequest(agent="codex", prompt="x"), request
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancel_during_transport_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation waits for the SDK start worker before closing its transport."""
    client = codex_agent.make_client(None)
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    finish_close = threading.Event()

    def start() -> None:
        started.set()
        release.wait()

    def close() -> None:
        closed.set()
        finish_close.wait()

    monkeypatch.setattr(client._client._sync, "start", start)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(client._client._sync, "close", close)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(
        client._client,  # pyright: ignore[reportPrivateUsage]
        "initialize",
        AsyncMock(side_effect=RuntimeError()),
    )
    monkeypatch.setattr(codex_agent, "make_client", MagicMock(return_value=client))
    app = create_app(Settings())
    app.state.codex_semaphore = asyncio.Semaphore(1)

    async def operation() -> None:
        async with main._codex_slot(app):  # pyright: ignore[reportPrivateUsage]
            await codex_agent.probe(None)

    task = asyncio.create_task(operation())
    await asyncio.to_thread(started.wait)
    task.cancel()
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    assert app.state.codex_semaphore.locked()
    assert not closed.is_set()
    release.set()
    assert await asyncio.to_thread(closed.wait, 1)
    assert app.state.codex_semaphore.locked()
    assert not task.done()
    finish_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert not app.state.codex_semaphore.locked()


@pytest.mark.asyncio
async def test_stalled_initialize_closed_on_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the transport unblocks an initialize worker after a deadline."""
    client = codex_agent.make_client(None)
    started = threading.Event()
    closed = threading.Event()
    finished = threading.Event()

    def initialize() -> InitializeResponse:
        started.set()
        closed.wait()
        finished.set()
        return InitializeResponse(userAgent="test/1")

    monkeypatch.setattr(client._client._sync, "start", lambda: None)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(client._client._sync, "initialize", initialize)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(client._client._sync, "close", closed.set)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(codex_agent, "make_client", MagicMock(return_value=client))
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.1):
            await codex_agent.probe(None)
    assert started.is_set()
    assert closed.is_set()
    assert await asyncio.to_thread(finished.wait, 1)


def test_saturated_executor_closes_real_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled handshake cannot queue its own shutdown behind itself."""
    binary = tmp_path / "silent-codex"
    handshake = tmp_path / "handshake"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import sys, time\n"
        "sys.stdin.readline()\n"
        f"open({str(handshake)!r}, 'w').close()\n"
        "time.sleep(30)\n"
    )
    binary.chmod(0o755)
    client = codex_agent.make_client(str(binary))
    sync = client._client._sync  # pyright: ignore[reportPrivateUsage]
    close = sync.close
    processes: list[subprocess.Popen[str] | None] = []

    def record_close() -> None:
        processes.append(sync._proc)  # pyright: ignore[reportPrivateUsage]
        close()

    monkeypatch.setattr(sync, "close", record_close)
    monkeypatch.setattr(codex_agent, "make_client", MagicMock(return_value=client))

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop.set_default_executor(executor)
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.2):
                    await codex_agent.probe(str(binary))
            assert time.monotonic() - started < 3
            assert handshake.exists()
            assert len(processes) == 1
            assert processes[0] is not None
            assert processes[0].poll() is not None

    asyncio.run(exercise())


def test_metadata_capacity_timeout_and_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata starts require capacity; cached health and OpenAPI remain available."""
    app = create_app(
        Settings(workspace_root=tmp_path, max_timeout_seconds=0.01, api_key="key")
    )
    probe = AsyncMock(return_value=True)

    async def slow_limits(_binary: str | None) -> dict[str, object]:
        await asyncio.Event().wait()
        return {}

    limits = AsyncMock(side_effect=slow_limits)
    monkeypatch.setattr(codex_agent, "probe", probe)
    monkeypatch.setattr(codex_agent, "read_rate_limits", limits)
    client = TestClient(app, headers={"Authorization": "Bearer key"})
    app.state.codex_semaphore = asyncio.Semaphore(0)
    assert client.get("/v1/agents").status_code == 429
    assert client.get("/v1/agents/codex/rate-limits").status_code == 429
    probe.assert_not_awaited()
    app.state.codex_semaphore = asyncio.Semaphore(1)
    assert client.get("/v1/agents").status_code == 200
    app.state.codex_semaphore = asyncio.Semaphore(0)
    assert client.get("/v1/agents").json()["codex"]["healthy"] is True
    app.state.codex_semaphore = asyncio.Semaphore(1)
    assert client.get("/v1/agents/codex/rate-limits").status_code == 504
    schema = client.get("/openapi.json").json()["paths"]["/v1/query"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    assert schema["discriminator"]["propertyName"] == "agent"
    assert len(schema["oneOf"]) == 2


def test_start_with_busy_default_executor(tmp_path: Path) -> None:
    """Startup does not wait behind unrelated default executor work."""
    binary = tmp_path / "silent-codex"
    marker = tmp_path / "started"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import time\n"
        f"open({str(marker)!r}, 'w').close()\n"
        "time.sleep(30)\n"
    )
    binary.chmod(0o755)
    release = threading.Event()
    occupied = threading.Event()

    def block() -> None:
        occupied.set()
        release.wait()

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop.set_default_executor(executor)
            loop.run_in_executor(None, block)
            assert occupied.wait(1)
            try:
                started = time.monotonic()
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(0.2):
                        await codex_agent.probe(str(binary))
                assert time.monotonic() - started < 1
                assert marker.exists()
            finally:
                release.set()

    asyncio.run(exercise())


def test_blocked_writer_child_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full stdin pipe cannot prevent the transport from reaping the child."""
    binary = tmp_path / "blocked-codex"
    initialized = tmp_path / "initialized"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        "request = json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'id': request['id'], 'result': "
        "{'userAgent': 'test/1'}}), flush=True)\n"
        "sys.stdin.readline()\n"
        f"open({str(initialized)!r}, 'w').close()\n"
        "time.sleep(30)\n"
    )
    binary.chmod(0o755)
    client = codex_agent.make_client(str(binary))
    sync = client._client._sync  # pyright: ignore[reportPrivateUsage]
    close = sync.close
    processes: list[subprocess.Popen[str] | None] = []

    def record_close() -> None:
        processes.append(sync._proc)  # pyright: ignore[reportPrivateUsage]
        close()

    monkeypatch.setattr(sync, "close", record_close)
    monkeypatch.setattr(codex_agent, "make_client", MagicMock(return_value=client))

    async def exercise() -> None:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.2):
                await codex_agent.run_codex(
                    prompt="x",
                    cwd=str(tmp_path),
                    model="x" * 1_000_000,
                    effort=None,
                    sandbox="read-only",
                    network_access=False,
                    binary=str(binary),
                )
        assert time.monotonic() - started < 2
        assert initialized.exists()
        assert len(processes) == 1
        assert processes[0] is not None
        assert processes[0].poll() is not None

    asyncio.run(exercise())
