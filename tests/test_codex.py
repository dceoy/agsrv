"""Tests for Codex integration without a live backend."""

from __future__ import annotations

import asyncio
import subprocess  # ruff: ignore[suspicious-subprocess-import]
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
from openai_codex.generated.v2_all import (
    ItemCompletedNotification,
    TurnCompletedNotification,
    TurnStatus,
)
from openai_codex.models import InitializeResponse, Notification
from starlette.requests import Request

from agsrv import codex, main
from agsrv.config import Settings
from agsrv.main import create_app
from agsrv.schemas import CodexQueryRequest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
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
    client = codex.make_client(None)
    handler = client._client._sync._approval_handler  # pyright: ignore[reportPrivateUsage]
    assert handler("anything", None) == {"decision": "decline"}


@pytest.mark.asyncio
async def test_probe_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize and normalize rate limits, preserving raw data."""
    client = FakeClient()
    monkeypatch.setattr(codex, "make_client", MagicMock(return_value=client))
    assert await codex.probe(None)
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
    result = await codex.read_rate_limits(None)
    assert result["primary"] == {"used_percent": 12, "resets_at": 123}
    assert result["secondary"] is None
    assert result["plan"] == "plus"
    assert result["reached_type"] == "x"
    assert result["reset_credits"] == {"availableCount": 1}
    client.request.assert_awaited_once()

    def fail(_: object) -> NoReturn:
        raise RuntimeError

    monkeypatch.setattr(codex, "make_client", fail)
    assert not await codex.probe(None)


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
    monkeypatch.setattr(codex, "make_client", MagicMock(return_value=client))
    item = MagicMock()
    item.model_dump.return_value = {"type": "commandExecution", "command": "pwd"}
    result = SimpleNamespace(
        items=[item],
        final_response="done",
        error=None,
        duration_ms=10,
        usage=None,
        id="turn",
        status=TurnStatus.completed,
    )
    run = AsyncMock(return_value=result)
    monkeypatch.setattr(codex, "collect_turn", run)
    monkeypatch.setattr(
        codex, "AsyncTurnHandle", MagicMock(return_value=SimpleNamespace())
    )
    thread_turn = AsyncMock()
    monkeypatch.setattr(
        codex,
        "AsyncThread",
        MagicMock(return_value=SimpleNamespace(turn=thread_turn)),
    )
    response = await codex.run_codex(
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
        assert thread_turn.call_args.kwargs["approval_mode"].value == "deny_all"


@pytest.mark.parametrize("network", [False, True])
@pytest.mark.parametrize(
    "status", ["failed", "interrupted", "completed", "missing", "completed_error"]
)
def test_terminal_results_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, network: bool, status: str
) -> None:
    """Typed SDK events preserve terminal results through the HTTP endpoint."""
    client = FakeClient()
    monkeypatch.setattr(codex, "make_client", MagicMock(return_value=client))
    monkeypatch.setattr(codex, "probe", AsyncMock(return_value=True))
    item = ItemCompletedNotification.model_validate({
        "threadId": "t",
        "turnId": "turn",
        "completedAtMs": 1,
        "item": {
            "type": "commandExecution",
            "id": "c",
            "command": "pwd",
            "cwd": "/workspace",
            "commandActions": [],
            "status": "completed",
        },
    })
    terminal = TurnCompletedNotification.model_validate({
        "threadId": "t",
        "turn": {
            "id": "turn",
            "items": [],
            "status": "completed"
            if status in {"missing", "completed_error"}
            else status,
            "error": {"message": "failed"}
            if status in {"failed", "completed_error"}
            else None,
        },
    })

    async def stream() -> AsyncGenerator[Notification, None]:  # ruff: ignore[unused-async]
        yield Notification("item/completed", item)
        if status != "missing":
            yield Notification("turn/completed", terminal)

    handle = SimpleNamespace(id="turn", stream=stream)
    monkeypatch.setattr(codex, "AsyncTurnHandle", MagicMock(return_value=handle))
    monkeypatch.setattr(
        codex,
        "AsyncThread",
        MagicMock(return_value=SimpleNamespace(turn=AsyncMock(return_value=handle))),
    )
    app = create_app(Settings(api_key="key", workspace_root=tmp_path))
    response = TestClient(app).post(
        "/v1/query",
        headers={"Authorization": "Bearer key"},
        json={"agent": "codex", "prompt": "hi", "options": {"network_access": network}},
    )
    if status == "missing":
        assert response.status_code == 502
        return
    assert response.status_code == 200
    result = response.json()
    assert result["is_error"] is (status != "completed")
    assert result["model"] == "actual"
    assert result["usage"] is None
    assert result["backend"]["id"] == "turn"
    assert result["backend"]["status"] == (
        "completed" if status == "completed_error" else status
    )
    assert result["backend"]["commands"][0]["command"] == "pwd"


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
    monkeypatch.setattr(codex, "probe", probe)
    monkeypatch.setattr(codex, "run_codex", run)
    monkeypatch.setattr(codex, "read_rate_limits", limits)
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
    monkeypatch.setattr(codex, "probe", mock)
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


async def assert_probe_cache_after_cleanup(
    http: AsyncClient, monkeypatch: pytest.MonkeyPatch, *, query_deadline: bool
) -> None:
    """Only a caller-interrupted probe is retried after cleanup."""
    healthy_probe = AsyncMock(return_value=True)
    monkeypatch.setattr(codex, "probe", healthy_probe)
    health = await http.get("/v1/agents")
    assert health.json()["codex"]["healthy"] is query_deadline
    assert healthy_probe.await_count == int(query_deadline)


@pytest.mark.asyncio
@pytest.mark.parametrize("query_deadline", [False, True])
async def test_probe_timeout_releases_waiters_but_keeps_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, query_deadline: bool
) -> None:
    """A blocked SDK start retains capacity until its late child is closed."""
    app = create_app(
        Settings(workspace_root=tmp_path, codex_max_concurrency=1, api_key="key")
    )
    client = codex.make_client(None)
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    processes: list[subprocess.Popen[str]] = []
    sync = client._client._sync  # pyright: ignore[reportPrivateUsage]

    def start() -> None:
        started.set()
        release.wait()
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], text=True
        )
        processes.append(process)
        sync._proc = process  # pyright: ignore[reportPrivateUsage]

    def close() -> None:
        processes[0].wait(timeout=1)
        closed.set()

    monkeypatch.setattr(client._client._sync, "start", start)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(client._client._sync, "close", close)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(codex, "make_client", MagicMock(return_value=client))
    monkeypatch.setattr(main, "CODEX_PROBE_TIMEOUT", 0.05)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer key"},
    ) as http:
        if query_deadline:
            first = asyncio.create_task(
                http.post(
                    "/v1/query",
                    json={"agent": "codex", "prompt": "x", "timeout_seconds": 0.02},
                )
            )
        else:
            first = asyncio.create_task(http.get("/v1/agents"))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            second = asyncio.create_task(http.get("/v1/agents"))
            first_response, second_response = await asyncio.wait_for(
                asyncio.gather(first, second), 0.5
            )
            if query_deadline:
                assert first_response.status_code == 504
            else:
                assert not first_response.json()["codex"]["healthy"]
            assert not second_response.json()["codex"]["healthy"]
            assert not release.is_set()
            assert app.state.codex_semaphore.locked()
            assert (await http.get("/v1/agents/codex/rate-limits")).status_code == 429
        finally:
            release.set()
        assert await asyncio.to_thread(closed.wait, 1)
        assert processes[0].poll() is not None
        for _ in range(100):
            if not app.state.codex_semaphore.locked():
                break
            await asyncio.sleep(0.01)
        assert app.state.codex_semaphore._value == 1  # pyright: ignore[reportPrivateUsage]
        assert not main._pump_tasks  # pyright: ignore[reportPrivateUsage]
        await assert_probe_cache_after_cleanup(
            http, monkeypatch, query_deadline=query_deadline
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [False, True])
async def test_warm_cache_start_timeout_keeps_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, query: bool
) -> None:
    """Both operations return at their deadline while late startup is reaped."""
    app = create_app(
        Settings(
            workspace_root=tmp_path,
            codex_max_concurrency=1,
            max_timeout_seconds=0.05,
            api_key="key",
        )
    )
    app.state.codex_healthy = True
    app.state.codex_probe_expires = time.monotonic() + 30
    client = codex.make_client(None)
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    processes: list[subprocess.Popen[str]] = []
    sync = client._client._sync  # pyright: ignore[reportPrivateUsage]

    def start() -> None:
        started.set()
        release.wait()
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], text=True
        )
        processes.append(process)
        sync._proc = process  # pyright: ignore[reportPrivateUsage]

    def close() -> None:
        processes[0].wait(timeout=1)
        closed.set()

    monkeypatch.setattr(sync, "start", start)
    monkeypatch.setattr(sync, "close", close)
    monkeypatch.setattr(codex, "make_client", MagicMock(return_value=client))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer key"},
    ) as http:
        if query:
            request = http.post("/v1/query", json={"agent": "codex", "prompt": "x"})
        else:
            request = http.get("/v1/agents/codex/rate-limits")
        first = asyncio.create_task(request)
        try:
            assert await asyncio.to_thread(started.wait, 1)
            response = await asyncio.wait_for(first, 0.25)
            assert response.status_code == 504
            assert not release.is_set()
            assert app.state.codex_semaphore.locked()
            assert (await http.get("/v1/agents/codex/rate-limits")).status_code == 429
        finally:
            release.set()
        assert await asyncio.to_thread(closed.wait, 1)
        assert processes[0].poll() is not None
        for _ in range(100):
            if not app.state.codex_semaphore.locked():
                break
            await asyncio.sleep(0.01)
        assert app.state.codex_semaphore._value == 1  # pyright: ignore[reportPrivateUsage]
        assert not main._pump_tasks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_cache", [False, True])
async def test_cancelled_probe_consumes_late_start_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, newer_cache: bool
) -> None:
    """A late startup failure is consumed without an orphaned future exception."""
    app = create_app(Settings(workspace_root=tmp_path))
    client = codex.make_client(None)
    started = threading.Event()
    release = threading.Event()
    errors: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()

    def start() -> NoReturn:
        started.set()
        release.wait()
        raise FileNotFoundError

    monkeypatch.setattr(client._client._sync, "start", start)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(codex, "make_client", MagicMock(return_value=client))
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    task = asyncio.create_task(main._codex_available(app))  # pyright: ignore[reportPrivateUsage]
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        if newer_cache:
            app.state.codex_probe_expires += 30
        expiry = app.state.codex_probe_expires
        release.set()
        for _ in range(100):
            if not main._pump_tasks:  # pyright: ignore[reportPrivateUsage]
                break
            await asyncio.sleep(0.01)
        assert not main._pump_tasks  # pyright: ignore[reportPrivateUsage]
        assert not errors
        assert app.state.codex_probe_expires == (expiry if newer_cache else 0)
    finally:
        release.set()
        loop.set_exception_handler(previous_handler)


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
    monkeypatch.setattr(codex, "probe", probe)
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
    monkeypatch.setattr(codex, "probe", slow)
    with pytest.raises(main.HTTPException) as error:
        await main._codex_query(  # pyright: ignore[reportPrivateUsage]
            CodexQueryRequest(agent="codex", prompt="x", timeout_seconds=0.01), request
        )
    assert error.value.status_code == 504
    await asyncio.wait_for(closed.wait(), 1)
    await asyncio.sleep(0)
    app.state.codex_probe_expires = 0
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
    await asyncio.wait_for(closed.wait(), 1)
    await asyncio.sleep(0)
    assert not app.state.codex_semaphore.locked()
    app.state.codex_probe_expires = 0
    disconnected = False
    request = Request({"type": "http", "app": app}, receive)
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
    client = codex.make_client(None)
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
    monkeypatch.setattr(codex, "make_client", MagicMock(return_value=client))
    app = create_app(Settings())
    app.state.codex_semaphore = asyncio.Semaphore(1)

    async def operation() -> None:
        async with main._codex_slot(app):  # pyright: ignore[reportPrivateUsage]
            await codex.probe(None)

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
    client = codex.make_client(None)
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
    monkeypatch.setattr(codex, "make_client", MagicMock(return_value=client))
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.1):
            await codex.probe(None)
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
    client = codex.make_client(str(binary))
    sync = client._client._sync  # pyright: ignore[reportPrivateUsage]
    close = sync.close
    processes: list[subprocess.Popen[str] | None] = []

    def record_close() -> None:
        processes.append(sync._proc)  # pyright: ignore[reportPrivateUsage]
        close()

    monkeypatch.setattr(sync, "close", record_close)
    monkeypatch.setattr(codex, "make_client", MagicMock(return_value=client))

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop.set_default_executor(executor)
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.2):
                    await codex.probe(str(binary))
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
    monkeypatch.setattr(codex, "probe", probe)
    monkeypatch.setattr(codex, "read_rate_limits", limits)
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
                        await codex.probe(str(binary))
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
    client = codex.make_client(str(binary))
    sync = client._client._sync  # pyright: ignore[reportPrivateUsage]
    close = sync.close
    processes: list[subprocess.Popen[str] | None] = []

    def record_close() -> None:
        processes.append(sync._proc)  # pyright: ignore[reportPrivateUsage]
        close()

    monkeypatch.setattr(sync, "close", record_close)
    monkeypatch.setattr(codex, "make_client", MagicMock(return_value=client))

    async def exercise() -> None:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.2):
                await codex.run_codex(
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
