"""Tests for Muse integration without a live backend."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from agsrv import main, muse_agent
from agsrv.config import Settings
from agsrv.main import (
    create_app,
)
from agsrv.schemas import MuseQueryRequest


def _choice(choice_id: str, decision: str) -> dict[str, str]:
    """Build an approval choice."""
    return {
        "choiceId": choice_id,
        "decision": decision,
        "label": choice_id,
        "scope": "once",
    }


def _request(choices: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Build an approval request."""
    return {"availableChoices": choices}


def test_serve_args_fixed() -> None:
    """Serve args are fixed and safe."""
    args = muse_agent.serve_args()
    assert args[0] == "serve"
    assert "--no-session-log" in args
    assert "--contributor-mode" in args
    assert "off" in args
    assert "--disable-sandbox" not in args
    assert "--trust-workspace" not in args
    assert muse_agent.APPROVAL_MODE in {"onRequest", "promptUnmatched", "denyUnmatched"}


def test_deny_approval_picks_denied() -> None:
    """Deny handler picks denied decision."""
    req: dict[str, list[dict[str, str]]] = _request([
        _choice("allow", "approved"),
        _choice("deny-1", "denied"),
    ])
    decision = muse_agent.deny_approval(req)  # pyright: ignore[reportArgumentType,reportUnknownArgumentType,reportUnknownVariableType]
    assert decision.choice_id == "deny-1"
    assert decision.choice_id != "allow"


def test_deny_approval_label_fallback() -> None:
    """Deny handler uses label fallback."""
    req: dict[str, list[dict[str, str]]] = _request([
        {"choiceId": "x", "decision": "approved", "label": "Deny it", "scope": "once"},
    ])
    decision = muse_agent.deny_approval(req)  # pyright: ignore[reportArgumentType,reportUnknownArgumentType,reportUnknownVariableType]
    assert decision.choice_id == "x"


def test_deny_approval_raises_without_denied() -> None:
    """Deny handler fails closed without denied choice."""
    req: dict[str, list[dict[str, str]]] = _request([
        _choice("a", "approved"),
        _choice("b", "approvedForSession"),
    ])
    with pytest.raises(RuntimeError):
        muse_agent.deny_approval(req)  # pyright: ignore[reportArgumentType]


def test_deny_approval_empty_request() -> None:
    """Deny handler raises on empty choices."""
    with pytest.raises(RuntimeError):
        muse_agent.deny_approval({"availableChoices": []})  # pyright: ignore[reportArgumentType,reportUnknownArgumentType]


def test_is_ephemeral() -> None:
    """Ephemerality check discriminates kinds."""
    client = MagicMock()
    client.durability = SimpleNamespace(kind="ephemeral")
    assert muse_agent._is_ephemeral(client)  # pyright: ignore[reportPrivateUsage]
    client.durability = SimpleNamespace(kind="durable")
    assert not muse_agent._is_ephemeral(client)  # pyright: ignore[reportPrivateUsage]
    client.durability = SimpleNamespace(kind="unrecognized")
    assert not muse_agent._is_ephemeral(client)  # pyright: ignore[reportPrivateUsage]


def test_usage_to_dict() -> None:
    """Usage mapping covers None and objects."""
    assert muse_agent._usage_to_dict(None) is None  # pyright: ignore[reportPrivateUsage]
    assert muse_agent._usage_to_dict({"a": 1}) == {"a": 1}  # pyright: ignore[reportPrivateUsage]
    obj = SimpleNamespace(
        inputTokens=1, outputTokens=2, cachedTokens=3, reasoningTokens=4
    )
    result = muse_agent._usage_to_dict(obj)  # pyright: ignore[reportPrivateUsage,reportUnknownVariableType]
    assert result == {
        "input_tokens": 1,
        "output_tokens": 2,
        "cached_tokens": 3,
        "reasoning_tokens": 4,
    }

    class Dump:
        """Dump stub."""

        def model_dump(self, mode: str = "json") -> dict[str, bool]:
            """Return dumped data."""
            _ = mode
            return {"dumped": True}

    assert muse_agent._usage_to_dict(Dump()) == {"dumped": True}  # pyright: ignore[reportPrivateUsage,reportUnknownVariableType]


def test_outcome_str() -> None:
    """Outcome coercion handles enums and strings."""
    assert muse_agent._outcome_str(SimpleNamespace(value="Completed")) == "completed"  # pyright: ignore[reportPrivateUsage]
    assert muse_agent._outcome_str(SimpleNamespace(name="FAILED")) == "failed"  # pyright: ignore[reportPrivateUsage]
    assert muse_agent._outcome_str("cancelled") == "cancelled"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_probe_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe reports ephemeral health."""

    class FakeClient:
        """Fake client with fixed durability."""

        def __init__(self, kind: str) -> None:
            self._kind = kind

        @property
        def durability(self) -> SimpleNamespace:
            """Durability stub."""
            return SimpleNamespace(kind=self._kind)

        async def close(self) -> None:
            """Close stub."""
            return None

    async def _spawn_ephemeral_async(_: object) -> FakeClient:  # pyright: ignore[reportUnusedFunction]
        await asyncio.sleep(0)
        return FakeClient("ephemeral")

    async def _spawn_durable_async(_: object) -> FakeClient:  # pyright: ignore[reportUnusedFunction]
        await asyncio.sleep(0)
        return FakeClient("durable")

    async def _fail(_: object) -> FakeClient:
        await asyncio.sleep(0)
        msg = "no binary"
        raise RuntimeError(msg)

    monkeypatch.setattr(muse_agent.MuseClient, "spawn", _spawn_ephemeral_async)
    assert await muse_agent.probe(None) is True
    monkeypatch.setattr(muse_agent.MuseClient, "spawn", _spawn_durable_async)
    assert await muse_agent.probe(None) is False
    monkeypatch.setattr(muse_agent.MuseClient, "spawn", _fail)
    assert await muse_agent.probe(None) is False


@pytest.mark.asyncio
async def test_run_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Completed turn maps to success response."""
    result = SimpleNamespace(
        outcome="completed",
        final_text="done",
        turn_id="t1",
        usage={"input_tokens": 1, "output_tokens": 2},
    )
    client = MagicMock()
    client.durability = SimpleNamespace(kind="ephemeral")
    session = MagicMock()
    session.run = AsyncMock(return_value=result)
    client.start_session = AsyncMock(return_value=session)
    client.close = AsyncMock()

    async def _spawn(_: object) -> MagicMock:
        await asyncio.sleep(0)
        return client

    monkeypatch.setattr(muse_agent.MuseClient, "spawn", _spawn)
    response = await muse_agent.run_muse(
        prompt="hi", cwd="/workspace", model="m1", effort="low", binary=None
    )
    assert response["result"] == "done"
    assert response["is_error"] is False
    assert response["model"] == "m1"
    assert response["backend"] == {"turn_id": "t1", "outcome": "completed"}
    assert response["usage"] == {"input_tokens": 1, "output_tokens": 2}
    client.start_session.assert_awaited_once()
    options = client.start_session.call_args.args[0]
    assert options.approval_mode == muse_agent.APPROVAL_MODE
    assert options.workspace_root == "/workspace"
    assert options.model_id == "m1"
    session.on_approval.assert_called_once()
    assert session.on_approval.call_args.args[0] is muse_agent.deny_approval


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
async def test_run_terminal_errors_are_is_error(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """Failed or cancelled turns map to is_error."""
    result = SimpleNamespace(
        outcome=outcome,
        final_text="partial",
        turn_id="t2",
        usage=None,
    )
    client = MagicMock()
    client.durability = SimpleNamespace(kind="ephemeral")
    session = MagicMock()
    session.run = AsyncMock(return_value=result)
    client.start_session = AsyncMock(return_value=session)
    client.close = AsyncMock()

    async def _spawn(_: object) -> MagicMock:
        await asyncio.sleep(0)
        return client

    monkeypatch.setattr(muse_agent.MuseClient, "spawn", _spawn)
    response = await muse_agent.run_muse(
        prompt="hi", cwd="/workspace", model=None, effort=None, binary=None
    )
    assert response["is_error"] is True
    assert response["usage"] is None
    assert response["result"] == "partial"


@pytest.mark.asyncio
async def test_run_fails_closed_on_durable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Durable host fails closed before starting a turn."""
    client = MagicMock()
    client.durability = SimpleNamespace(kind="durable")
    client.start_session = AsyncMock()
    client.close = AsyncMock()

    async def _spawn(_: object) -> MagicMock:
        await asyncio.sleep(0)
        return client

    monkeypatch.setattr(muse_agent.MuseClient, "spawn", _spawn)
    with pytest.raises(RuntimeError, match="ephemeral"):
        await muse_agent.run_muse(
            prompt="hi", cwd="/workspace", model=None, effort=None, binary=None
        )
    client.start_session.assert_not_awaited()
    client.close.assert_awaited()


@pytest.mark.asyncio
async def test_run_sdk_exception_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK exception before terminal result propagates."""
    client = MagicMock()
    client.durability = SimpleNamespace(kind="ephemeral")
    session = MagicMock()
    session.run = AsyncMock(side_effect=RuntimeError("transport"))
    client.start_session = AsyncMock(return_value=session)
    client.close = AsyncMock()

    async def _spawn(_: object) -> MagicMock:
        await asyncio.sleep(0)
        return client

    monkeypatch.setattr(muse_agent.MuseClient, "spawn", _spawn)
    with pytest.raises(RuntimeError):
        await muse_agent.run_muse(
            prompt="hi", cwd="/workspace", model=None, effort=None, binary=None
        )


@pytest.mark.asyncio
async def test_run_cancellation_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation closes the Muse client."""
    client = MagicMock()
    client.durability = SimpleNamespace(kind="ephemeral")
    started = asyncio.Event()

    async def _sleep_run(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        started.set()
        await asyncio.sleep(10)

    session = MagicMock()
    session.run = MagicMock(side_effect=_sleep_run)
    client.start_session = AsyncMock(return_value=session)
    client.close = AsyncMock()

    async def _spawn(_: object) -> MagicMock:
        await asyncio.sleep(0)
        return client

    monkeypatch.setattr(muse_agent.MuseClient, "spawn", _spawn)
    task = asyncio.create_task(
        muse_agent.run_muse(
            prompt="hi", cwd="/workspace", model=None, effort=None, binary=None
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    client.close.assert_awaited()


@pytest.mark.asyncio
async def test_shutdown_closes_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown closes active hosts."""
    _ = monkeypatch
    client = MagicMock()
    client.close = AsyncMock()
    muse_agent._active_hosts.add(client)  # pyright: ignore[reportPrivateUsage]
    await muse_agent.shutdown()
    client.close.assert_awaited()
    assert not muse_agent._active_hosts  # pyright: ignore[reportPrivateUsage]


def test_endpoints(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise routing and validation for Muse."""
    probe = AsyncMock(return_value=True)
    run_result = {
        "agent": "muse",
        "result": "ok",
        "model": None,
        "is_error": False,
        "duration_ms": 0,
        "usage": None,
        "backend": {"turn_id": "t", "outcome": "completed"},
    }
    run = AsyncMock(return_value=run_result)
    monkeypatch.setattr(muse_agent, "probe", probe)
    monkeypatch.setattr(muse_agent, "run_muse", run)
    path_tmp = tmp_path  # type: ignore[assignment]
    app = create_app(
        Settings(
            api_key="key",
            workspace_root=Path(str(path_tmp)),
            muse_enabled=True,
            muse_bin=None,
        )
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer key"}
    body = {"agent": "muse", "prompt": "hi"}
    agents = client.get("/v1/agents", headers=headers).json()
    assert agents["muse"]["healthy"] is True
    assert agents["muse"]["enabled"] is True
    assert agents["muse"]["supports"] == {
        "sandbox": True,
        "network_access": True,
        "rate_limits": False,
        "streaming": False,
    }
    probe.assert_awaited()
    response = client.post("/v1/query", headers=headers, json=body)
    assert response.status_code == 200
    app.state.settings = Settings(
        api_key="key", workspace_root=Path(str(path_tmp)), muse_enabled=False
    )
    app.state.muse_probe_expires = 0
    assert client.get("/v1/agents", headers=headers).json()["muse"]["healthy"] is False
    assert client.post("/v1/query", headers=headers, json=body).status_code == 503
    app.state.settings = Settings(
        api_key="key", workspace_root=Path(str(path_tmp)), muse_enabled=True
    )
    app.state.muse_probe_expires = 0
    monkeypatch.setattr(muse_agent, "probe", AsyncMock(return_value=True))
    monkeypatch.setattr(muse_agent, "run_muse", AsyncMock(return_value=run_result))
    for field in (
        "system_prompt",
        "max_budget_usd",
        "allowed_tools",
        "permission_mode",
        "options",
        "session_id",
    ):
        response = client.post("/v1/query", headers=headers, json={**body, field: "x"})
        assert response.status_code == 422, field
    bad_effort = client.post(
        "/v1/query", headers=headers, json={**body, "effort": "invalid"}
    )
    assert bad_effort.status_code == 422
    bad_cwd = client.post(
        "/v1/query", headers=headers, json={**body, "cwd": "../outside"}
    )
    assert bad_cwd.status_code == 403
    assert (
        client.post("/v1/query/stream", headers=headers, json=body).status_code == 404
    )
    codex_body = {"agent": "codex", "prompt": "hi"}
    assert (
        client.post("/v1/query/stream", headers=headers, json=codex_body).status_code
        == 404
    )


def test_probe_cache(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Health probe is cached."""
    probe = AsyncMock(return_value=True)
    monkeypatch.setattr(muse_agent, "probe", probe)
    app = create_app(
        Settings(api_key="key", workspace_root=Path(str(tmp_path)), muse_enabled=True)
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer key"}
    client.get("/v1/agents", headers=headers)
    probe.assert_awaited_once()
    client.get("/v1/agents", headers=headers)
    probe.assert_awaited_once()
    app.state.muse_probe_expires = 0
    client.get("/v1/agents", headers=headers)
    assert probe.await_count == 2


def test_deny_extra_branches() -> None:
    """Cover remaining deny fallbacks."""
    req = {
        "availableChoices": [
            {
                "choiceId": "approved-ux",
                "decision": "approved",
                "label": "x",
                "scope": "once",
            },
            {
                "choiceId": "denied-opt",
                "decision": "approvedForSession",
                "label": "y",
                "scope": "once",
            },
        ]
    }
    assert muse_agent.deny_approval(req).choice_id == "denied-opt"  # pyright: ignore[reportArgumentType]
    req2 = {
        "availableChoices": [
            {"choiceId": "a", "decision": "approved", "label": "a", "scope": "once"},
            {"choiceId": "b", "decision": "timedOut", "label": "b", "scope": "once"},
        ]
    }
    assert muse_agent.deny_approval(req2).choice_id == "b"  # pyright: ignore[reportArgumentType]


def test_stderr_and_client_info() -> None:
    """Stderr logger is reachable."""
    muse_agent._on_stderr("test")  # pyright: ignore[reportPrivateUsage]


def test_on_error_via_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture approval error handler from run."""
    result = SimpleNamespace(
        outcome="completed", final_text="ok", turn_id="t", usage=None
    )
    client = MagicMock()
    client.durability = SimpleNamespace(kind="ephemeral")
    session = MagicMock()
    session.run = AsyncMock(return_value=result)
    captured = {}

    def _cap(handler: object) -> None:
        captured["h"] = handler

    session.on_approval = MagicMock()
    session.on_approval_error = _cap
    client.start_session = AsyncMock(return_value=session)
    client.close = AsyncMock()

    async def _spawn(_: object) -> MagicMock:
        await asyncio.sleep(0)
        return client

    monkeypatch.setattr(muse_agent.MuseClient, "spawn", _spawn)
    asyncio.run(
        muse_agent.run_muse(
            prompt="hi", cwd="/workspace", model=None, effort=None, binary=None
        )
    )
    assert "h" in captured
    with pytest.raises(RuntimeError):
        captured["h"](ValueError("boom"))


def test_muse_probe_slot_429(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe slot exhaustion returns 429 via agents."""
    monkeypatch.setattr(muse_agent, "probe", AsyncMock(return_value=True))
    app = create_app(
        Settings(api_key="key", workspace_root=Path(str(tmp_path)), muse_enabled=True)
    )
    app.state.muse_semaphore = asyncio.Semaphore(0)
    client = TestClient(app)
    headers = {"Authorization": "Bearer key"}
    assert client.get("/v1/agents", headers=headers).status_code == 429


def test_muse_query_503_when_disabled(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Muse query returns 503 when disabled."""
    monkeypatch.setattr(muse_agent, "probe", AsyncMock(return_value=True))
    app = create_app(
        Settings(api_key="key", workspace_root=Path(str(tmp_path)), muse_enabled=False)
    )
    app.state.muse_probe_expires = 0
    client = TestClient(app)
    headers = {"Authorization": "Bearer key"}
    body = {"agent": "muse", "prompt": "x"}
    assert client.post("/v1/query", headers=headers, json=body).status_code == 503


def test_muse_query_timeout_504(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Muse query timeout maps to 504."""
    monkeypatch.setattr(muse_agent, "probe", AsyncMock(return_value=True))

    async def _slow(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        await asyncio.sleep(10)
        return {}

    monkeypatch.setattr(muse_agent, "run_muse", AsyncMock(side_effect=_slow))
    app = create_app(
        Settings(
            api_key="key",
            workspace_root=Path(str(tmp_path)),
            muse_enabled=True,
            max_timeout_seconds=10,
        )
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer key"}
    r = client.post(
        "/v1/query",
        headers=headers,
        json={"agent": "muse", "prompt": "x", "timeout_seconds": 0.05},
    )
    assert r.status_code == 504


def test_muse_query_sdk_exception_502(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Muse query SDK error maps to 502."""
    monkeypatch.setattr(muse_agent, "probe", AsyncMock(return_value=True))
    monkeypatch.setattr(
        muse_agent, "run_muse", AsyncMock(side_effect=RuntimeError("boom"))
    )
    app = create_app(
        Settings(api_key="key", workspace_root=Path(str(tmp_path)), muse_enabled=True)
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer key"}
    assert (
        client.post(
            "/v1/query", headers=headers, json={"agent": "muse", "prompt": "x"}
        ).status_code
        == 502
    )


def test_muse_query_capacity_429(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Muse query capacity exhaustion maps to 429."""
    monkeypatch.setattr(muse_agent, "probe", AsyncMock(return_value=True))
    app = create_app(
        Settings(api_key="key", workspace_root=Path(str(tmp_path)), muse_enabled=True)
    )
    app.state.muse_semaphore = asyncio.Semaphore(0)
    client = TestClient(app)
    headers = {"Authorization": "Bearer key"}
    assert (
        client.post(
            "/v1/query", headers=headers, json={"agent": "muse", "prompt": "x"}
        ).status_code
        == 429
    )


@pytest.mark.asyncio
async def test_muse_probe_timeout_and_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timed probe timeout and cancelled probe invalidate cache."""
    app = create_app(
        Settings(
            api_key="key",
            workspace_root=Path("/workspace"),
            muse_enabled=True,
            muse_max_concurrency=1,
        )
    )
    started = asyncio.Event()

    async def _slow(_binary: object) -> bool:
        started.set()
        await asyncio.sleep(0.2)
        return True

    monkeypatch.setattr(muse_agent, "probe", _slow)
    monkeypatch.setattr(main, "MUSE_PROBE_TIMEOUT", 0.01)
    async with main._muse_slot(app) as lease:  # pyright: ignore[reportPrivateUsage]
        result = await main._timed_muse_probe(app, lease)  # pyright: ignore[reportPrivateUsage]
    assert result is False
    # Cancelled path invalidates with a fresh app after detach releases
    await asyncio.sleep(0.3)
    app = create_app(
        Settings(
            api_key="key",
            workspace_root=Path("/workspace"),
            muse_enabled=True,
            muse_max_concurrency=1,
        )
    )
    app.state.muse_probe_expires = 0
    app.state.muse_probe_expires = 0

    async def _never(_binary: object) -> bool:
        await asyncio.sleep(0.2)
        return True

    monkeypatch.setattr(muse_agent, "probe", _never)
    monkeypatch.setattr(main, "MUSE_PROBE_TIMEOUT", 0.5)
    async with main._muse_slot(app) as lease:  # pyright: ignore[reportPrivateUsage]
        task = asyncio.create_task(main._timed_muse_probe(app, lease))  # pyright: ignore[reportPrivateUsage]
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_muse_query_with_disconnect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Muse query disconnect is handled."""
    monkeypatch.setattr(muse_agent, "probe", AsyncMock(return_value=True))
    monkeypatch.setattr(
        muse_agent, "run_muse", AsyncMock(side_effect=asyncio.CancelledError)
    )
    app = create_app(
        Settings(api_key="key", workspace_root=tmp_path, muse_enabled=True)
    )
    # Directly test _await_muse cancellation transfers lease

    app.state.muse_probe_expires = 0
    was_stored = False

    async def _run() -> None:
        nonlocal was_stored
        async with main._muse_slot(app) as lease:  # pyright: ignore[reportPrivateUsage]
            coro = asyncio.sleep(0.2)
            task = asyncio.create_task(main._await_muse(coro, lease))  # pyright: ignore[reportPrivateUsage]
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            was_stored = True

    asyncio.run(_run())
    assert was_stored is True


def test_muse_query_400_timeout_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Muse query rejects timeout exceeding server max."""
    monkeypatch.setattr(muse_agent, "probe", AsyncMock(return_value=True))
    monkeypatch.setattr(muse_agent, "run_muse", AsyncMock())
    app = create_app(
        Settings(
            api_key="key",
            workspace_root=tmp_path,
            muse_enabled=True,
            max_timeout_seconds=10,
        )
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer key"}
    body = {"agent": "muse", "prompt": "x", "timeout_seconds": 9999}
    assert client.post("/v1/query", headers=headers, json=body).status_code == 400


@pytest.mark.asyncio
async def test_muse_agent_cancelled_outer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling run_muse cancels its run task."""
    client = MagicMock()
    client.durability = SimpleNamespace(kind="ephemeral")
    started = asyncio.Event()

    async def _long(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        started.set()
        await asyncio.sleep(10)
        return {}

    session = MagicMock()
    session.run = MagicMock(side_effect=_long)
    client.start_session = AsyncMock(return_value=session)
    client.close = AsyncMock()

    async def _spawn(_: object) -> MagicMock:
        await asyncio.sleep(0)
        return client

    monkeypatch.setattr(muse_agent.MuseClient, "spawn", _spawn)
    task = asyncio.create_task(
        muse_agent.run_muse(
            prompt="hi", cwd="/workspace", model=None, effort=None, binary=None
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_muse_job_success_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Durable job executes Muse to success."""
    monkeypatch.setattr(muse_agent, "probe", AsyncMock(return_value=True))
    result = {
        "agent": "muse",
        "result": "ok",
        "model": None,
        "is_error": False,
        "duration_ms": 0,
        "usage": None,
        "backend": {"turn_id": "t", "outcome": "completed"},
    }
    monkeypatch.setattr(muse_agent, "run_muse", AsyncMock(return_value=result))

    app = create_app(
        Settings(
            api_key="key",
            workspace_root=tmp_path,
            muse_enabled=True,
            jobs_db_path=tmp_path / "jobs.sqlite3",
        )
    )
    app.state.jobs.start()
    body = MuseQueryRequest(agent="muse", prompt="hi")

    async def _run() -> None:
        job = app.state.jobs.submit(body, "Bearer key", None)
        # wait for completion
        status = app.state.jobs.get(job["job_id"], "Bearer key")
        for _ in range(50):
            status = app.state.jobs.get(job["job_id"], "Bearer key")
            if status["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                break
            await asyncio.sleep(0.05)
        assert status["status"] == "succeeded"
        assert status["result"]["result"] == "ok"

    asyncio.run(_run())
    app.state.jobs.stop()
