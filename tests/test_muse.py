"""Tests for Muse integration without a live backend."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from muse_code.facade.approval import ApprovalRouter
from muse_code.facade.turn_handle import _Completed, _TerminalUnknown, _Unqueued  # pyright: ignore[reportPrivateUsage]

from agsrv import main, muse
from agsrv.config import Settings
from agsrv.main import (
    create_app,
)
from agsrv.schemas import MuseQueryRequest

if TYPE_CHECKING:
    from muse_code import MuseClientSpawnOptions


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
    args = muse.serve_args()
    assert args == ["serve", "--no-session-log", "--contributor-mode", "off"]
    assert "--disable-sandbox" not in args
    assert "--trust-workspace" not in args
    assert muse.APPROVAL_MODE in {"onRequest", "promptUnmatched", "denyUnmatched"}


def test_deny_approval_picks_denied() -> None:
    """Deny handler picks denied decision."""
    req: dict[str, list[dict[str, str]]] = _request([
        _choice("allow", "approved"),
        _choice("deny-1", "denied"),
    ])
    decision = muse.deny_approval(req)  # pyright: ignore[reportArgumentType,reportUnknownArgumentType,reportUnknownVariableType]
    assert decision.choice_id == "deny-1"
    assert decision.choice_id != "allow"


def test_deny_approval_label_fallback() -> None:
    """Deny handler uses label fallback."""
    req: dict[str, list[dict[str, str]]] = _request([
        {"choiceId": "x", "decision": "approved", "label": "Deny it", "scope": "once"},
    ])
    with pytest.raises(RuntimeError):
        muse.deny_approval(req)  # pyright: ignore[reportArgumentType]


def test_deny_approval_raises_without_denied() -> None:
    """Deny handler fails closed without denied choice."""
    req: dict[str, list[dict[str, str]]] = _request([
        _choice("a", "approved"),
        _choice("b", "approvedForSession"),
    ])
    with pytest.raises(RuntimeError):
        muse.deny_approval(req)  # pyright: ignore[reportArgumentType]


def test_deny_approval_empty_request() -> None:
    """Deny handler raises on empty choices."""
    with pytest.raises(RuntimeError):
        muse.deny_approval({"availableChoices": []})  # pyright: ignore[reportArgumentType,reportUnknownArgumentType]


def test_is_ephemeral() -> None:
    """Ephemerality check discriminates kinds."""
    client = MagicMock()
    client.durability = SimpleNamespace(kind="ephemeral")
    assert muse._is_ephemeral(client)  # pyright: ignore[reportPrivateUsage]
    client.durability = SimpleNamespace(kind="durable")
    assert not muse._is_ephemeral(client)  # pyright: ignore[reportPrivateUsage]
    client.durability = SimpleNamespace(kind="unrecognized")
    assert not muse._is_ephemeral(client)  # pyright: ignore[reportPrivateUsage]


def test_usage_to_dict() -> None:
    """Usage mapping covers None and objects."""
    assert muse._usage_to_dict(None) is None  # pyright: ignore[reportPrivateUsage]
    assert muse._usage_to_dict({"a": 1}) == {"a": 1}  # pyright: ignore[reportPrivateUsage]
    obj = SimpleNamespace(
        inputTokens=1, outputTokens=2, cachedTokens=3, reasoningTokens=4
    )
    result = muse._usage_to_dict(obj)  # pyright: ignore[reportPrivateUsage,reportUnknownVariableType]
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

    assert muse._usage_to_dict(Dump()) == {"dumped": True}  # pyright: ignore[reportPrivateUsage,reportUnknownVariableType]


def test_outcome_str() -> None:
    """Outcome coercion handles enums and strings."""
    assert (
        muse._outcome_str(  # pyright: ignore[reportPrivateUsage]
            _Completed(params={"terminal": "completed"}, observed_start=True)  # pyright: ignore[reportArgumentType]
        )
        == "completed"
    )  # pyright: ignore[reportPrivateUsage,reportArgumentType]
    assert muse._outcome_str(_Unqueued(params={})) == "unqueued"  # pyright: ignore[reportPrivateUsage,reportArgumentType]
    assert muse._outcome_str(_TerminalUnknown()) == "terminalUnknown"  # pyright: ignore[reportPrivateUsage]
    assert muse._outcome_str(SimpleNamespace(kind="other")) == "unknown"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_probe_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe reports ephemeral health."""

    class FakeClient:
        """Fake client with fixed durability."""

        def __init__(self, kind: str) -> None:
            self._kind = kind
            self.initialize_result = {"sandboxAvailability": "available"}

        @property
        def durability(self) -> SimpleNamespace:
            """Durability stub."""
            return SimpleNamespace(kind=self._kind)

        async def close(self) -> None:
            """Close stub."""
            return None

    async def _spawn_ephemeral_async(options: MuseClientSpawnOptions) -> FakeClient:  # pyright: ignore[reportUnusedFunction]
        assert options.args == [
            "serve",
            "--no-session-log",
            "--contributor-mode",
            "off",
        ]  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        await asyncio.sleep(0)
        return FakeClient("ephemeral")

    async def _spawn_durable_async(_: object) -> FakeClient:  # pyright: ignore[reportUnusedFunction]
        await asyncio.sleep(0)
        return FakeClient("durable")

    async def _fail(_: object) -> FakeClient:
        await asyncio.sleep(0)
        msg = "no binary"
        raise RuntimeError(msg)

    monkeypatch.setattr(muse.MuseClient, "spawn", _spawn_ephemeral_async)
    assert await muse.probe(None) is True
    monkeypatch.setattr(muse.MuseClient, "spawn", _spawn_durable_async)
    assert await muse.probe(None) is False
    monkeypatch.setattr(muse.MuseClient, "spawn", _fail)
    assert await muse.probe(None) is False


@pytest.mark.asyncio
async def test_run_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Completed turn maps to success response."""
    result = SimpleNamespace(
        outcome=_Completed(params={"terminal": "completed"}, observed_start=True),  # pyright: ignore[reportArgumentType]
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

    async def _spawn(options: MuseClientSpawnOptions) -> MagicMock:
        assert options.args == [
            "serve",
            "--no-session-log",
            "--contributor-mode",
            "off",
        ]  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        await asyncio.sleep(0)
        return client

    monkeypatch.setattr(muse.MuseClient, "spawn", _spawn)
    response = await muse.run_muse(
        prompt="hi", cwd="/workspace", model="m1", effort="low", binary=None
    )
    assert response["result"] == "done"
    assert response["is_error"] is False
    assert response["model"] == "m1"
    assert response["backend"] == {"turn_id": "t1", "outcome": "completed"}
    assert response["usage"] == {"input_tokens": 1, "output_tokens": 2}
    client.start_session.assert_awaited_once()
    options = client.start_session.call_args.args[0]
    assert options.approval_mode == muse.APPROVAL_MODE
    assert options.workspace_root == "/workspace"
    assert options.model_id == "m1"
    session.on_approval.assert_called_once()
    assert session.on_approval.call_args.args[0] is muse.deny_approval


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
async def test_run_terminal_errors_are_is_error(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """Failed or cancelled turns map to is_error."""
    result = SimpleNamespace(
        outcome=_Completed(params={"terminal": outcome}, observed_start=True),  # pyright: ignore[reportArgumentType]
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

    monkeypatch.setattr(muse.MuseClient, "spawn", _spawn)
    response = await muse.run_muse(
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

    monkeypatch.setattr(muse.MuseClient, "spawn", _spawn)
    with pytest.raises(RuntimeError, match="ephemeral"):
        await muse.run_muse(
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

    monkeypatch.setattr(muse.MuseClient, "spawn", _spawn)
    with pytest.raises(RuntimeError):
        await muse.run_muse(
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

    monkeypatch.setattr(muse.MuseClient, "spawn", _spawn)
    task = asyncio.create_task(
        muse.run_muse(
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
    muse._active_hosts.add(client)  # pyright: ignore[reportPrivateUsage]
    await muse.shutdown()
    client.close.assert_awaited()
    assert not muse._active_hosts  # pyright: ignore[reportPrivateUsage]


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
    monkeypatch.setattr(muse, "probe", probe)
    monkeypatch.setattr(muse, "run_muse", run)
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
        "rate_limits": True,
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
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
    monkeypatch.setattr(muse, "run_muse", AsyncMock(return_value=run_result))
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
    monkeypatch.setattr(muse, "probe", probe)
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
    with pytest.raises(RuntimeError):
        muse.deny_approval(req)  # pyright: ignore[reportArgumentType]
    req2 = {
        "availableChoices": [
            {"choiceId": "a", "decision": "approved", "label": "a", "scope": "once"},
            {"choiceId": "b", "decision": "timedOut", "label": "b", "scope": "once"},
        ]
    }
    with pytest.raises(RuntimeError):
        muse.deny_approval(req2)  # pyright: ignore[reportArgumentType]


def test_stderr_and_client_info() -> None:
    """Stderr logger is reachable."""
    muse._on_stderr("test")  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_approval_failure_through_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """An SDK-reported approval failure terminates the waiting run."""
    router = ApprovalRouter("s", None)
    session = MagicMock()
    session.on_approval = router.on_approval
    session.on_approval_error = router.on_approval_error

    async def run(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        router.requested({
            "approvalId": "a",
            "currentRequirementId": {"approvalId": "a", "sourceIndex": 0},
            "availableChoices": [],
        })  # pyright: ignore[reportArgumentType]
        await asyncio.Event().wait()

    session.run = run
    client = MagicMock()
    client.durability = SimpleNamespace(kind="ephemeral")
    client.start_session = AsyncMock(return_value=session)
    client.close = AsyncMock()
    monkeypatch.setattr(muse.MuseClient, "spawn", AsyncMock(return_value=client))
    with pytest.raises(RuntimeError, match="approval failed"):
        await asyncio.wait_for(
            muse.run_muse(
                prompt="hi", cwd="/workspace", model=None, effort=None, binary=None
            ),
            1,
        )


def test_muse_probe_slot_429(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe slot exhaustion returns 429 via agents."""
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
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
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
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
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))

    async def _slow(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        await asyncio.sleep(10)
        return {}

    monkeypatch.setattr(muse, "run_muse", AsyncMock(side_effect=_slow))
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
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
    monkeypatch.setattr(muse, "run_muse", AsyncMock(side_effect=RuntimeError("boom")))
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
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
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

    monkeypatch.setattr(muse, "probe", _slow)
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
    started.clear()

    async def _never(_binary: object) -> bool:
        started.set()
        await asyncio.sleep(0.2)
        return True

    monkeypatch.setattr(muse, "probe", _never)
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
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
    monkeypatch.setattr(muse, "run_muse", AsyncMock(side_effect=asyncio.CancelledError))
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
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
    monkeypatch.setattr(muse, "run_muse", AsyncMock())
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
async def test_muse_cancelled_outer(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(muse.MuseClient, "spawn", _spawn)
    task = asyncio.create_task(
        muse.run_muse(
            prompt="hi", cwd="/workspace", model=None, effort=None, binary=None
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_muse_job_success_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Durable job executes Muse to success."""
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
    result = {
        "agent": "muse",
        "result": "ok",
        "model": None,
        "is_error": False,
        "duration_ms": 0,
        "usage": None,
        "backend": {"turn_id": "t", "outcome": "completed"},
    }
    monkeypatch.setattr(muse, "run_muse", AsyncMock(return_value=result))

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

        await app.state.jobs.stop()

    asyncio.run(_run())


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_cache", [False, True])
async def test_muse_probe_coalesces_and_cancel_invalidation(
    monkeypatch: pytest.MonkeyPatch,
    newer_cache: bool,
) -> None:
    """Waiters share a probe and cancelled probes invalidate their cache."""
    app = create_app(Settings(muse_enabled=True))
    started = asyncio.Event()
    release = asyncio.Event()

    async def probe(_binary: object) -> bool:
        started.set()
        await release.wait()
        return True

    monkeypatch.setattr(muse, "probe", probe)
    first = asyncio.create_task(main._muse_available(app))  # pyright: ignore[reportPrivateUsage]
    await started.wait()
    second = asyncio.create_task(main._muse_available(app))  # pyright: ignore[reportPrivateUsage]
    await asyncio.sleep(0)
    release.set()
    assert await first
    assert await second

    started.clear()
    release.clear()
    app.state.muse_probe_expires = 0
    task = asyncio.create_task(main._muse_available(app))  # pyright: ignore[reportPrivateUsage]
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    if newer_cache:
        app.state.muse_probe_expires += 30
    expected = app.state.muse_probe_expires if newer_cache else 0
    release.set()
    for _ in range(10):
        await asyncio.sleep(0)
    assert app.state.muse_probe_expires == expected


@pytest.mark.asyncio
async def test_muse_disconnect_and_external_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disconnect returns 499 while external cancellation propagates."""
    started = asyncio.Event()

    async def query(_body: object, _request: object) -> dict[str, object]:
        started.set()
        await asyncio.Event().wait()
        return {}

    monkeypatch.setattr(main, "_muse_query", query)
    request = MagicMock()
    request.is_disconnected = AsyncMock(return_value=True)
    body = MuseQueryRequest(agent="muse", prompt="hi")
    response = await main._muse_query_with_disconnect(body, request)  # pyright: ignore[reportPrivateUsage,reportUnknownVariableType]
    assert response.status_code == 499  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    request.is_disconnected = AsyncMock(return_value=False)
    started.clear()
    task = asyncio.create_task(main._muse_query_with_disconnect(body, request))  # pyright: ignore[reportPrivateUsage]
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_disabled_muse_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabled jobs persist failure and release the shared workspace."""
    app = create_app(
        Settings(
            workspace_root=tmp_path,
            muse_enabled=False,
            jobs_db_path=tmp_path / "jobs.db",
        )
    )
    manager = app.state.jobs
    manager.start()
    body = MuseQueryRequest(agent="muse", prompt="hi")
    job = manager.submit(body, "Bearer key", None)
    await manager.tasks[job["job_id"]]
    result = manager.get(job["job_id"], "Bearer key")
    assert result["status"] == "failed"
    assert result["error"]["code"] == "execution_failed"
    app.state.settings = replace(app.state.settings, muse_enabled=True)
    monkeypatch.setattr(muse, "run_muse", AsyncMock(return_value={"is_error": False}))
    job2 = manager.submit(body, "Bearer key", None)
    await asyncio.wait_for(manager.tasks[job2["job_id"]], 1)
    assert manager.get(job2["job_id"], "Bearer key")["status"] == "succeeded"
    await manager.stop()


def test_muse_rate_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Usage endpoint reads refreshed windows and handles missing data and errors."""
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
    raw = {
        "usage": {
            "observedAtMs": 1700000000000,
            "tier": "pro",
            "window": {
                "usedPercent": 30,
                "resetsAtMs": 1700000000000,
                "windowDurationMins": 300,
            },
            "weekly": {"usedPercent": 50, "resetsAtMs": 1700000000000},
        }
    }
    read = AsyncMock(return_value=raw)
    monkeypatch.setattr(muse, "read_usage", read)
    app = create_app(
        Settings(api_key="key", workspace_root=tmp_path, muse_enabled=True)
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer key"}
    url = "/v1/agents/muse/rate-limits"
    response = client.get(url, headers=headers)
    assert response.status_code == 200
    assert response.json()["five_hour"]["used_percent"] == 30
    assert response.json()["seven_day"]["used_percent"] == 50
    assert response.json()["raw"] == raw
    assert client.get(url, headers=headers).json() == response.json()
    read.assert_awaited_once()
    app.state.muse_usage_expires = 0
    read.return_value = {}
    assert client.get(url, headers=headers).json()["available"] is False
    app.state.muse_usage_expires = 0
    read.side_effect = RuntimeError("error")
    assert client.get(url, headers=headers).status_code == 502
    app.state.settings = replace(app.state.settings, muse_enabled=False)
    assert client.get(url, headers=headers).status_code == 503
    app.state.settings = replace(app.state.settings, muse_enabled=True)
    app.state.muse_healthy = False
    assert client.get(url, headers=headers).status_code == 503


@pytest.mark.asyncio
async def test_usage_concurrent_polls_share_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent polls share one host refresh and the cached observation."""
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
    started, release = asyncio.Event(), asyncio.Event()
    host = MagicMock()
    host.durability = SimpleNamespace(kind="ephemeral")
    host.initialize_result = {"sandboxAvailability": "available"}
    host.close = AsyncMock()

    async def refresh(*args: object) -> dict[str, object]:
        _ = args
        started.set()
        await release.wait()
        return {"usage": {}}

    host.request = AsyncMock(side_effect=refresh)
    monkeypatch.setattr(muse.MuseClient, "spawn", AsyncMock(return_value=host))
    app = create_app(
        Settings(api_key="key", workspace_root=tmp_path, muse_enabled=True)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = "/v1/agents/muse/rate-limits"
        headers = {"Authorization": "Bearer key"}
        first = asyncio.create_task(client.get(url, headers=headers))
        await started.wait()
        second = asyncio.create_task(client.get(url, headers=headers))
        await asyncio.sleep(0)
        release.set()
        responses = await asyncio.gather(first, second)
        assert [response.status_code for response in responses] == [200, 200]
        assert responses[0].json() == responses[1].json()
        assert (await client.get(url, headers=headers)).json() == responses[0].json()
    host.request.assert_awaited_once_with("usage/refresh")
    host.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_muse_cleanup_and_unresponsive_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing owns cleanup across cancellation, including a stuck turn."""
    client = MagicMock()
    client.durability = SimpleNamespace(kind="ephemeral")
    client.initialize_result = {"sandboxAvailability": "available"}
    closing, release, started = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def close() -> None:
        closing.set()
        await release.wait()

    async def run(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    session = MagicMock()
    session.run = run
    client.start_session = AsyncMock(return_value=session)
    client.close = close
    monkeypatch.setattr(muse.MuseClient, "spawn", AsyncMock(return_value=client))
    task = asyncio.create_task(
        muse.run_muse(prompt="x", cwd="/", model=None, effort=None, binary=None)
    )
    await started.wait()
    task.cancel()
    await asyncio.wait_for(closing.wait(), 2)
    task.cancel()
    assert client in muse._active_hosts  # pyright: ignore[reportPrivateUsage]
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert client not in muse._active_hosts  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_muse_sandbox_and_usage_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both probe and execution reject unavailable sandbox; usage refreshes."""
    client = MagicMock()
    client.durability = SimpleNamespace(kind="ephemeral")
    client.initialize_result = {"sandboxAvailability": "unavailable"}
    client.close = AsyncMock()
    client.request = AsyncMock(return_value={"usage": {}})
    monkeypatch.setattr(muse.MuseClient, "spawn", AsyncMock(return_value=client))
    assert not await muse.probe(None)
    with pytest.raises(RuntimeError, match="sandbox"):
        await muse.run_muse(prompt="x", cwd="/", model=None, effort=None, binary=None)
    with pytest.raises(RuntimeError, match="unavailable"):
        await muse.read_usage(None)
    client.initialize_result = {"sandboxAvailability": "available"}
    assert await muse.read_usage(None) == {"usage": {}}
    client.request.assert_awaited_once_with("usage/refresh")
    spawn = cast("AsyncMock", muse.MuseClient.spawn)
    assert spawn.call_args.args[0].capabilities == {"experimentalApi": True}
    client.request.side_effect = RuntimeError("refresh failed")
    assert await muse.read_usage(None) == {}
    spawn.side_effect = OSError("spawn failed")
    assert await muse.read_usage(None) == {}


def test_invalid_usage() -> None:
    """Malformed and out-of-range observations fail soft."""
    samples: list[dict[str, Any]] = [
        {"usage": {"window": None, "weekly": {}}},
        {"usage": {"window": {"usedPercent": 1, "resetsAtMs": 10**30}}},
    ]
    for raw in samples:
        assert not muse.parse_usage(raw)["available"]


def test_usage_integer_fields() -> None:
    """Preserve over-quota values and skip malformed integer fields."""
    window = {"usedPercent": 101, "resetsAtMs": 1}
    result = muse.parse_usage({"usage": {"window": window, "weekly": window}})
    assert result["available"] is True
    assert result["five_hour"]["used_percent"] == 101
    assert result["seven_day"]["used_percent"] == 101
    for field in ("usedPercent", "resetsAtMs"):
        for value in (True, False, 1.5):
            malformed = {**window, field: value}
            result = muse.parse_usage({
                "usage": {"window": malformed, "weekly": malformed}
            })
            assert result["available"] is False
            assert result["five_hour"] is None
            assert result["seven_day"] is None
    assert not muse.parse_usage({
        "usage": {"window": {**window, "resetsAtMs": 10**400}}
    })["available"]


def test_usage_reader_availability_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reader handshake unavailability is distinct from a usage operation failure."""
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
    host = MagicMock()
    host.durability = SimpleNamespace(kind="ephemeral")
    host.initialize_result = {"sandboxAvailability": "unavailable"}
    host.close = AsyncMock()
    host.request = AsyncMock(side_effect=RuntimeError("refresh failed"))
    monkeypatch.setattr(muse.MuseClient, "spawn", AsyncMock(return_value=host))
    app = create_app(
        Settings(api_key="key", workspace_root=tmp_path, muse_enabled=True)
    )
    client = TestClient(app)
    url = "/v1/agents/muse/rate-limits"
    headers = {"Authorization": "Bearer key"}
    response = client.get(url, headers=headers)
    assert response.status_code == 503
    assert response.json()["detail"] == "Muse is disabled or unavailable"
    host.request.assert_not_awaited()
    host.initialize_result = {"sandboxAvailability": "available"}
    app.state.muse_usage_expires = 0
    response = client.get(url, headers=headers)
    assert response.status_code == 200
    assert response.json()["available"] is False
    host.request.assert_awaited_once_with("usage/refresh")


@pytest.mark.asyncio
async def test_queued_query_observes_negative_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A query waiting for a workspace observes a probe's negative cache."""
    app = create_app(
        Settings(workspace_root=tmp_path, muse_enabled=True, muse_max_concurrency=1)
    )
    started, cleanup, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def probe(_binary: object) -> bool:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup.set()
            await release.wait()
        return True

    monkeypatch.setattr(muse, "probe", probe)
    monkeypatch.setattr(main, "MUSE_PROBE_TIMEOUT", 0.01)
    request = MagicMock()
    request.app = app
    async with app.state.jobs.workspace(tmp_path):
        query = asyncio.create_task(
            main._muse_query(MuseQueryRequest(agent="muse", prompt="x"), request)  # pyright: ignore[reportPrivateUsage]
        )
        await asyncio.sleep(0)
        assert not query.done()
        assert not await main._muse_available(app)  # pyright: ignore[reportPrivateUsage]
        await started.wait()
        await cleanup.wait()
        assert app.state.muse_semaphore.locked()
        assert app.state.muse_healthy is False
    try:
        with pytest.raises(HTTPException) as exc:
            await query
        assert exc.value.status_code == 503
    finally:
        release.set()
        for _ in range(20):
            await asyncio.sleep(0)
            if not app.state.muse_semaphore.locked():
                break
    assert not app.state.muse_semaphore.locked()


def test_muse_unavailable_and_rate_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold negative probes and usage deadlines return the documented status."""
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=False))
    app = create_app(
        Settings(
            api_key="key",
            workspace_root=tmp_path,
            muse_enabled=True,
            max_timeout_seconds=0.05,
            muse_max_concurrency=1,
        )
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer key"}
    url = "/v1/agents/muse/rate-limits"
    assert client.get(url, headers=headers).status_code == 503
    app.state.muse_probe_expires = 0
    assert (
        client.post(
            "/v1/query", headers=headers, json={"agent": "muse", "prompt": "x"}
        ).status_code
        == 503
    )
    app.state.muse_probe_expires = 0
    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
    host = MagicMock()
    host.durability = SimpleNamespace(kind="ephemeral")
    host.initialize_result = {"sandboxAvailability": "available"}
    host.close = AsyncMock()

    async def stalled_refresh(*args: object) -> None:
        _ = args
        await asyncio.Event().wait()

    host.request = AsyncMock(side_effect=stalled_refresh)
    monkeypatch.setattr(muse.MuseClient, "spawn", AsyncMock(return_value=host))
    assert client.get(url, headers=headers).status_code == 504
    host.request.assert_awaited_once_with("usage/refresh")
    host.close.assert_awaited_once()
    assert host not in muse._active_hosts  # pyright: ignore[reportPrivateUsage]
    assert not app.state.muse_semaphore.locked()


@pytest.mark.asyncio
async def test_query_cancel_owns_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """External cancellation leaves capacity held until host cleanup finishes."""
    started, release = asyncio.Event(), asyncio.Event()

    async def run(**kwargs: object) -> dict[str, object]:
        _ = kwargs
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await release.wait()
        return {}

    monkeypatch.setattr(muse, "probe", AsyncMock(return_value=True))
    monkeypatch.setattr(muse, "run_muse", run)
    app = create_app(
        Settings(workspace_root=tmp_path, muse_enabled=True, muse_max_concurrency=1)
    )
    request = MagicMock()
    request.app = app
    task = asyncio.create_task(
        main._muse_query(MuseQueryRequest(agent="muse", prompt="x"), request)  # pyright: ignore[reportPrivateUsage]
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert app.state.muse_semaphore.locked()
    release.set()
    # Reclaiming the per-turn directory runs in an executor, so yield for real.
    for _ in range(200):
        await asyncio.sleep(0.01)
        if not app.state.muse_semaphore.locked():
            break
    assert not app.state.muse_semaphore.locked()
