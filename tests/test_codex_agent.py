"""Tests for Codex integration without a live backend."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn, Self
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from agsrv import codex_agent
from agsrv.config import Settings
from agsrv.main import create_app

if TYPE_CHECKING:
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

    async def __aenter__(self) -> Self:
        """Enter the fake context."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the fake context."""


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
    run = AsyncMock(return_value={"agent": "codex", "result": "ok"})
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
    assert client.get("/v1/agents", headers=headers).json()["codex"]["healthy"] is False
