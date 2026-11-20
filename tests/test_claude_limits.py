"""Tests for Claude subscription quota reporting."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import ResultMessage
from fastapi import Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from agsrv import main
from agsrv.claude_limits import parse_usage
from agsrv.config import Settings
from agsrv.main import VERSION, create_app
from tests.test_main import CleanupRunner, FakeRunner

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


NOW = datetime(2026, 11, 20, 18, 0, tzinfo=UTC)
SAMPLE = (
    "You are currently using your subscription to power your Claude Code usage\n\n"
    "Current session: 25% used · resets Nov 21, 4:10am (Asia/Tokyo)\n"
    "Current week (all models): 63% used · resets Nov 22, 7:59pm (Asia/Tokyo)"
)


def test_parse_sample() -> None:
    """Parse the observed subscription output."""
    result = parse_usage(SAMPLE, NOW)
    assert result == {
        "available": True,
        "five_hour": {"used_percent": 25, "resets_at": "2026-11-21T04:10:00+09:00"},
        "seven_day": {"used_percent": 63, "resets_at": "2026-11-22T19:59:00+09:00"},
        "seven_day_opus": None,
        "seven_day_sonnet": None,
        "raw": SAMPLE,
        "fetched_at": "2026-11-20T18:00:00+00:00",
    }


def test_parse_variants() -> None:
    """Handle model windows, unknown zones, absent zones and bad times."""
    result = parse_usage(
        "Current week (Opus): 4% used · resets Dec 1, 1:00pm (Bad/Zone)\n"
        "Current week (Sonnet): 5% used · resets Dec 2, 2:00am\n"
        "Current session: 7% used · resets nonsense\n"
        "Current week (Other): 9% used · resets Dec 1, 1:00pm (UTC)",
        NOW,
    )
    assert result["seven_day_opus"]["resets_at"] == "2026-12-01T13:00:00+00:00"
    assert result["seven_day_sonnet"]["used_percent"] == 5
    assert result["five_hour"] == {"used_percent": 7, "resets_at": None}
    assert result["seven_day"] is None
    assert parse_usage("Current session: unknown\nother", NOW)["available"] is False
    assert (
        parse_usage("Current session: 9% used · resets Jan 1, 1:00am (UTC)", NOW)[
            "five_hour"
        ]["resets_at"]
        == "2027-01-01T01:00:00+00:00"
    )
    assert (
        parse_usage("Current session: 9% used", NOW)["five_hour"]["resets_at"] is None
    )
    assert parse_usage("Current session: 9% used · resets Dec 1, 1:00pm (../bad)", NOW)[
        "available"
    ]


def test_endpoint(tmp_path: Path) -> None:
    """Exercise authentication, success, failures and capacity."""
    message = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=0,
        is_error=False,
        num_turns=0,
        session_id="s",
        result=SAMPLE,
        total_cost_usd=0.0,
    )
    runner = FakeRunner(messages=[message])
    runner.stream = MagicMock(wraps=runner.stream)
    app = create_app(Settings(api_key="key", workspace_root=tmp_path), runner)
    client = TestClient(app)
    path = "/v1/agents/claude/rate-limits"
    headers = {"Authorization": "Bearer key"}
    assert client.get(path).status_code == 401
    response = client.get(path, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["available"]
    assert data["five_hour"]["used_percent"] == 25
    assert data["backend"]["subtype"] == "success"
    assert runner.stream_calls == 1
    assert runner.stream.call_args.args[0] == "/usage"
    assert runner.options is not None
    assert runner.options.tools == []
    message.result = "API billing"
    assert client.get(path, headers=headers).json()["available"] is False
    message.is_error = True
    assert client.get(path, headers=headers).status_code == 502
    message.is_error = False
    runner.fail_after = 0
    assert client.get(path, headers=headers).status_code == 502
    runner.fail_after = None
    runner.delay = 0.1
    app.state.settings = replace(app.state.settings, max_timeout_seconds=0.01)
    assert client.get(path, headers=headers).status_code == 504
    app.state.semaphore = asyncio.Semaphore(0)
    assert client.get(path, headers=headers).status_code == 429
    assert (
        client.get("/v1/agents", headers=headers).json()["claude"]["supports"][
            "rate_limits"
        ]
        is True
    )


def test_dst_rollover_and_gap() -> None:
    """Compare instants across DST and reject nonexistent local reset times."""
    rollover = parse_usage(
        "Current session: 12% used · resets Nov 1, 12:15am (America/New_York)",
        datetime(2026, 11, 1, 16, 45, tzinfo=UTC),
    )
    assert rollover["five_hour"]["resets_at"] == "2027-11-01T00:15:00-04:00"
    gap = parse_usage(
        "Current session: 12% used · resets Mar 8, 2:30am (America/New_York)",
        datetime(2026, 3, 8, 5, 0, tzinfo=UTC),
    )
    assert gap["five_hour"] == {"used_percent": 12, "resets_at": None}


def test_invalid_percentages_and_timezone() -> None:
    """Malformed lines cannot suppress other valid usage windows."""
    for invalid in ("101", "9" * 4301):
        raw = f"Current session: {invalid}% used\nCurrent week (Opus): 100% used"
        result = parse_usage(raw, NOW)
        assert result["five_hour"] is None
        assert result["seven_day_opus"]["used_percent"] == 100
        assert not parse_usage(f"Current session: {invalid}% used", NOW)["available"]
    raw = f"Current session: 0% used · resets Dec 1, 1:00pm ({'x' * 300})"
    assert parse_usage(raw, NOW)["five_hour"] == {
        "used_percent": 0,
        "resets_at": "2026-12-01T13:00:00+00:00",
    }


def test_documented_info_version() -> None:
    """The REST example tracks the running API version."""
    docs = Path("docs/api/rest-api.md").read_text(encoding="utf-8")
    assert f'"version": "{VERSION}"' in docs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("waiting", "external"), [(False, False), (True, False), (False, True)]
)
async def test_disconnect_quota_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, waiting: bool, external: bool
) -> None:
    """Disconnect cancels either workspace waiting or an active usage stream."""
    runner = CleanupRunner()
    app = create_app(
        Settings(api_key="key", workspace_root=tmp_path, max_concurrency=1), runner
    )
    original_prepare = main._prepare  # pyright: ignore[reportPrivateUsage]

    async def prepare(request: Request, body: Any) -> Any:  # ruff: ignore[any-type]
        body.sandbox = False
        return await original_prepare(request, body)

    monkeypatch.setattr(main, "_prepare", prepare)
    monkeypatch.setattr(main, "DISCONNECT_POLL_INTERVAL", 0.01)
    disconnected = False

    async def receive() -> dict[str, str]:
        if disconnected:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    endpoint = next(
        route.endpoint
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/v1/agents/claude/rate-limits"
    )
    request = Request({"type": "http", "app": app}, receive)
    async with app.state.jobs.workspace(tmp_path) if waiting else _empty_context():
        task = asyncio.create_task(endpoint(request))
        if waiting:
            while not app.state.semaphore.locked():  # ruff: ignore[async-busy-wait]
                await asyncio.sleep(0.01)
        else:
            await asyncio.wait_for(runner.started.wait(), 1)
        if external:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        else:
            disconnected = True
            response = await asyncio.wait_for(task, 1)
            assert response.status_code == 499
        assert not app.state.semaphore.locked()
        if waiting:
            assert not runner.started.is_set()
        else:
            assert runner.closed.is_set()


@asynccontextmanager
async def _empty_context() -> AsyncGenerator[None, None]:
    yield
