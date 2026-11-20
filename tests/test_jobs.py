"""Integration coverage for durable asynchronous jobs without provider logins."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from claude_agent_sdk import Message, ResultMessage
from fastapi.testclient import TestClient

from agsrv.config import Settings
from agsrv.main import create_app
from tests.test_main import FakeRunner, canned

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from claude_agent_sdk import ClaudeAgentOptions

AUTH = {"Authorization": "Bearer secret"}
BODY = {"agent": "claude", "prompt": "inspect", "sandbox": False}


def _settings(root: Path, **overrides: Any) -> Settings:  # ruff: ignore[any-type]
    return replace(
        Settings(api_key="secret", workspace_root=root, jobs_db_path=root / "jobs.db"),
        **overrides,
    )


def _wait(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(300):
        response = client.get(f"/v1/jobs/{job_id}", headers=AUTH)
        assert response.status_code == 200
        value: dict[str, Any] = response.json()
        if value["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return value
        time.sleep(0.01)
    pytest.fail("job did not reach a terminal state")


def test_submit_poll_and_idempotency(tmp_path: Path) -> None:
    """The client may disappear without cancelling its server-owned task."""
    runner = FakeRunner(delay=0.05)
    with TestClient(create_app(_settings(tmp_path), runner)) as client:
        headers = {**AUTH, "Idempotency-Key": "vm-retry-01"}
        first = client.post("/v1/jobs", headers=headers, json=BODY)
        assert first.status_code == 202
        job_id = first.json()["job_id"]
        assert first.json()["status_url"] == f"/v1/jobs/{job_id}"
        # The original request has ended before the replacement VM polls.
        again = client.post("/v1/jobs", headers=headers, json=BODY)
        assert again.status_code == 202
        assert again.json()["job_id"] == job_id
        conflict = client.post(
            "/v1/jobs", headers=headers, json={**BODY, "prompt": "other"}
        )
        assert conflict.status_code == 409
        finished = _wait(client, job_id)
        assert finished["status"] == "succeeded"
        assert finished["agent"] == "claude"
        assert finished["result"]["result"] == "done"
        assert finished["created_at"] and finished["started_at"] and finished["finished_at"]
        assert finished["error"] is None
        assert runner.stream_calls == 1
        assert client.post(
            f"/v1/jobs/{job_id}/cancel", headers=AUTH
        ).json()["status"] == "succeeded"
        assert client.get(f"/v1/jobs/{job_id}", headers={"Authorization": "Bearer other"}).status_code == 401
        assert client.get("/v1/jobs/unknown", headers=AUTH).status_code == 404


def test_queue_saturation_and_cancellation(tmp_path: Path) -> None:
    """Queue admission is bounded, and queued/running jobs can be cancelled."""
    runner = FakeRunner(delay=0.3)
    settings = _settings(tmp_path, jobs_max_pending=1)
    with TestClient(create_app(settings, runner)) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        extra = client.post("/v1/jobs", headers=AUTH, json={**BODY, "prompt": "two"})
        assert extra.status_code == 429
        assert extra.headers["retry-after"] == "5"
        cancelled = client.post(f"/v1/jobs/{job_id}/cancel", headers=AUTH)
        assert cancelled.status_code == 200
        assert _wait(client, job_id)["status"] == "cancelled"
        assert client.post(f"/v1/jobs/{job_id}/cancel", headers=AUTH).status_code == 200


def test_timeout_and_transport_failure(tmp_path: Path) -> None:
    """Deadlines and SDK failures are sanitized but recorded as failures."""
    with TestClient(
        create_app(_settings(tmp_path), FakeRunner(delay=0.2))
    ) as client:
        response = client.post(
            "/v1/jobs", headers=AUTH, json={**BODY, "timeout_seconds": 0.01}
        )
        outcome = _wait(client, response.json()["job_id"])
        assert outcome["status"] == "failed"
        assert outcome["error"]["code"] == "deadline_exceeded"
    with TestClient(
        create_app(_settings(tmp_path), FakeRunner(fail_after=0))
    ) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        outcome = _wait(client, job_id)
        assert outcome["status"] == "failed"
        assert outcome["error"]["code"] == "execution_failed"
        assert "boom" not in str(outcome)


def test_agent_error_and_size_limit(tmp_path: Path) -> None:
    """Agent-level failures retain normalized results, with a storage size cap."""
    messages = canned()
    messages[-1] = replace(
        messages[-1], is_error=True, subtype="error_max_turns"
    )
    with TestClient(
        create_app(_settings(tmp_path), FakeRunner(messages))
    ) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        outcome = _wait(client, job_id)
        assert outcome["status"] == "failed"
        assert outcome["error"]["code"] == "agent_failed"
        assert outcome["result"]["is_error"] is True
    with TestClient(
        create_app(_settings(tmp_path, jobs_max_result_bytes=16), FakeRunner())
    ) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        outcome = _wait(client, job_id)
        assert outcome["status"] == "failed"
        assert outcome["error"]["code"] == "result_too_large"
        assert outcome["result"] is None


def test_validation_and_auth(tmp_path: Path) -> None:
    """Validation errors do not enqueue jobs or bypass authentication."""
    with TestClient(
        create_app(_settings(tmp_path, jobs_max_payload_bytes=300))
    ) as client:
        assert client.post("/v1/jobs", json=BODY).status_code == 401
        assert client.post("/v1/jobs", headers=AUTH, json={**BODY, "prompt": ""}).status_code == 422
        assert client.post("/v1/jobs", headers=AUTH, json={**BODY, "timeout_seconds": 4000}).status_code == 400
        assert client.post("/v1/jobs", headers=AUTH, json={**BODY, "cwd": "../"}).status_code == 403
        assert client.post("/v1/jobs", headers=AUTH, json={**BODY, "sandbox": "invalid"}).status_code == 422
        assert client.post(
            "/v1/jobs", headers={**AUTH, "Idempotency-Key": "é"}, json=BODY
        ).status_code == 400
        assert client.post(
            "/v1/jobs", headers=AUTH, json={**BODY, "prompt": "x" * 1000}
        ).status_code == 413


def test_restart_reconciles_without_replay(tmp_path: Path) -> None:
    """Persisted in-flight records become interrupted rather than restarted."""
    settings = _settings(tmp_path)
    app = create_app(settings, FakeRunner())
    with TestClient(app):
        pass
    now = "2026-11-19T00:00:00+00:00"
    with sqlite3.connect(settings.jobs_db_path) as db:
        db.execute(
            """INSERT INTO jobs
               (job_id, caller, payload_hash, request_json, agent, workspace,
                status, created_at)
               VALUES ('orphan', ?, 'hash', '{}', 'claude', ?, 'running', ?)""",
            ("ignored", str(tmp_path), now),
        )
    with TestClient(create_app(settings, FakeRunner())):
        with sqlite3.connect(settings.jobs_db_path) as db:
            assert db.execute(
                "SELECT status FROM jobs WHERE job_id='orphan'"
            ).fetchone()[0] == "interrupted"


def test_retention_expires_keys(tmp_path: Path) -> None:
    """An expired completed job releases its idempotency key."""
    settings = _settings(tmp_path, jobs_retention_seconds=0)
    with TestClient(create_app(settings, FakeRunner())) as client:
        headers = {**AUTH, "Idempotency-Key": "short-lived"}
        old = client.post("/v1/jobs", headers=headers, json=BODY).json()["job_id"]
        # Read the committed state before it is eligible for GC on the next call.
        time.sleep(0.04)
        replacement = client.post("/v1/jobs", headers=headers, json=BODY)
        assert replacement.status_code == 202
        assert replacement.json()["job_id"] != old


class MeteredRunner(FakeRunner):
    """Count overlapping provider executions to verify workspace exclusivity."""

    active = 0
    peak = 0

    async def stream(
        self, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.09)
            async for message in super().stream(prompt, options):
                yield message
        finally:
            self.active -= 1


def test_workspace_jobs_do_not_overlap(tmp_path: Path) -> None:
    """Even separate job IDs cannot mutate overlapping workspaces at once."""
    runner = MeteredRunner()
    with TestClient(create_app(_settings(tmp_path), runner)) as client:
        one = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        two = client.post(
            "/v1/jobs", headers=AUTH, json={**BODY, "prompt": "next"}
        ).json()["job_id"]
        assert _wait(client, one)["status"] == "succeeded"
        assert _wait(client, two)["status"] == "succeeded"
        assert runner.peak == 1


def test_codex_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex uses its own execution adapter and captures normalized metadata."""

    async def fake_codex(**_kwargs: object) -> dict[str, Any]:
        return {"agent": "codex", "result": "done", "is_error": False}

    monkeypatch.setattr("agsrv.jobs.codex_agent.run_codex", fake_codex)
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post(
            "/v1/jobs", headers=AUTH, json={"agent": "codex", "prompt": "x"}
        )
        assert response.status_code == 202
        result = _wait(client, response.json()["job_id"])
        assert result["status"] == "succeeded"
        assert result["result"]["agent"] == "codex"
