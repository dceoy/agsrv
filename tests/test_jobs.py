"""Integration coverage for durable asynchronous jobs without provider logins."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from agsrv.config import Settings
from agsrv.main import create_app
from tests.test_main import FakeRunner, canned

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from claude_agent_sdk import Message

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
        assert finished["created_at"]
        assert finished["started_at"]
        assert finished["finished_at"]
        assert finished["error"] is None
        assert runner.stream_calls == 1
        assert (
            client.post(f"/v1/jobs/{job_id}/cancel", headers=AUTH).json()["status"]
            == "succeeded"
        )
        assert (
            client.get(
                f"/v1/jobs/{job_id}", headers={"Authorization": "Bearer other"}
            ).status_code
            == 401
        )
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
    with TestClient(create_app(_settings(tmp_path), FakeRunner(delay=0.2))) as client:
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
    messages[-1] = replace(messages[-1], is_error=True, subtype="error_max_turns")
    with TestClient(create_app(_settings(tmp_path), FakeRunner(messages))) as client:
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
        assert (
            client.post(
                "/v1/jobs", headers=AUTH, json={**BODY, "prompt": ""}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/jobs", headers=AUTH, json={**BODY, "timeout_seconds": 4000}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/v1/jobs", headers=AUTH, json={**BODY, "cwd": "../"}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/v1/jobs", headers=AUTH, json={**BODY, "sandbox": "invalid"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/jobs", headers={**AUTH, "Idempotency-Key": "x" * 129}, json=BODY
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/v1/jobs", headers=AUTH, json={**BODY, "prompt": "x" * 1000}
            ).status_code
            == 413
        )


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
    with (
        TestClient(create_app(settings, FakeRunner())),
        sqlite3.connect(settings.jobs_db_path) as db,
    ):
        assert (
            db.execute("SELECT status FROM jobs WHERE job_id='orphan'").fetchone()[0]
            == "interrupted"
        )


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
        """Yield metered agent events while tracking concurrent runs."""
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
        await asyncio.sleep(0)
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

def test_job_admission_failure_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enforce shutdown, sandbox initialization, and key validation failures."""
    from fastapi import HTTPException  # ruff: ignore[import-outside-toplevel]
    from agsrv.sandbox import SandboxSetupError  # ruff: ignore[import-outside-toplevel]
    from agsrv.schemas import QueryRequest  # ruff: ignore[import-outside-toplevel]

    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        body = QueryRequest(agent="claude", prompt="x", sandbox=False)
        for bad in ("é", "\t", "", "x" * 129):
            with pytest.raises(HTTPException) as exc:
                app.state.jobs.submit(body, "Bearer secret", bad)
            assert exc.value.status_code == 400

        def no_home() -> Path:
            raise SandboxSetupError("missing home")

        monkeypatch.setattr("agsrv.jobs.service_home", no_home)
        assert client.post(
            "/v1/jobs", headers=AUTH, json={"agent": "claude", "prompt": "x"}
        ).status_code == 503

        app.state.jobs.stopping = True
        assert client.post("/v1/jobs", headers=AUTH, json=BODY).status_code == 503


def test_cancel_without_tracked_task(tmp_path: Path) -> None:
    """Cancel a stored queued job even when no in-memory task exists."""
    import hashlib  # ruff: ignore[import-outside-toplevel]

    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        principal = hashlib.sha256("Bearer secret".encode()).hexdigest()
        with sqlite3.connect(app.state.jobs.path) as db:
            db.execute(
                """INSERT INTO jobs
                    (job_id, caller, payload_hash, request_json, agent, workspace,
                     status, created_at)
                    VALUES ('orphan-queue', ?, 'hash', '{}', 'claude', ?, 'queued', ?)""",
                (principal, str(tmp_path), "2026-11-20T13:00:00.000+00:00"),
            )
        response = client.post("/v1/jobs/orphan-queue/cancel", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"


def test_disabled_codex_and_missing_claude_result(tmp_path: Path) -> None:
    """Classify terminal failures without leaking backend internals."""
    with TestClient(
        create_app(_settings(tmp_path, codex_enabled=False))
    ) as client:
        job_id = client.post(
            "/v1/jobs", headers=AUTH, json={"agent": "codex", "prompt": "x"}
        ).json()["job_id"]
        result = _wait(client, job_id)
        assert result["status"] == "failed"
        assert result["error"]["code"] == "execution_failed"

    with TestClient(
        create_app(_settings(tmp_path), FakeRunner(canned()[:2]))
    ) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        result = _wait(client, job_id)
        assert result["status"] == "failed"
        assert result["error"]["code"] == "agent_result_missing"


def test_sandbox_runtime_setup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Represent failed sandbox prerequisites as a stable error code."""
    from agsrv.sandbox import SandboxSetupError  # ruff: ignore[import-outside-toplevel]

    def broken(*_args: object, **_kwargs: object) -> None:
        raise SandboxSetupError("bundled CLI is unavailable")

    monkeypatch.setattr("agsrv.jobs.build_options", broken)
    with TestClient(create_app(_settings(tmp_path), FakeRunner())) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        result = _wait(client, job_id)
        assert result["status"] == "failed"
        assert result["error"]["code"] == "sandbox_unavailable"


@pytest.mark.asyncio
async def test_orphaned_run_and_unstarted_manager_stop(tmp_path: Path) -> None:
    """A non-existent record cannot start, and uninitialized shutdown is safe."""
    from agsrv.schemas import QueryRequest  # ruff: ignore[import-outside-toplevel]

    app = create_app(_settings(tmp_path))
    manager = app.state.jobs
    manager.start()
    await manager._run(  # pyright: ignore[reportPrivateUsage]
        "does-not-exist",
        QueryRequest(agent="claude", prompt="x", sandbox=False),
        tmp_path,
    )
    await manager.stop()
    unstarted = create_app(_settings(tmp_path)).state.jobs
    await unstarted.stop()


def test_shutdown_interrupts_active_jobs(tmp_path: Path) -> None:
    """Shutdown drains in-memory tasks and leaves an interrupted record."""
    app = create_app(_settings(tmp_path), FakeRunner(delay=30.0))
    with TestClient(app) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
    with sqlite3.connect(app.state.jobs.path) as db:
        assert db.execute(
            "SELECT status FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()[0] == "interrupted"


def test_cancel_queued_workspace_job(tmp_path: Path) -> None:
    """Cancelling a workspace waiter does not release an unowned lock."""
    app = create_app(_settings(tmp_path), FakeRunner(delay=0.35))
    with TestClient(app) as client:
        first = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        for _ in range(100):
            if client.get(f"/v1/jobs/{first}", headers=AUTH).json()["status"] == "running":
                break
            time.sleep(0.005)
        second = client.post(
            "/v1/jobs", headers=AUTH, json={**BODY, "prompt": "other"}
        ).json()["job_id"]
        assert client.post(
            f"/v1/jobs/{second}/cancel", headers=AUTH
        ).status_code == 200
        assert _wait(client, second)["status"] == "cancelled"
        assert _wait(client, first)["status"] == "succeeded"
