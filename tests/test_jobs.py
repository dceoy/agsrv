"""Integration coverage for durable asynchronous jobs without provider logins."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agsrv.config import Settings
from agsrv.main import create_app
from agsrv.sandbox import SandboxSetupError
from agsrv.schemas import QueryRequest
from tests.test_main import FakeRunner, canned

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from claude_agent_sdk import ClaudeAgentOptions, Message

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

    monkeypatch.setattr("agsrv.jobs.codex.run_codex", fake_codex)
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
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        body = QueryRequest(agent="claude", prompt="x", sandbox=False)
        for bad in ("é", "\t", "", "x" * 129):
            with pytest.raises(HTTPException) as exc:
                app.state.jobs.submit(body, "Bearer secret", bad)
            assert exc.value.status_code == 400

        def no_home() -> Path:
            msg = "missing home"
            raise SandboxSetupError(msg)

        monkeypatch.setattr("agsrv.jobs.service_home", no_home)
        assert (
            client.post(
                "/v1/jobs", headers=AUTH, json={"agent": "claude", "prompt": "x"}
            ).status_code
            == 503
        )

        app.state.jobs.stopping = True
        assert client.post("/v1/jobs", headers=AUTH, json=BODY).status_code == 503


def test_cancel_without_tracked_task(tmp_path: Path) -> None:
    """Cancel a stored queued job even when no in-memory task exists."""
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        principal = hashlib.sha256(b"Bearer secret").hexdigest()
        with sqlite3.connect(app.state.jobs.path) as db:
            db.execute(
                """INSERT INTO jobs
                    (job_id, caller, payload_hash, request_json, agent, workspace,
                     status, created_at)
                    VALUES ('orphan-queue', ?, 'hash', '{}', 'claude', ?,
                            'queued', ?)""",
                (principal, str(tmp_path), "2026-11-20T13:00:00.000+00:00"),
            )
        response = client.post("/v1/jobs/orphan-queue/cancel", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"


def test_disabled_codex_and_missing_claude_result(tmp_path: Path) -> None:
    """Classify terminal failures without leaking backend internals."""
    with TestClient(create_app(_settings(tmp_path, codex_enabled=False))) as client:
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

    def broken(*_args: object, **_kwargs: object) -> None:
        msg = "bundled CLI is unavailable"
        raise SandboxSetupError(msg)

    monkeypatch.setattr("agsrv.jobs.build_options", broken)
    with TestClient(create_app(_settings(tmp_path), FakeRunner())) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        result = _wait(client, job_id)
        assert result["status"] == "failed"
        assert result["error"]["code"] == "sandbox_unavailable"


@pytest.mark.asyncio
async def test_orphaned_run_and_unstarted_manager_stop(tmp_path: Path) -> None:
    """A non-existent record cannot start, and uninitialized shutdown is safe."""
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
        assert (
            db.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()[
                0
            ]
            == "interrupted"
        )


def test_cancel_queued_workspace_job(tmp_path: Path) -> None:
    """Cancelling a workspace waiter does not release an unowned lock."""
    app = create_app(_settings(tmp_path), FakeRunner(delay=0.35))
    with TestClient(app) as client:
        first = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        for _ in range(100):
            if (
                client.get(f"/v1/jobs/{first}", headers=AUTH).json()["status"]
                == "running"
            ):
                break
            time.sleep(0.005)
        second = client.post(
            "/v1/jobs", headers=AUTH, json={**BODY, "prompt": "other"}
        ).json()["job_id"]
        assert client.post(f"/v1/jobs/{second}/cancel", headers=AUTH).status_code == 200
        assert _wait(client, second)["status"] == "cancelled"
        assert _wait(client, first)["status"] == "succeeded"


def test_job_database_rejects_concurrent_owners(tmp_path: Path) -> None:
    """A second worker must not interrupt jobs owned by the first."""
    settings = _settings(tmp_path)
    first = create_app(settings, FakeRunner(delay=0.2))
    second = create_app(settings, FakeRunner())
    with TestClient(first) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        with (
            pytest.raises(RuntimeError, match="owned by another worker"),
            TestClient(second),
        ):
            pass
        assert _wait(client, job_id)["status"] == "succeeded"
    # A clean shutdown relinquishes ownership for the replacement worker.
    with TestClient(second) as client:
        assert client.get("/health").status_code == 200


def test_database_lock_released_after_startup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed schema initialization must not strand the ownership lock."""
    settings = _settings(tmp_path)
    manager = create_app(settings).state.jobs

    def fail_connection() -> None:
        msg = "database unavailable"
        raise sqlite3.OperationalError(msg)

    monkeypatch.setattr(manager, "_connect", fail_connection)
    with pytest.raises(sqlite3.OperationalError, match="database unavailable"):
        manager.start()
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200


def test_sync_claude_query_shares_job_workspace_lock(tmp_path: Path) -> None:
    """A sync query must not execute concurrently with a job in the same cwd."""
    runner = MeteredRunner()
    with TestClient(create_app(_settings(tmp_path), runner)) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        query = client.post("/v1/query", headers=AUTH, json=BODY)
        assert query.status_code == 200
        assert _wait(client, job_id)["status"] == "succeeded"
        assert runner.stream_calls == 2
        assert runner.peak == 1


def test_sync_codex_query_shares_job_workspace_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The detached Codex operation also holds the job workspace lease."""
    codex_active = False

    async def fake_available(*_args: object, **_kwargs: object) -> bool:
        await asyncio.sleep(0)
        return True

    async def fake_codex(**_kwargs: object) -> dict[str, Any]:
        nonlocal codex_active
        assert runner.active == 0
        codex_active = True
        try:
            await asyncio.sleep(0.08)
            assert runner.active == 0
            return {
                "agent": "codex",
                "result": "done",
                "model": None,
                "is_error": False,
                "duration_ms": 0,
                "usage": None,
                "backend": {},
            }
        finally:
            codex_active = False

    class CheckedRunner(MeteredRunner):
        async def stream(
            self, prompt: str, options: ClaudeAgentOptions
        ) -> AsyncIterator[Message]:
            assert not codex_active
            async for message in super().stream(prompt, options):
                assert not codex_active
                yield message

    runner = CheckedRunner()
    monkeypatch.setattr("agsrv.main._codex_available", fake_available)
    monkeypatch.setattr("agsrv.main.codex.run_codex", fake_codex)
    with TestClient(create_app(_settings(tmp_path), runner)) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        response = client.post(
            "/v1/query",
            headers=AUTH,
            json={"agent": "codex", "prompt": "inspect"},
        )
        assert response.status_code == 200
        assert _wait(client, job_id)["status"] == "succeeded"
        assert runner.peak == 1


@pytest.mark.parametrize(
    ("max_bytes", "answer", "expected_flag", "with_messages"),
    [
        (350, "done", "messages_truncated", True),
        (100, "x" * 2048, "result_truncated", True),
        (100, "x" * 2048, "result_truncated", False),
    ],
)
def test_oversized_results_preserve_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_bytes: int,
    answer: str,
    expected_flag: str,
    with_messages: bool,
) -> None:
    """A completed coding run must not become failed due to an oversized transcript."""
    app = create_app(_settings(tmp_path, jobs_max_result_bytes=max_bytes))

    async def oversized_result(_body: object, _cwd: object) -> dict[str, Any]:
        await asyncio.sleep(0)
        result: dict[str, Any] = {
            "agent": "claude",
            "is_error": False,
            "result": answer,
        }
        if with_messages:
            result["messages"] = [{"text": "z" * 2048}]
        return result

    monkeypatch.setattr(app.state.jobs, "_execute", oversized_result)
    with TestClient(app) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        outcome = _wait(client, job_id)
        assert outcome["status"] == "succeeded"
        assert outcome["error"] is None
        assert outcome["result"][expected_flag] is True
        if expected_flag == "messages_truncated":
            assert outcome["result"]["result"] == "done"
        else:
            assert outcome["result"]["agent"] == "claude"
            assert outcome["result"]["result_truncated"] is True


@pytest.mark.asyncio
async def test_cancel_workspace_waiter_before_acquisition(tmp_path: Path) -> None:
    """Cancelled waiters never release another execution's workspace lease."""
    manager = create_app(_settings(tmp_path)).state.jobs
    manager.start()
    async with manager.workspace(tmp_path):
        task = asyncio.create_task(
            manager._run(  # pyright: ignore[reportPrivateUsage]
                "not-yet-running",
                QueryRequest(agent="claude", prompt="x", sandbox=False),
                tmp_path,
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        await task
    await manager.stop()


def test_reused_app_lifespan_reacquires_database_lock(tmp_path: Path) -> None:
    """Restarting the same ASGI app must reacquire the owner lock."""
    app = create_app(_settings(tmp_path), FakeRunner())
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    with TestClient(app) as client:
        job_id = client.post("/v1/jobs", headers=AUTH, json=BODY).json()["job_id"]
        assert _wait(client, job_id)["status"] == "succeeded"
