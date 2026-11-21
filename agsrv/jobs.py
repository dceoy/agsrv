"""Durable single-worker job management for disconnected agent clients.

The SQLite journal survives client turnover and process restarts. Execution is
in-process and deliberately never replayed after an unclean server restart.
Run exactly one ASGI worker for a given jobs database.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import sqlite3
from contextlib import aclosing, asynccontextmanager, closing
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, BinaryIO, cast
from uuid import uuid4

from fastapi import HTTPException, status

from agsrv import codex
from agsrv.claude import AgentExecutionError, build_options, collect_result
from agsrv.sandbox import SandboxSetupError, service_home
from agsrv.schemas import CodexQueryRequest, QueryRequest
from agsrv.security import (
    check_permission_mode,
    check_sandbox_cwd,
    check_sandbox_request,
    resolve_cwd,
)
from agsrv.tmpguard import check_space, turn_tmpdir

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from claude_agent_sdk import Message
    from fastapi import FastAPI

    from agsrv.config import Settings

logger = logging.getLogger(__name__)
TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
MAX_IDEMPOTENCY_KEY_LENGTH = 128
RETRY_AFTER = {"Retry-After": "5"}


def _now() -> str:
    """Return a lexically sortable UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _principal(authorization: str | None) -> str:
    """Identify the authenticated bearer principal without persisting its key.

    Returns:
        SHA-256 principal identifier.
    """
    return hashlib.sha256((authorization or "anonymous").encode()).hexdigest()


def _canonical(body: QueryRequest | CodexQueryRequest) -> str:
    """Serialize validated requests in a stable canonical representation.

    Returns:
        Canonical JSON string.
    """
    return json.dumps(
        body.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )


class JobManager:
    """SQLite-backed, bounded in-process scheduler for asynchronous agent runs."""

    def __init__(self, app: FastAPI) -> None:
        """Create a scheduler for the supplied application."""
        self.app = app
        settings: Settings = app.state.settings
        self.path = settings.jobs_db_path
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.workspaces: set[Path] = set()
        self.workspace_condition = asyncio.Condition()
        self.running_slots = asyncio.Semaphore(
            settings.max_concurrency + settings.codex_max_concurrency
        )
        self.started = False
        self.stopping = False
        self._lock_file: BinaryIO | None = None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def start(self) -> None:
        """Initialize storage, owning its database for the worker lifetime.

        Raises:
            RuntimeError: If another worker already owns the database.
        """
        if self.started:
            return
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock").open("a+b")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock.close()
            msg = "jobs database is owned by another worker"
            raise RuntimeError(msg) from exc
        self._lock_file = lock
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            with closing(self._connect()) as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute(
                    """CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        caller TEXT NOT NULL,
                        idempotency_key TEXT,
                        payload_hash TEXT NOT NULL,
                        request_json TEXT NOT NULL,
                        agent TEXT NOT NULL,
                        workspace TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        result_json TEXT,
                        error_code TEXT,
                        cancel_requested INTEGER NOT NULL DEFAULT 0
                    )"""
                )
                db.execute(
                    """CREATE UNIQUE INDEX IF NOT EXISTS jobs_idempotency
                       ON jobs (caller, idempotency_key)
                       WHERE idempotency_key IS NOT NULL"""
                )
                db.execute(
                    "UPDATE jobs SET status='interrupted', finished_at=?, "
                    "error_code='server_interrupted' "
                    "WHERE status IN ('queued','running')",
                    (_now(),),
                )
                db.commit()
            self.path.chmod(0o600)
        except BaseException:
            lock.close()
            self._lock_file = None
            raise
        self.started = True
        self.stopping = False

    def _gc(self, db: sqlite3.Connection) -> None:
        """Expire only terminal records; their idempotency keys expire with them."""
        cutoff = (
            datetime.now(UTC)
            - timedelta(seconds=self.app.state.settings.jobs_retention_seconds)
        ).isoformat(timespec="milliseconds")
        db.execute(
            "DELETE FROM jobs WHERE status IN ('succeeded','failed','cancelled',"
            "'interrupted') AND finished_at < ?",
            (cutoff,),
        )

    def _validate(self, body: QueryRequest | CodexQueryRequest) -> Path:
        """Validate execution policy before creating a durable record.

        Returns:
            Resolved workspace path.

        Raises:
            HTTPException: If the execution policy is invalid.
        """
        settings: Settings = self.app.state.settings
        cwd = resolve_cwd(settings, body.cwd)
        if body.timeout_seconds is not None and (
            body.timeout_seconds > settings.max_timeout_seconds
        ):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "timeout_seconds exceeds server maximum "
                f"{settings.max_timeout_seconds}",
            )
        if isinstance(body, QueryRequest):
            check_permission_mode(settings, body.permission_mode)
            check_sandbox_request(settings, body.sandbox, body.permission_mode)
            if body.sandbox:
                try:
                    check_sandbox_cwd(cwd, service_home())
                except SandboxSetupError as exc:
                    raise HTTPException(
                        status.HTTP_503_SERVICE_UNAVAILABLE,
                        "sandbox unavailable",
                    ) from exc
        return cwd

    def submit(
        self,
        body: QueryRequest | CodexQueryRequest,
        authorization: str | None,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        """Atomically deduplicate and enqueue a bounded job.

        Returns:
            Accepted job identifier and polling URL.

        Raises:
            HTTPException: If validation or queue capacity checks fail.
        """
        self.start()
        if self.stopping:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "server stopping")
        if idempotency_key is not None and (
            not 1 <= len(idempotency_key) <= MAX_IDEMPOTENCY_KEY_LENGTH
            or not idempotency_key.isascii()
            or not idempotency_key.isprintable()
        ):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid Idempotency-Key")
        cwd = self._validate(body)
        payload = _canonical(body)
        settings: Settings = self.app.state.settings
        if len(payload.encode()) > settings.jobs_max_payload_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "job payload too large"
            )
        caller = _principal(authorization)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        job_id = uuid4().hex
        created = _now()
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            self._gc(db)
            existing = (
                db.execute(
                    "SELECT job_id, payload_hash, status FROM jobs "
                    "WHERE caller=? AND idempotency_key=?",
                    (caller, idempotency_key),
                ).fetchone()
                if idempotency_key is not None
                else None
            )
            if existing is not None:
                if existing["payload_hash"] != digest:
                    raise HTTPException(
                        status.HTTP_409_CONFLICT, "Idempotency-Key payload mismatch"
                    )
                return self._accepted(existing["job_id"], existing["status"])
            check_space(settings, cwd)
            pending = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
            if pending >= settings.jobs_max_pending:
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    "job queue is full",
                    headers=RETRY_AFTER,
                )
            db.execute(
                """INSERT INTO jobs
                   (job_id, caller, idempotency_key, payload_hash, request_json,
                    agent, workspace, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?)""",
                (
                    job_id,
                    caller,
                    idempotency_key,
                    digest,
                    payload,
                    body.agent,
                    str(cwd),
                    created,
                ),
            )
            db.commit()
        task = asyncio.create_task(self._run(job_id, body, cwd))
        self.tasks[job_id] = task
        task.add_done_callback(lambda _: self.tasks.pop(job_id, None))
        return self._accepted(job_id, "queued")

    @staticmethod
    def _accepted(job_id: str, state: str) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "status": state,
            "status_url": f"/v1/jobs/{job_id}",
        }

    def _row(self, job_id: str, authorization: str | None) -> sqlite3.Row:
        self.start()
        with closing(self._connect()) as db:
            self._gc(db)
            db.commit()
            row = db.execute(
                "SELECT * FROM jobs WHERE job_id=? AND caller=?",
                (job_id, _principal(authorization)),
            ).fetchone()
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        return row

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        result = json.loads(row["result_json"]) if row["result_json"] else None
        code = row["error_code"]
        return {
            "job_id": row["job_id"],
            "agent": row["agent"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "result": result,
            "error": {"code": code, "message": code.replace("_", " ")}
            if code
            else None,
        }

    def get(self, job_id: str, authorization: str | None) -> dict[str, Any]:
        """Retrieve a job only within its authenticated bearer scope.

        Returns:
            Public job status and result.
        """
        return self._public(self._row(job_id, authorization))

    def cancel(self, job_id: str, authorization: str | None) -> dict[str, Any]:
        """Request cancellation; queued jobs are cancelled atomically.

        Returns:
            The current public job status.
        """
        row = self._row(job_id, authorization)
        if row["status"] in TERMINAL:
            return self._public(row)
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """UPDATE jobs SET cancel_requested=1,
                   status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,
                   finished_at=CASE WHEN status='queued' THEN ? ELSE finished_at END
                   WHERE job_id=? AND caller=? AND status IN ('queued','running')""",
                (_now(), job_id, _principal(authorization)),
            )
            db.commit()
        task = self.tasks.get(job_id)
        if task is not None:
            task.cancel()
        return self.get(job_id, authorization)

    def _mark_running(self, job_id: str) -> bool:
        with closing(self._connect()) as db:
            cursor = db.execute(
                "UPDATE jobs SET status='running', started_at=? "
                "WHERE job_id=? AND status='queued' AND cancel_requested=0",
                (_now(), job_id),
            )
            db.commit()
            return cursor.rowcount == 1

    def _finish(
        self, job_id: str, outcome: str, result: dict[str, Any] | None, code: str | None
    ) -> None:
        settings: Settings = self.app.state.settings
        serialized = (
            json.dumps(result, separators=(",", ":"), ensure_ascii=False)
            if result is not None
            else None
        )
        if (
            serialized is not None
            and len(serialized.encode()) > settings.jobs_max_result_bytes
        ):
            compact = dict(cast("dict[str, Any]", result))
            if "messages" in compact:
                compact.pop("messages")
                compact["messages_truncated"] = True
                serialized = json.dumps(
                    compact, separators=(",", ":"), ensure_ascii=False
                )
            if len(serialized.encode()) > settings.jobs_max_result_bytes:
                serialized = json.dumps(
                    {
                        "agent": compact.get("agent"),
                        "is_error": outcome != "succeeded",
                        "result_truncated": True,
                    },
                    separators=(",", ":"),
                )
            if len(serialized.encode()) > settings.jobs_max_result_bytes:
                outcome, code, serialized = "failed", "result_too_large", None
        with closing(self._connect()) as db:
            db.execute(
                """UPDATE jobs SET status=?, finished_at=?, result_json=?, error_code=?
                   WHERE job_id=? AND status IN ('queued','running')""",
                (outcome, _now(), serialized, code, job_id),
            )
            db.commit()

    async def _acquire_workspace(self, cwd: Path) -> None:
        async with self.workspace_condition:
            await self.workspace_condition.wait_for(
                lambda: (
                    not any(
                        cwd.is_relative_to(active) or active.is_relative_to(cwd)
                        for active in self.workspaces
                    )
                )
            )
            self.workspaces.add(cwd)

    async def _release_workspace(self, cwd: Path) -> None:
        async with self.workspace_condition:
            self.workspaces.discard(cwd)
            self.workspace_condition.notify_all()

    @asynccontextmanager
    async def workspace(self, cwd: Path) -> AsyncGenerator[None]:
        """Hold a shared exclusive workspace lease for any agent operation."""
        await self._acquire_workspace(cwd)
        try:
            yield
        finally:
            await self._release_workspace(cwd)

    async def _execute(
        self, body: QueryRequest | CodexQueryRequest, cwd: Path
    ) -> dict[str, Any]:
        settings: Settings = self.app.state.settings
        if isinstance(body, QueryRequest):
            async with self.app.state.semaphore:
                check_space(settings, cwd)
                async with turn_tmpdir(settings.workspace_root, cwd) as tmpdir:
                    home = service_home() if body.sandbox else None
                    options = build_options(body, settings, cwd, home, tmpdir)
                    runner = self.app.state.runner
                    stream = runner.stream(body.prompt, options)
                    async with aclosing(
                        cast("AsyncGenerator[Message]", stream)
                    ) as messages:
                        result = await collect_result(
                            messages, include_messages=body.include_messages
                        )
                    return result.model_dump(mode="json")
        if not settings.codex_enabled:
            msg = "Codex unavailable"
            raise RuntimeError(msg)
        async with self.app.state.codex_semaphore:
            check_space(settings, cwd)
            async with turn_tmpdir(settings.workspace_root, cwd) as tmpdir:
                return await codex.run_codex(
                    prompt=body.prompt,
                    cwd=str(cwd),
                    model=body.model,
                    effort=body.effort,
                    sandbox=body.options.sandbox,
                    network_access=body.options.network_access,
                    binary=settings.codex_bin,
                    tmpdir=str(tmpdir),
                )

    async def _run(
        self, job_id: str, body: QueryRequest | CodexQueryRequest, cwd: Path
    ) -> None:
        acquired = False
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            timeout = (
                body.timeout_seconds or self.app.state.settings.max_timeout_seconds
            )
            async with asyncio.timeout(timeout):
                await self._acquire_workspace(cwd)
                acquired = True
                async with self.running_slots:
                    if not self._mark_running(job_id):
                        return
                    result = await self._execute(body, cwd)
            if result.get("is_error"):
                self._finish(job_id, "failed", result, "agent_failed")
            else:
                self._finish(job_id, "succeeded", result, None)
        except asyncio.CancelledError:
            self._finish(
                job_id,
                "interrupted" if self.stopping else "cancelled",
                None,
                "server_interrupted" if self.stopping else "cancelled",
            )
        except TimeoutError:
            self._finish(job_id, "failed", None, "deadline_exceeded")
        except SandboxSetupError:
            self._finish(job_id, "failed", None, "sandbox_unavailable")
        except AgentExecutionError:
            self._finish(job_id, "failed", None, "agent_result_missing")
        except Exception:
            logger.exception("asynchronous agent job failed: %s", job_id)
            self._finish(job_id, "failed", None, "execution_failed")
        finally:
            if acquired:
                await self._release_workspace(cwd)

    async def stop(self) -> None:
        """Cancel owned jobs and relinquish the exclusive database lock."""
        self.stopping = True
        try:
            tasks = list(self.tasks.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if self.started:
                with closing(self._connect()) as db:
                    db.execute(
                        "UPDATE jobs SET status='interrupted', finished_at=?, "
                        "error_code='server_interrupted' "
                        "WHERE status IN ('queued','running')",
                        (_now(),),
                    )
                    db.commit()
        finally:
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None
            self.started = False
