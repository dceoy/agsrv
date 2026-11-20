"""FastAPI application for the stateless Claude Agent API."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import aclosing, asynccontextmanager, suppress
from typing import TYPE_CHECKING, Annotated, Any, TypeVar, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import Field

from agsrv import codex_agent, muse_agent
from agsrv.agent import (
    AgentExecutionError,
    AgentRunner,
    SdkAgentRunner,
    build_options,
    collect_result,
)
from agsrv.config import Settings
from agsrv.jobs import JobManager
from agsrv.sandbox import SandboxSetupError, service_home
from agsrv.schemas import (
    CodexQueryRequest,
    CodexQueryResponse,
    InfoResponse,
    MuseQueryRequest,
    MuseQueryResponse,
    QueryRequest,
    QueryResponse,
)
from agsrv.security import (
    auth_dependency,
    check_permission_mode,
    check_sandbox_cwd,
    check_sandbox_request,
    resolve_cwd,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine

    from claude_agent_sdk import ClaudeAgentOptions, Message

logger = logging.getLogger(__name__)

VERSION = "0.3.0"
DISCONNECT_POLL_INTERVAL = 0.1
CODEX_PROBE_TIMEOUT = 10.0
MUSE_PROBE_TIMEOUT = 10.0

# Keep detached Codex cleanup alive if the request task is cancelled.
_pump_tasks: set[asyncio.Task[Any]] = set()

T = TypeVar("T")

Auth = Depends(auth_dependency)


async def _prepare(
    request: Request, body: QueryRequest
) -> tuple[ClaudeAgentOptions, float, asyncio.Semaphore]:
    """Validate a request and acquire a concurrency slot.

    Returns:
        SDK options, effective timeout, and the semaphore (already acquired).

    Raises:
        HTTPException: On validation, permission, or capacity errors.
    """
    settings: Settings = request.app.state.settings
    cwd = resolve_cwd(settings, body.cwd)
    check_permission_mode(settings, body.permission_mode)
    check_sandbox_request(settings, body.sandbox, body.permission_mode)
    home = None
    home_error: SandboxSetupError | None = None
    if body.sandbox:
        try:
            home = service_home()
        except SandboxSetupError as exc:
            home_error = exc  # 503 is deferred so that 429 takes precedence
        else:
            check_sandbox_cwd(cwd, home)
    timeout = (
        settings.max_timeout_seconds
        if body.timeout_seconds is None
        else body.timeout_seconds
    )
    if timeout > settings.max_timeout_seconds:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"timeout_seconds exceeds server maximum {settings.max_timeout_seconds}",
        )
    semaphore: asyncio.Semaphore = request.app.state.semaphore
    if semaphore.locked():
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "agent concurrency limit reached"
        )
    if home_error is not None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, str(home_error)
        ) from home_error
    await semaphore.acquire()  # immediate: not locked, so no suspension
    try:
        options = build_options(body, settings, cwd, home)
    except BaseException as exc:
        semaphore.release()
        if isinstance(exc, SandboxSetupError):
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        raise
    return options, timeout, semaphore


class _SlotLease:
    """Release an acquired semaphore slot at most once."""

    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self.semaphore = semaphore
        self.released = False
        self.claimed = False

    def release(self) -> None:
        """Return the slot if it is still held."""
        if not self.released:
            self.released = True
            self.semaphore.release()


@asynccontextmanager
async def _codex_slot(app: FastAPI) -> AsyncGenerator[_SlotLease, None]:
    """Acquire capacity for an operation.

    Yields:
        After acquiring the slot.

    Raises:
        HTTPException: If capacity is exhausted.
    """
    semaphore: asyncio.Semaphore = app.state.codex_semaphore
    if semaphore.locked():
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "agent concurrency limit reached"
        )
    await semaphore.acquire()
    lease = _SlotLease(semaphore)
    try:
        yield lease
    finally:
        if not lease.claimed:
            lease.release()


async def _codex_available(app: FastAPI, *, lease: _SlotLease | None = None) -> bool:
    settings: Settings = app.state.settings
    if not settings.codex_enabled:
        return False
    if time.monotonic() < app.state.codex_probe_expires:
        return cast("bool", app.state.codex_healthy)
    async with app.state.codex_probe_lock:
        now = time.monotonic()
        if now >= app.state.codex_probe_expires:
            if lease is None:
                async with _codex_slot(app) as acquired:
                    app.state.codex_healthy = await _timed_probe(app, acquired)
            else:
                app.state.codex_healthy = await _timed_probe(app, lease)
            app.state.codex_probe_expires = time.monotonic() + 30
    return cast("bool", app.state.codex_healthy)


async def _timed_probe(app: FastAPI, lease: _SlotLease) -> bool:
    """Release the caller at the deadline while cleanup retains capacity.

    Returns:
        The probe result, or false on timeout.

    Raises:
        asyncio.CancelledError: If the caller is cancelled.
    """
    task = asyncio.create_task(codex_agent.probe(app.state.settings.codex_bin))
    try:
        done, _ = await asyncio.wait({task}, timeout=CODEX_PROBE_TIMEOUT)
    except asyncio.CancelledError:
        _detach_codex(task, lease)
        app.state.codex_healthy = False
        expires = time.monotonic() + 30
        app.state.codex_probe_expires = expires

        def invalidate(_operation: asyncio.Task[bool]) -> None:
            if app.state.codex_probe_expires == expires:
                app.state.codex_probe_expires = 0

        task.add_done_callback(invalidate)
        raise
    if done:
        return task.result()
    _detach_codex(task, lease)
    return False


def _detach_codex(task: asyncio.Task[Any], lease: _SlotLease) -> None:
    """Transfer capacity to a cancelled operation until its cleanup finishes."""
    lease.claimed = True
    _pump_tasks.add(task)

    def finished(operation: asyncio.Task[Any]) -> None:
        _pump_tasks.discard(operation)
        lease.release()
        with suppress(asyncio.CancelledError):
            operation.exception()

    task.add_done_callback(finished)
    task.cancel()


async def _await_codex(operation: Coroutine[Any, Any, T], lease: _SlotLease) -> T:
    """Run an operation while allowing its cleanup to outlive the caller.

    Returns:
        The operation result.

    Raises:
        asyncio.CancelledError: If the caller is cancelled.
    """
    task = asyncio.create_task(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        _detach_codex(task, lease)
        raise


async def _codex_query(body: CodexQueryRequest, request: Request) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    cwd = resolve_cwd(settings, body.cwd)
    timeout = body.timeout_seconds or settings.max_timeout_seconds
    if timeout > settings.max_timeout_seconds:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"timeout_seconds exceeds server maximum {settings.max_timeout_seconds}",
        )
    semaphore: asyncio.Semaphore = request.app.state.codex_semaphore
    if semaphore.locked():
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "agent concurrency limit reached"
        )
    await semaphore.acquire()
    lease = _SlotLease(semaphore)

    async def execute() -> dict[str, Any]:
        async with request.app.state.jobs.workspace(cwd):
            return await codex_agent.run_codex(
                prompt=body.prompt,
                cwd=str(cwd),
                model=body.model,
                effort=body.effort,
                sandbox=body.options.sandbox,
                network_access=body.options.network_access,
                binary=settings.codex_bin,
            )

    try:
        async with asyncio.timeout(timeout):
            if not await _codex_available(request.app, lease=lease):
                raise HTTPException(  # ruff: ignore[raise-within-try]
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    "Codex is disabled or unavailable",
                )
            return await _await_codex(execute(), lease)
    except TimeoutError as exc:
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT, "agent run timed out"
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Codex execution failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "agent execution failed"
        ) from exc
    finally:
        if not lease.claimed:
            lease.release()


async def _codex_query_with_disconnect(
    body: CodexQueryRequest, request: Request
) -> dict[str, Any] | Response:
    task = cast("asyncio.Task[object]", asyncio.current_task())
    client_gone = False

    async def watch_disconnect() -> None:
        nonlocal client_gone
        while not await request.is_disconnected():  # ruff: ignore[async-busy-wait]
            await asyncio.sleep(DISCONNECT_POLL_INTERVAL)
        client_gone = True
        task.cancel()

    watcher = asyncio.create_task(watch_disconnect())
    try:
        return await _codex_query(body, request)
    except asyncio.CancelledError:
        if client_gone and task.uncancel() == 0:
            return Response(status_code=499)
        raise
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher


@asynccontextmanager
async def _muse_slot(app: FastAPI) -> AsyncGenerator[_SlotLease, None]:
    """Acquire Muse capacity.

    Yields:
        After acquiring the slot.

    Raises:
        HTTPException: If capacity is exhausted.
    """
    semaphore: asyncio.Semaphore = app.state.muse_semaphore
    if semaphore.locked():
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "agent concurrency limit reached"
        )
    await semaphore.acquire()
    lease = _SlotLease(semaphore)
    try:
        yield lease
    finally:
        if not lease.claimed:
            lease.release()


async def _muse_available(app: FastAPI, *, lease: _SlotLease | None = None) -> bool:
    settings: Settings = app.state.settings
    if not settings.muse_enabled:
        return False
    if time.monotonic() < app.state.muse_probe_expires:
        return cast("bool", app.state.muse_healthy)
    async with app.state.muse_probe_lock:
        now = time.monotonic()
        if now >= app.state.muse_probe_expires:
            if lease is None:
                async with _muse_slot(app) as acquired:
                    app.state.muse_healthy = await _timed_muse_probe(app, acquired)
            else:
                app.state.muse_healthy = await _timed_muse_probe(app, lease)
            app.state.muse_probe_expires = time.monotonic() + 30
    return cast("bool", app.state.muse_healthy)


async def _timed_muse_probe(app: FastAPI, lease: _SlotLease) -> bool:
    """Release the caller at the deadline while cleanup retains capacity.

    Returns:
        The probe result, or false on timeout.

    Raises:
        asyncio.CancelledError: If the caller is cancelled.
    """
    task = asyncio.create_task(muse_agent.probe(app.state.settings.muse_bin))
    try:
        done, _ = await asyncio.wait({task}, timeout=MUSE_PROBE_TIMEOUT)
    except asyncio.CancelledError:
        _detach_muse(task, lease)
        app.state.muse_healthy = False
        expires = time.monotonic() + 30
        app.state.muse_probe_expires = expires

        def invalidate(_operation: asyncio.Task[bool]) -> None:
            if app.state.muse_probe_expires == expires:
                app.state.muse_probe_expires = 0

        task.add_done_callback(invalidate)
        raise
    if done:
        return task.result()
    _detach_muse(task, lease)
    return False


def _detach_muse(task: asyncio.Task[Any], lease: _SlotLease) -> None:
    """Transfer capacity to a cancelled operation until its cleanup finishes."""
    lease.claimed = True
    _pump_tasks.add(task)

    def finished(operation: asyncio.Task[Any]) -> None:
        _pump_tasks.discard(operation)
        lease.release()
        with suppress(asyncio.CancelledError):
            operation.exception()

    task.add_done_callback(finished)
    task.cancel()


async def _await_muse(operation: Coroutine[Any, Any, T], lease: _SlotLease) -> T:
    """Run an operation while allowing its cleanup to outlive the caller.

    Returns:
        The operation result.

    Raises:
        asyncio.CancelledError: If the caller is cancelled.
    """
    task = asyncio.create_task(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        _detach_muse(task, lease)
        raise


async def _muse_query(body: MuseQueryRequest, request: Request) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    cwd = resolve_cwd(settings, body.cwd)
    timeout = body.timeout_seconds or settings.max_timeout_seconds
    if timeout > settings.max_timeout_seconds:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"timeout_seconds exceeds server maximum {settings.max_timeout_seconds}",
        )
    semaphore: asyncio.Semaphore = request.app.state.muse_semaphore
    if semaphore.locked():
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "agent concurrency limit reached"
        )
    await semaphore.acquire()
    lease = _SlotLease(semaphore)

    async def execute() -> dict[str, Any]:
        async with request.app.state.jobs.workspace(cwd):
            return await muse_agent.run_muse(
                prompt=body.prompt,
                cwd=str(cwd),
                model=body.model or settings.muse_model,
                effort=body.effort,
                binary=settings.muse_bin,
            )

    try:
        async with asyncio.timeout(timeout):
            if not await _muse_available(request.app, lease=lease):
                raise HTTPException(  # ruff: ignore[raise-within-try]
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    "Muse is disabled or unavailable",
                )
            return await _await_muse(execute(), lease)
    except TimeoutError as exc:
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT, "agent run timed out"
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Muse execution failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "agent execution failed"
        ) from exc
    finally:
        if not lease.claimed:
            lease.release()


async def _muse_query_with_disconnect(
    body: MuseQueryRequest, request: Request
) -> dict[str, Any] | Response:
    task = cast("asyncio.Task[object]", asyncio.current_task())
    client_gone = False

    async def watch_disconnect() -> None:
        nonlocal client_gone
        while not await request.is_disconnected():  # ruff: ignore[async-busy-wait]
            await asyncio.sleep(DISCONNECT_POLL_INTERVAL)
        client_gone = True
        task.cancel()

    watcher = asyncio.create_task(watch_disconnect())
    try:
        return await _muse_query(body, request)
    except asyncio.CancelledError:
        if client_gone and task.uncancel() == 0:
            return Response(status_code=499)
        raise
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher


def _add_meta_routes(app: FastAPI, settings: Settings) -> None:
    """Register the health and info routes."""

    @app.get("/health")
    async def health() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        """Report API process readiness.

        Returns:
            Status for the running HTTP server.
        """
        return {"status": "ok"}

    @app.get("/v1/info", dependencies=[Auth])
    async def info() -> InfoResponse:  # pyright: ignore[reportUnusedFunction]
        """Describe server defaults.

        Returns:
            Server capabilities.
        """
        return InfoResponse(
            version=VERSION,
            stateless=True,
            default_model=settings.default_model,
            workspace_root=str(settings.workspace_root),
            default_permission_mode=settings.permission_mode,
            default_allowed_tools=list(settings.allowed_tools),
            max_concurrency=settings.max_concurrency,
            auth_required=bool(settings.api_key and settings.api_key.strip()),
            sandbox_allowed_domains=list(settings.sandbox_allowed_domains),
        )

    @app.get("/v1/agents", dependencies=[Auth])
    async def agents() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return {
            "claude": {
                "enabled": True,
                "healthy": True,
                "default_model": settings.default_model,
                "supports": {
                    "sandbox": True,
                    "network_access": False,
                    "rate_limits": False,
                    "streaming": False,
                },
            },
            "codex": {
                "enabled": settings.codex_enabled,
                "healthy": await _codex_available(app),
                "default_model": None,
                "supports": {
                    "sandbox": True,
                    "network_access": True,
                    "rate_limits": True,
                    "streaming": False,
                },
            },
            "muse": {
                "enabled": settings.muse_enabled,
                "healthy": await _muse_available(app),
                "default_model": settings.muse_model,
                "supports": {
                    "sandbox": True,
                    "network_access": True,
                    "rate_limits": False,
                    "streaming": False,
                },
            },
        }

    @app.get("/v1/agents/codex/rate-limits", dependencies=[Auth])
    async def rate_limits() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        try:
            async with (
                _codex_slot(app) as lease,
                asyncio.timeout(settings.max_timeout_seconds),
            ):
                if not await _codex_available(app, lease=lease):
                    raise HTTPException(  # ruff: ignore[raise-within-try]
                        status.HTTP_503_SERVICE_UNAVAILABLE,
                        "Codex is disabled or unavailable",
                    )
                return await _await_codex(
                    codex_agent.read_rate_limits(settings.codex_bin), lease
                )
        except TimeoutError as exc:
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT, "Codex rate limits timed out"
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Codex rate limits failed")
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, "Codex rate limits failed"
            ) from exc

    _ = (health, info, agents, rate_limits)


def _add_query_routes(app: FastAPI) -> None:  # ruff: ignore[complex-structure]
    """Register the agent execution routes."""

    @app.post(
        "/v1/query",
        dependencies=[Auth],
        response_model=Annotated[
            QueryResponse | CodexQueryResponse | MuseQueryResponse,
            Field(discriminator="agent"),
        ],
    )
    async def run_query(  # pyright: ignore[reportUnusedFunction]
        body: Annotated[
            QueryRequest | CodexQueryRequest | MuseQueryRequest,
            Field(discriminator="agent"),
        ],
        request: Request,
    ) -> QueryResponse | Response | dict[str, Any]:
        """Run one ephemeral agent query and return its result.

        Args:
            body: Query request.
            request: Incoming request.

        Returns:
            Final result and optional Claude message transcript.

        Raises:
            HTTPException: 502 on agent failure, 504 on timeout.
            asyncio.CancelledError: If cancellation is not solely due to disconnect.
        """
        if isinstance(body, CodexQueryRequest):
            return await _codex_query_with_disconnect(body, request)
        if isinstance(body, MuseQueryRequest):
            return await _muse_query_with_disconnect(body, request)
        options, timeout, semaphore = await _prepare(request, body)
        agent_runner: AgentRunner = request.app.state.runner
        task = cast("asyncio.Task[object]", asyncio.current_task())

        client_gone = False

        async def watch_disconnect() -> None:
            nonlocal client_gone
            while not await request.is_disconnected():  # ASGI disconnect polling  # ruff: ignore[async-busy-wait]
                await asyncio.sleep(DISCONNECT_POLL_INTERVAL)
            logger.info("client disconnected; cancelling agent run")
            client_gone = True
            task.cancel()

        watcher = asyncio.create_task(watch_disconnect())
        try:
            async with asyncio.timeout(timeout):
                cwd = resolve_cwd(request.app.state.settings, body.cwd)
                async with request.app.state.jobs.workspace(cwd):
                    stream = agent_runner.stream(body.prompt, options)
                    async with aclosing(
                        cast("AsyncGenerator[Message]", stream)
                    ) as messages:
                        return await collect_result(
                            messages, include_messages=body.include_messages
                        )
        except asyncio.CancelledError:
            if client_gone and task.uncancel() == 0:
                return Response(status_code=499)
            raise
        except TimeoutError as exc:
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT, "agent run timed out"
            ) from exc
        except AgentExecutionError as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
        except Exception as exc:
            logger.exception("agent execution failed")
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, "agent execution failed"
            ) from exc
        finally:
            watcher.cancel()
            try:
                with suppress(asyncio.CancelledError):
                    await watcher
            finally:
                semaphore.release()

    _ = run_query


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Open the durable queue and stop all owned jobs on shutdown."""
    app.state.jobs.start()
    try:
        yield
    finally:
        await muse_agent.shutdown()
        await app.state.jobs.stop()


def _add_job_routes(app: FastAPI) -> None:
    """Register durable polling and cancellation endpoints."""

    @app.post("/v1/jobs", dependencies=[Auth], status_code=status.HTTP_202_ACCEPTED)
    async def submit_job(  # pyright: ignore[reportUnusedFunction]
        body: Annotated[
            QueryRequest | CodexQueryRequest | MuseQueryRequest,
            Field(discriminator="agent"),
        ],
        request: Request,
        idempotency_key: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            request.app.state.jobs.submit(body, authorization, idempotency_key),
        )

    @app.get("/v1/jobs/{job_id}", dependencies=[Auth])
    async def get_job(  # pyright: ignore[reportUnusedFunction]
        job_id: str,
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        return cast("dict[str, Any]", request.app.state.jobs.get(job_id, authorization))

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Auth])
    async def cancel_job(  # pyright: ignore[reportUnusedFunction]
        job_id: str,
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        return cast(
            "dict[str, Any]", request.app.state.jobs.cancel(job_id, authorization)
        )

    _ = (submit_job, get_job, cancel_job)


def create_app(
    settings: Settings | None = None, runner: AgentRunner | None = None
) -> FastAPI:
    """Create the application.

    Args:
        settings: Server settings; read from the environment if omitted.
        runner: Agent runner; defaults to the real SDK runner.

    Returns:
        Configured FastAPI app.
    """
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="agsrv",
        description="Stateless HTTP API for remote Claude Code execution.",
        version=VERSION,
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.state.runner = runner or SdkAgentRunner()
    app.state.semaphore = asyncio.Semaphore(settings.max_concurrency)
    app.state.codex_semaphore = asyncio.Semaphore(settings.codex_max_concurrency)
    app.state.codex_probe_lock = asyncio.Lock()
    app.state.codex_probe_expires = 0.0
    app.state.codex_healthy = False
    app.state.muse_semaphore = asyncio.Semaphore(settings.muse_max_concurrency)
    app.state.muse_probe_lock = asyncio.Lock()
    app.state.muse_probe_expires = 0.0
    app.state.muse_healthy = False
    app.state.jobs = JobManager(app)
    _add_meta_routes(app, settings)
    _add_query_routes(app)
    _add_job_routes(app)
    return app


app = create_app()
