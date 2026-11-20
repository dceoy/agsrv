"""Muse SDK helpers for ephemeral, non-interactive runs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

from muse_code import (
    MuseClient,
    MuseClientSpawnOptions,
    StartSessionOptions,
)
from muse_code.facade.approval import ApprovalDecisionInput

if TYPE_CHECKING:
    from muse_code_msp import ApprovalRequestParams

logger = logging.getLogger(__name__)

APPROVAL_MODE = "onRequest"
SERVE_ARGS: tuple[str, ...] = ("serve", "--no-session-log", "--contributor-mode", "off")

_active_hosts: set[MuseClient[Any]] = set()


def serve_args() -> list[str]:
    """Return fixed serve arguments for the Muse host.

    Returns:
        Memory-only arguments with contributor mode off.
    """
    return list(SERVE_ARGS)


def deny_approval(request: ApprovalRequestParams) -> ApprovalDecisionInput:
    """Decline every approval request.

    Args:
        request: SDK approval request.

    Returns:
        Decision input selecting the denied choice.

    Raises:
        RuntimeError: If no deny-like choice is offered.
    """
    choices = request.get("availableChoices") or []
    for choice in choices:
        decision = str(choice.get("decision", ""))
        if decision in {"denied", "deniedPolicyAmendment"}:
            return ApprovalDecisionInput(choice_id=str(choice["choiceId"]))
    for choice in choices:
        cid = str(choice.get("choiceId", ""))
        label = str(choice.get("label", ""))
        if "deny" in cid.lower() or "deny" in label.lower():
            return ApprovalDecisionInput(choice_id=str(choice["choiceId"]))
    for choice in choices:
        if "denied" in str(choice.get("choiceId", "")).lower():
            return ApprovalDecisionInput(choice_id=str(choice["choiceId"]))
    if not choices:
        msg = "Muse approval has no deny choice"
        raise RuntimeError(msg)
    for choice in choices:
        decision = str(choice.get("decision", ""))
        if decision not in {
            "approved",
            "approvedForSession",
            "approvedPolicyAmendment",
        }:
            return ApprovalDecisionInput(choice_id=str(choice["choiceId"]))
    msg = "Muse approval has no deny choice"
    raise RuntimeError(msg)


def _on_stderr(chunk: str) -> None:
    logger.debug("muse host stderr: %s", chunk[:500])


def _client_info() -> dict[str, str]:
    from agsrv.main import VERSION  # ruff: ignore[import-outside-top-level]

    return {"name": "agsrv", "version": VERSION, "title": "agsrv"}


def _is_ephemeral(client: MuseClient[Any]) -> bool:
    durability = client.durability
    kind = getattr(durability, "kind", None)
    return kind == "ephemeral"


async def _spawn(binary: str | None, cwd: str | None) -> MuseClient[Any]:
    opts = MuseClientSpawnOptions(
        binary,
        _client_info(),  # pyright: ignore[reportArgumentType]
        args=serve_args(),
        cwd=cwd,
        on_stderr=_on_stderr,
    )
    client: MuseClient[Any] = await MuseClient.spawn(opts)
    _active_hosts.add(client)
    return client


async def probe(binary: str | None) -> bool:
    """Check ephemerality without running a turn.

    Returns:
        True when the host spawns and reports ephemeral durability.
    """
    client: MuseClient[Any] | None = None
    try:
        client = await _spawn(binary, None)
        return _is_ephemeral(client)
    except Exception:  # ruff: ignore[blind-except]  # any failure means unhealthy
        return False
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()
            _active_hosts.discard(client)


def _usage_to_dict(usage: Any | None) -> dict[str, Any] | None:  # ruff: ignore[any-type]  # pyright: ignore[reportAny]
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage  # pyright: ignore[reportUnknownVariableType,reportReturnType]
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump(mode="json")  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
        return dict(dumped)  # pyright: ignore[reportUnknownVariableType,reportReturnType]
    result: dict[str, Any] = {  # pyright: ignore[reportUnknownVariableType]
        "input_tokens": getattr(
            usage, "inputTokens", getattr(usage, "input_tokens", None)
        ),
        "output_tokens": getattr(
            usage, "outputTokens", getattr(usage, "output_tokens", None)
        ),
        "cached_tokens": getattr(
            usage, "cachedTokens", getattr(usage, "cached_tokens", None)
        ),
        "reasoning_tokens": getattr(  # pyright: ignore[reportUnknownVariableType]
            usage,
            "reasoningTokens",
            getattr(usage, "reasoning_tokens", None),
        ),
    }
    return result


def _outcome_str(value: Any) -> str:  # ruff: ignore[any-type]
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    val = getattr(value, "value", None)
    if isinstance(val, str):
        return val.lower()
    return str(value).lower()


async def run_muse(
    *,
    prompt: str,
    cwd: str,
    model: str | None,
    effort: str | None,
    binary: str | None,
) -> dict[str, Any]:
    """Run a single ephemeral Muse turn.

    Args:
        prompt: User input.
        cwd: Resolved workspace directory.
        model: Optional requested model.
        effort: Optional reasoning effort.
        binary: Optional Muse binary path.

    Returns:
        Normalized response dict.

    Raises:
        RuntimeError: If the backend is not ephemeral or approval fails.
        asyncio.CancelledError: If cancelled.
    """
    started = time.monotonic()
    client: MuseClient[Any] | None = None
    run_task: asyncio.Task[Any] | None = None
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        client = await _spawn(binary, cwd)
        if not _is_ephemeral(client):
            msg = "Muse backend requires ephemeral sessions"
            raise RuntimeError(msg)
        session = await client.start_session(
            StartSessionOptions(
                approval_mode=APPROVAL_MODE,
                workspace_root=cwd,
                model_id=model,
            )
        )
        session.on_approval(deny_approval)

        def _on_error(failure: Any) -> None:  # ruff: ignore[any-type]
            msg = "Muse approval failed"
            raise RuntimeError(msg) from failure

        session.on_approval_error(_on_error)
        run_task = asyncio.create_task(
            session.run(prompt, reasoning_effort=effort)  # pyright: ignore[reportArgumentType]
        )
        try:
            result = await asyncio.shield(run_task)
        except asyncio.CancelledError:
            run_task.cancel()
            with _suppress_cancel():
                await run_task
            raise
        duration_ms = int((time.monotonic() - started) * 1000)
        outcome = _outcome_str(getattr(result, "outcome", "completed"))
        is_error = outcome != "completed"
        usage = _usage_to_dict(getattr(result, "usage", None))
        return {
            "agent": "muse",
            "result": getattr(result, "final_text", None),
            "model": model,
            "is_error": is_error,
            "duration_ms": duration_ms,
            "usage": usage,
            "backend": {
                "turn_id": getattr(result, "turn_id", None),
                "outcome": outcome,
            },
        }
    except asyncio.CancelledError:
        if run_task is not None and not run_task.done():
            run_task.cancel()
        raise
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()
            _active_hosts.discard(client)


@contextlib.contextmanager
def _suppress_cancel() -> Any:  # ruff: ignore[any-type]
    """Suppress CancelledError while draining a cancelled run.

    Yields:
        Suppresses CancelledError until the block exits.
    """
    with contextlib.suppress(asyncio.CancelledError):
        yield


async def shutdown() -> None:
    """Close any active Muse hosts for app shutdown."""
    hosts = list(_active_hosts)
    _active_hosts.clear()
    for host in hosts:
        with contextlib.suppress(Exception):
            await host.close()
