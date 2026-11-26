"""Muse SDK helpers for ephemeral, non-interactive runs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from muse_code import (
    MuseClient,
    MuseClientSpawnOptions,
    StartSessionOptions,
)
from muse_code.facade.approval import ApprovalDecisionInput

from agsrv.sandbox import SANDBOX_DENIED_ENV_VARS

if TYPE_CHECKING:
    from collections.abc import Callable

    from muse_code.facade.approval import ApprovalFailure
    from muse_code_msp import ApprovalRequestParams

logger = logging.getLogger(__name__)

APPROVAL_MODE = "onRequest"
SERVE_ARGS: tuple[str, ...] = ("serve", "--no-session-log", "--contributor-mode", "off")

DISABLE_SANDBOX_ARG = "--disable-sandbox"

_active_hosts: set[MuseClient[Any]] = set()


class MuseUnavailableError(RuntimeError):
    """The Muse host cannot provide the required availability guarantees."""


def serve_args(*, sandbox: bool = True) -> list[str]:
    """Return serve arguments for the Muse host.

    Args:
        sandbox: Keep the host sandbox enabled. Only an explicit ``False``
            adds ``--disable-sandbox``.

    Returns:
        Memory-only arguments with contributor mode off, plus
        ``--disable-sandbox`` when the sandbox is explicitly disabled.
    """
    args = list(SERVE_ARGS)
    if not sandbox:
        args.append(DISABLE_SANDBOX_ARG)
    return args


def deny_approval(request: ApprovalRequestParams) -> ApprovalDecisionInput:
    """Decline every approval request.

    Args:
        request: SDK approval request.

    Returns:
        An unambiguous denial or abort, otherwise an input the SDK rejects locally.
    """
    choices = request.get("availableChoices") or []
    # The SDK matches the first entry for an ID and sends only that ID.
    decisions_by_id: dict[str, set[str | None]] = {}
    for choice in choices:
        decisions_by_id.setdefault(choice["choiceId"], set()).add(
            choice.get("decision")
        )
    for choice_id, decisions in decisions_by_id.items():
        if len(decisions) == 1 and decisions & {"denied", "deniedPolicyAmendment"}:
            return ApprovalDecisionInput(choice_id=choice_id)
    logger.warning(
        "Muse approval has no deny choice (offered decisions: %s)",
        [choice.get("decision") for choice in choices],
    )
    for choice_id, decisions in decisions_by_id.items():
        if decisions == {"abort"}:
            return ApprovalDecisionInput(choice_id=choice_id)

    # The SDK requires a decision input. Extending the greatest offered ID
    # guarantees an unoffered ID, which the router rejects before sending.
    greatest_id = max(decisions_by_id, default="")
    return ApprovalDecisionInput(choice_id=greatest_id + "_")


def _on_stderr(chunk: str) -> None:
    logger.debug("muse host stderr: %s", chunk[:500])


def _client_info() -> dict[str, str]:
    from agsrv.main import VERSION  # ruff: ignore[import-outside-top-level]

    return {"name": "agsrv", "version": VERSION, "title": "agsrv"}


def _is_ephemeral(client: MuseClient[Any]) -> bool:
    durability = client.durability
    kind = getattr(durability, "kind", None)
    return kind == "ephemeral"


async def _spawn(
    binary: str | None,
    cwd: str | None,
    tmpdir: str | None = None,
    *,
    sandbox: bool = True,
    experimental: bool = False,
) -> MuseClient[Any]:
    env = {k: v for k, v in os.environ.items() if k not in SANDBOX_DENIED_ENV_VARS}
    if tmpdir:
        env["TMPDIR"] = tmpdir
    opts = MuseClientSpawnOptions(
        binary,
        _client_info(),  # pyright: ignore[reportArgumentType]
        args=serve_args(sandbox=sandbox),
        cwd=cwd,
        env=env,
        capabilities={"experimentalApi": True} if experimental else None,
        on_stderr=_on_stderr,
    )
    client: MuseClient[Any] = await MuseClient.spawn(opts)
    _active_hosts.add(client)
    return client


def _sandbox_available(client: MuseClient[Any]) -> bool:
    return client.initialize_result.get("sandboxAvailability") != "unavailable"


async def _close(client: MuseClient[Any]) -> None:
    async def cleanup() -> None:
        try:
            with contextlib.suppress(Exception):
                await client.close()
        finally:
            _active_hosts.discard(client)

    task = asyncio.create_task(cleanup())
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


async def probe(binary: str | None) -> bool:
    """Check ephemerality without running a turn.

    Returns:
        True when the host spawns and reports ephemeral durability.
    """
    client: MuseClient[Any] | None = None
    try:
        client = await _spawn(binary, None)
        return _is_ephemeral(client) and _sandbox_available(client)
    except Exception:  # ruff: ignore[blind-except]  # any failure means unhealthy
        return False
    finally:
        if client is not None:
            await _close(client)


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
    kind = value.kind
    if kind == "completed":
        return str(value.params["terminal"])
    if kind in {"unqueued", "terminalUnknown"}:
        return str(kind)
    return "unknown"


async def run_muse(
    *,
    prompt: str,
    cwd: str,
    model: str | None,
    effort: str | None,
    binary: str | None,
    tmpdir: str | None = None,
    sandbox: bool = True,
) -> dict[str, Any]:
    """Run a single ephemeral Muse turn.

    Args:
        prompt: User input.
        cwd: Resolved workspace directory.
        model: Optional requested model.
        effort: Optional reasoning effort.
        binary: Optional Muse binary path.
        tmpdir: Private per-turn directory exported to the host as ``TMPDIR``.
        sandbox: Keep the host sandbox on (default). ``False`` spawns this
            turn's host with ``--disable-sandbox`` and skips the sandbox
            availability check for this turn only.

    Returns:
        Normalized response dict.

    Raises:
        RuntimeError: If the backend is not ephemeral or the sandbox is unavailable.
    """
    started = time.monotonic()
    client: MuseClient[Any] | None = None
    try:
        client = await _spawn(binary, cwd, tmpdir, sandbox=sandbox)
        if not _is_ephemeral(client):
            msg = "Muse backend requires ephemeral sessions"
            raise RuntimeError(msg)
        if sandbox and not _sandbox_available(client):
            msg = "Muse sandbox unavailable"
            raise RuntimeError(msg)
        session = await client.start_session(
            StartSessionOptions(
                approval_mode=APPROVAL_MODE,
                workspace_root=cwd,
                model_id=model,
            )
        )
        session.on_approval(deny_approval)

        def _on_error(failure: ApprovalFailure) -> None:
            logger.warning("Muse approval could not be resolved: %r", failure)

        session.on_approval_error(_on_error)
        run_task = asyncio.create_task(
            session.run(prompt, reasoning_effort=effort)  # pyright: ignore[reportArgumentType]
        )
        try:
            # The query or job deadline also bounds turns left pending by the host.
            result = await asyncio.shield(run_task)
        finally:
            if not run_task.done():
                run_task.cancel()
                await asyncio.wait({run_task}, timeout=0.5)
            if run_task.done():
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    run_task.result()
            else:
                run_task.add_done_callback(_consume_task)
        duration_ms = int((time.monotonic() - started) * 1000)
        outcome = _outcome_str(result.outcome)
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
                "sandbox": sandbox,
            },
        }
    finally:
        if client is not None:
            await _close(client)


def _consume_task(task: asyncio.Task[Any]) -> None:
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.result()


def parse_usage(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the SDK subscription payload.

    Returns:
        Normalized windows and the original payload.
    """
    usage = raw.get("usage")
    windows: dict[str, Any] = {"five_hour": None, "seven_day": None}
    if isinstance(usage, dict):
        for source, target in (("window", "five_hour"), ("weekly", "seven_day")):
            item = cast("dict[str, Any]", usage).get(source)
            if isinstance(item, dict):
                item = cast("dict[str, Any]", item)
                percent, reset = item.get("usedPercent"), item.get("resetsAtMs")
                if type(percent) is int and percent >= 0 and type(reset) is int:
                    try:
                        stamp = datetime.fromtimestamp(reset / 1000, UTC).isoformat()
                    except (ValueError, OverflowError, OSError):
                        continue
                    windows[target] = {"used_percent": percent, "resets_at": stamp}
    return {
        "available": any(v is not None for v in windows.values()),
        **windows,
        "raw": raw,
        "fetched_at": datetime.now(UTC).isoformat(),
    }


async def read_usage(
    binary: str | None, on_dispatch: Callable[[], None]
) -> dict[str, Any]:
    """Refresh subscription usage without a model turn.

    Returns:
        The refreshed usage payload, or an empty dict if the refresh fails.

    Raises:
        MuseUnavailableError: If the host is unavailable.
    """
    try:
        client = await _spawn(binary, None, experimental=True)
    except Exception:  # ruff: ignore[blind-except]  # unavailable usage fails closed
        return {}
    try:
        if not _is_ephemeral(client) or not _sandbox_available(client):
            msg = "Muse unavailable"
            raise MuseUnavailableError(msg)
        try:
            on_dispatch()
            return dict(await client.request("usage/refresh"))
        except Exception:  # ruff: ignore[blind-except]  # unavailable usage fails closed
            return {}
    finally:
        await _close(client)


async def shutdown() -> None:
    """Close any active Muse hosts for app shutdown."""
    hosts = list(_active_hosts)
    for host in hosts:
        await _close(host)
