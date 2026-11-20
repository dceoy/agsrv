"""Codex SDK helpers for ephemeral, non-interactive runs."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any, cast

from openai_codex import (
    ApprovalMode,
    AsyncCodex,
    AsyncThread,
    AsyncTurnHandle,
    CodexConfig,
    Sandbox,
)
from openai_codex._initialize_metadata import (  # ruff: ignore[import-private-name]
    validate_initialize_metadata,
)
from openai_codex.generated.v2_all import GetAccountRateLimitsResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from openai_codex.generated.v2_all import ReasoningEffort
    from openai_codex.models import JsonObject


def decline_approval(method: str, params: JsonObject | None) -> JsonObject:
    """Decline server-initiated approval requests.

    Returns:
        A denial decision for the SDK transport.
    """
    _ = (method, params)
    return {"decision": "decline"}


def make_client(binary: str | None) -> AsyncCodex:
    """Create a client with a fail-closed approval handler.

    Args:
        binary: Optional Codex binary path.

    Returns:
        An uninitialized SDK client.
    """
    client = AsyncCodex(CodexConfig(codex_bin=binary))
    client._client._sync._approval_handler = decline_approval  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
    return client


@asynccontextmanager
async def client_context(binary: str | None) -> AsyncGenerator[AsyncCodex, None]:
    """Start the transport separately so a stalled handshake can be closed.

    Yields:
        The initialized client.

    Raises:
        asyncio.CancelledError: If cancelled during cleanup.
    """
    client = make_client(binary)
    sync = client._client._sync  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
    startup_executor = ThreadPoolExecutor(max_workers=1)
    start = asyncio.get_running_loop().run_in_executor(startup_executor, sync.start)

    async def cleanup() -> None:
        with suppress(Exception):
            await start
        startup_executor.shutdown(wait=True)
        # Kill first: SDK close closes stdin before signaling, and a writer may
        # hold its lock while blocked on a full pipe.
        proc = sync._proc  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
        if proc is not None:
            with suppress(ProcessLookupError):
                proc.kill()
        # A stalled SDK RPC must not prevent its transport from closing.
        with ThreadPoolExecutor(max_workers=1) as executor:
            await asyncio.get_running_loop().run_in_executor(
                executor,
                sync.close,
            )

    try:
        await asyncio.wait({start})
        start.result()
        payload = await client._client.initialize()  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
        client._init = validate_initialize_metadata(payload)  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
        client._initialized = True  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
        yield client
    finally:
        closing = asyncio.create_task(cleanup())
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        closing.result()
        if cancelled:
            raise asyncio.CancelledError


def _window(window: dict[str, Any] | None) -> dict[str, Any] | None:
    if window is None:
        return None
    return {
        "used_percent": window.get("usedPercent"),
        "resets_at": window.get("resetsAt"),
    }


async def probe(binary: str | None) -> bool:
    """Initialize the configured Codex binary.

    Returns:
        Whether initialization succeeded.
    """
    try:
        async with client_context(binary):
            return True
    except Exception:  # ruff: ignore[blind-except]  # any initialization failure means unhealthy
        return False


async def read_rate_limits(binary: str | None) -> dict[str, Any]:
    """Read the current account's rate limits from the Codex app server.

    Args:
        binary: Optional Codex binary path.

    Returns:
        Normalized fields and the SDK payload.
    """
    async with client_context(binary) as client:
        response = await client._client.request(  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
            "account/rateLimits/read", None, response_model=GetAccountRateLimitsResponse
        )
    raw = response.model_dump(mode="json", by_alias=True)
    limits = raw["rateLimits"]
    return {
        "plan": limits.get("planType"),
        "primary": _window(limits.get("primary")),
        "secondary": _window(limits.get("secondary")),
        "reached_type": limits.get("rateLimitReachedType"),
        "reset_credits": raw.get("rateLimitResetCredits"),
        "backend": raw,
    }


async def run_codex(
    *,
    prompt: str,
    cwd: str,
    model: str | None,
    effort: str | None,
    sandbox: str,
    network_access: bool,
    binary: str | None,
) -> dict[str, Any]:
    """Run a single ephemeral Codex turn.

    Args:
        prompt: User input.
        cwd: Resolved workspace directory.
        model: Optional requested model.
        effort: Optional reasoning effort.
        sandbox: Sandbox mode.
        network_access: Whether sandbox network access is enabled.
        binary: Optional Codex binary path.

    Returns:
        The collected turn and thread metadata.
    """
    async with client_context(binary) as client:
        started = await client._client.thread_start({  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
            "cwd": cwd,
            "model": model,
            "ephemeral": True,
            "approvalPolicy": "never",
            "sandbox": sandbox,
        })
        thread = started.thread
        if network_access:
            policy: dict[str, Any] = {
                "type": "readOnly" if sandbox == "read-only" else "workspaceWrite",
                "networkAccess": True,
            }
            turn, subscription = await client._client._start_turn(  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
                thread.id,
                prompt,
                {
                    "threadId": thread.id,
                    "input": [{"type": "text", "text": prompt}],
                    "cwd": cwd,
                    "approvalPolicy": "never",
                    "sandboxPolicy": policy,
                    "effort": effort,
                },
                for_handle=True,
            )
            result = await AsyncTurnHandle(
                client, thread.id, turn.turn.id, _subscription=subscription
            ).run()
        else:
            result = await AsyncThread(client, thread.id).run(
                prompt,
                cwd=cwd,
                approval_mode=ApprovalMode.deny_all,
                sandbox=Sandbox(sandbox),
                effort=cast("ReasoningEffort | None", effort),
            )
        items = [item.model_dump(mode="json", by_alias=True) for item in result.items]
        commands = [item for item in items if item.get("type") == "commandExecution"]
        return {
            "agent": "codex",
            "result": result.final_response,
            "model": getattr(thread, "model", None),
            "is_error": result.error is not None,
            "duration_ms": result.duration_ms or 0,
            "usage": result.usage.model_dump(mode="json") if result.usage else None,
            "backend": {
                "id": result.id,
                "status": result.status.value,
                "commands": commands,
            },
        }
