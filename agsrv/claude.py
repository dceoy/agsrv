"""Agent execution helpers around the stateless Claude Agent SDK ``query()``."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
from contextlib import aclosing, suppress
from typing import TYPE_CHECKING, Any, Protocol, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultError,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from agsrv.sandbox import (
    build_sandbox_settings,
    bundled_cli_path,
    prepare_sandbox_cache,
    sandbox_env,
    service_home,
)
from agsrv.schemas import QueryResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from claude_agent_sdk import Message

    from agsrv.config import Settings
    from agsrv.schemas import QueryRequest


class AgentExecutionError(Exception):
    """Raised when an agent run cannot produce a final result."""


class AgentRunner(Protocol):
    """Runs a single ephemeral agent query."""

    def stream(
        self, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        """Stream SDK messages for one prompt.

        Args:
            prompt: User prompt.
            options: SDK options.
        """
        ...


class SdkAgentRunner:
    """Runner backed by ``claude_agent_sdk.query``."""

    def stream(  # ruff: ignore[no-self-use]  # implements the AgentRunner protocol
        self, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        """Stream SDK messages for one prompt.

        Args:
            prompt: User prompt.
            options: SDK options.

        Returns:
            Async iterator of SDK messages.
        """
        return query(prompt=prompt, options=options)


def json_safe(value: object) -> Any:  # ruff: ignore[any-type]
    """Recursively convert a value into JSON-serializable data.

    Args:
        value: Arbitrary value.

    Returns:
        JSON-compatible data; unknown objects become ``str(value)``.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {
            str(k): json_safe(v) for k, v in cast("dict[object, object]", value).items()
        }
    if isinstance(value, list | tuple | set | frozenset):
        return [json_safe(v) for v in cast("list[object]", value)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    return str(value)


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _generic(obj: object) -> dict[str, Any]:
    data = json_safe(obj)
    kind = _snake(type(obj).__name__)
    return {"type": kind, "data": data}


def block_to_dict(block: object) -> dict[str, Any]:
    """Serialize a content block.

    Args:
        block: SDK content block.

    Returns:
        JSON-compatible dict with a ``type`` key.
    """
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": json_safe(block.input),
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": json_safe(block.content),
            "is_error": block.is_error,
        }
    return _generic(block)


def message_to_dict(message: object) -> dict[str, Any]:
    """Serialize an SDK message.

    Args:
        message: SDK message.

    Returns:
        JSON-compatible dict with a ``type`` key.
    """
    if isinstance(message, SystemMessage):
        return {
            "type": "system",
            "subtype": message.subtype,
            "data": json_safe(message.data),
        }
    if isinstance(message, AssistantMessage):
        return {
            "type": "assistant",
            "model": message.model,
            "content": [block_to_dict(b) for b in message.content],
            "error": message.error,
            "usage": json_safe(message.usage),
            "stop_reason": message.stop_reason,
            "session_id": message.session_id,
            "uuid": message.uuid,
        }
    if isinstance(message, UserMessage):
        content = (
            message.content
            if isinstance(message.content, str)
            else [block_to_dict(b) for b in message.content]
        )
        return {
            "type": "user",
            "content": content,
            "uuid": message.uuid,
            "parent_tool_use_id": message.parent_tool_use_id,
        }
    if isinstance(message, ResultMessage):
        return {
            "type": "result",
            "subtype": message.subtype,
            "is_error": message.is_error,
            "result": message.result,
            "session_id": message.session_id,
            "num_turns": message.num_turns,
            "duration_ms": message.duration_ms,
            "duration_api_ms": message.duration_api_ms,
            "total_cost_usd": message.total_cost_usd,
            "usage": json_safe(message.usage),
            "stop_reason": message.stop_reason,
            "errors": json_safe(message.errors),
        }
    return _generic(message)


def build_options(
    request: QueryRequest,
    settings: Settings,
    cwd: Path,
    home: Path | None = None,
    tmpdir: Path | None = None,
) -> ClaudeAgentOptions:
    """Build stateless SDK options from a request and server defaults.

    Args:
        request: Validated request.
        settings: Server settings.
        cwd: Resolved working directory.
        home: Service home directory for sandboxed runs; looked up if omitted.
        tmpdir: Private per-turn temporary directory.

    Returns:
        SDK options; never sets session/resume fields.
    """
    kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "permission_mode": request.permission_mode or settings.permission_mode,
        "allowed_tools": (
            request.allowed_tools
            if request.allowed_tools is not None
            else list(settings.allowed_tools)
        ),
        # Stateless and hermetic runs: the CLI must not persist a resumable
        # session transcript, and host/project settings (hooks, MCP servers,
        # permission rules) must not leak into runs.
        "setting_sources": [],
        "extra_args": {"no-session-persistence": None},
        "env": {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"},
        "strict_mcp_config": True,
        "system_prompt": {"type": "preset", "preset": "claude_code"},
    }
    if tmpdir is not None:
        kwargs["env"]["TMPDIR"] = str(tmpdir)
    model = request.model or settings.default_model
    optional: dict[str, Any] = {
        "model": model,
        "max_turns": request.max_turns,
        "effort": request.effort,
    }
    kwargs.update({k: v for k, v in optional.items() if v is not None})
    if request.fast_mode:
        kwargs["settings"] = json.dumps({"fastMode": True})
    if request.sandbox:
        home = home or service_home()
        kwargs["cli_path"] = bundled_cli_path()
        prepare_sandbox_cache(home)
        kwargs["sandbox"] = build_sandbox_settings(settings, home)
        kwargs["env"] = {**kwargs["env"], **sandbox_env(home, tmpdir)}
    return ClaudeAgentOptions(**kwargs)


async def collect_result(
    stream: AsyncIterator[Message], *, include_messages: bool = True
) -> QueryResponse:
    """Consume a message stream and build the final response.

    Args:
        stream: SDK message stream.
        include_messages: Whether to include the serialized transcript.

    Returns:
        Response built from the final ``ResultMessage``.

    Raises:
        AgentExecutionError: If the stream ends without a ``ResultMessage``.
        ResultError: If the SDK fails before producing a result.
    """
    messages: list[dict[str, Any]] = []
    final: ResultMessage | None = None
    model: str | None = None
    try:  # ruff: ignore[too-many-statements-in-try-clause]  # consume the complete stream
        async for message in stream:
            if isinstance(message, AssistantMessage):
                model = message.model
            if include_messages:
                messages.append(message_to_dict(message))
            if isinstance(message, ResultMessage):
                final = message
    except ResultError:
        if final is None:
            raise
    if final is None:
        msg = "agent stream ended without a result message"
        raise AgentExecutionError(msg)
    return QueryResponse(
        result=final.result,
        session_id=final.session_id,
        model=model,
        subtype=final.subtype,
        is_error=final.is_error,
        num_turns=final.num_turns,
        duration_ms=final.duration_ms,
        duration_api_ms=final.duration_api_ms,
        total_cost_usd=final.total_cost_usd,
        usage=final.usage,
        stop_reason=final.stop_reason,
        messages=messages,
    )


AUTH_STATUS_TIMEOUT = 5.0


async def _logged_in(env: dict[str, str], cwd: str | None) -> bool:
    """Ask the bundled CLI whether it is logged in.

    Runs ``<bundled cli> auth status`` with the query path's environment (minus
    ``CLAUDECODE``, which the SDK strips for turns) in the workspace directory.
    Any spawn, timeout, exit-code, or parse failure counts as not logged in.
    Non-JSON noise before the first ``{`` on stdout is ignored.

    Args:
        env: Environment overrides applied by the query path.
        cwd: Working directory for the subprocess (the workspace).

    Returns:
        True only if the JSON output has ``loggedIn`` exactly ``true``.
    """
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            str(bundled_cli_path()),
            "auth",
            "status",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={k: v for k, v in {**os.environ, **env}.items() if k != "CLAUDECODE"},
            cwd=cwd,
        )
        stdout, _ = await asyncio.wait_for(
            process.communicate(), timeout=AUTH_STATUS_TIMEOUT
        )
        if process.returncode != 0:
            return False
        text = stdout.decode()
        payload: Any = json.loads(text[text.index("{") :])
    except Exception:  # ruff: ignore[blind-except]
        return False
    finally:
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await asyncio.shield(process.wait())
    return (
        isinstance(payload, dict)
        and cast("dict[str, Any]", payload).get("loggedIn") is True
    )


async def probe(runner: AgentRunner, settings: Settings) -> bool:
    """Check that Claude is usable without spending tokens.

    Healthy means the CLI is spawnable, an ephemeral ``/usage`` run succeeds
    without ``is_error``, and ``auth status`` reports ``loggedIn: true``. This
    does not prove the credential is still valid server-side.

    The probe never uses the sandbox, so sandbox prerequisites cannot fail it
    or create cache directories; sandbox eligibility stays per request.

    Returns:
        Whether Claude is considered healthy.
    """
    from agsrv.schemas import QueryRequest  # ruff: ignore[import-outside-top-level]

    try:  # ruff: ignore[too-many-statements-in-try-clause]
        body = QueryRequest(
            agent="claude",
            prompt="/usage",
            allowed_tools=[],
            max_turns=1,
            sandbox=False,
        )
        options = build_options(body, settings, settings.workspace_root)
        options.tools = []
        stream = runner.stream(body.prompt, options)
        async with aclosing(cast("Any", stream)) as messages:
            result = await collect_result(messages, include_messages=False)
        if result.is_error:
            return False
        return await _logged_in(dict(options.env), str(options.cwd))
    except Exception:  # ruff: ignore[blind-except]
        return False
