"""Tests for the stateless agent API."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, replace
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    Message,
    ResultError,
    ResultMessage,
    ServerToolUseBlock,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from agsrv.claude import (
    AgentExecutionError,
    AgentRunner,
    SdkAgentRunner,
    block_to_dict,
    build_options,
    collect_result,
    json_safe,
    message_to_dict,
)
from agsrv.config import DEFAULT_SANDBOX_ALLOWED_DOMAINS, Settings
from agsrv.main import _SlotLease, create_app  # pyright: ignore[reportPrivateUsage]
from agsrv.sandbox import (
    SandboxSetupError,
    build_sandbox_settings,
    bundled_cli_path,
    prepare_sandbox_cache,
    sandbox_cache_dir,
    sandbox_env,
    service_home,
)
from agsrv.schemas import QueryRequest
from agsrv.security import check_sandbox_cwd, check_sandbox_request

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Callable,
        Coroutine,
        Iterator,
        MutableMapping,
    )


def canned() -> list[Message]:
    """Build a canned message transcript.

    Returns:
        SDK messages.
    """
    return [
        SystemMessage(subtype="init", data={"a": 1}),
        AssistantMessage(
            content=[
                ThinkingBlock(thinking="hmm", signature="sig"),
                TextBlock(text="hello"),
                ToolUseBlock(id="t1", name="Bash", input={"command": "ls"}),
            ],
            model="m",
            session_id="s1",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="t1", content="out", is_error=False)]
        ),
        UserMessage(content="plain"),
        ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=5,
            is_error=False,
            num_turns=2,
            session_id="s1",
            total_cost_usd=0.01,
            result="done",
        ),
    ]


class FakeRunner:
    """Runner yielding canned messages, optionally failing or sleeping."""

    def __init__(
        self,
        messages: list[Message] | None = None,
        *,
        fail_after: int | None = None,
        delay: float = 0.0,
    ) -> None:
        """Initialize the fake.

        Args:
            messages: Messages to yield.
            fail_after: Raise after yielding this many messages.
            delay: Seconds to sleep before yielding.
        """
        self.messages = canned() if messages is None else messages
        self.fail_after = fail_after
        self.delay = delay
        self.options: ClaudeAgentOptions | None = None
        self.stream_calls = 0

    def stream(
        self, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        """Count the call itself, then return the message generator.

        Args:
            prompt: Forwarded.
            options: Forwarded.

        Returns:
            Async iterator of canned messages.
        """
        self.stream_calls += 1
        return self._generate(prompt, options)

    async def _generate(
        self, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        """Yield canned messages.

        Args:
            prompt: Ignored.
            options: Captured for assertions.

        Yields:
            Canned messages.

        Raises:
            RuntimeError: When configured to fail.
        """
        assert prompt
        self.options = options
        await asyncio.sleep(self.delay)
        for i, message in enumerate(self.messages):
            if self.fail_after is not None and i >= self.fail_after:
                msg = "boom"
                raise RuntimeError(msg)
            yield message


def make_settings(tmp_path: Path, **kwargs: Any) -> Settings:  # ruff: ignore[any-type]
    """Build test settings rooted in tmp_path.

    Args:
        tmp_path: Workspace root.
        **kwargs: Overrides.

    Returns:
        Settings.
    """
    base = Settings(
        api_key="secret", workspace_root=tmp_path.resolve(), max_concurrency=2
    )
    return replace(base, **kwargs)


def client_for(
    settings: Settings, runner: FakeRunner | None = None
) -> tuple[TestClient, FakeRunner]:
    """Create a client with a fake runner.

    Args:
        settings: Settings.
        runner: Optional runner.

    Returns:
        Client and runner.
    """
    runner = runner or FakeRunner()
    return TestClient(create_app(settings, runner), base_url="http://localhost"), runner


AUTH = {"Authorization": "Bearer secret"}


def test_health(tmp_path: Path) -> None:
    """Verify the public health endpoint."""
    client, _ = client_for(make_settings(tmp_path))
    with client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unauthenticated_hosts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Local development accepts loopback hosts only and warns at startup."""
    settings = make_settings(tmp_path, api_key=None, allow_unauthenticated=True)
    with TestClient(create_app(settings)) as client:
        assert "AGSRV_ALLOW_UNAUTHENTICATED is enabled" in caplog.text
        for host in ("localhost", "127.0.0.1:8000", "[::1]:8000"):
            assert client.get("/health", headers={"Host": host}).status_code == 200
        for host in ("example.org", "testserver"):
            assert client.get("/health", headers={"Host": host}).status_code == 400

    with TestClient(create_app(make_settings(tmp_path))) as client:
        assert client.get("/health", headers={"Host": "example.org"}).status_code == 200


def test_query_success(tmp_path: Path) -> None:
    """Verify a successful query and transcript serialization."""
    client, runner = client_for(make_settings(tmp_path))
    response = client.post(
        "/v1/query",
        json={"agent": "claude", "sandbox": False, "prompt": "hi"},
        headers=AUTH,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["result"] == "done"
    assert body["session_id"] == "s1"
    assert body["model"] == "m"
    assert body["total_cost_usd"] == pytest.approx(0.01)
    assert body["is_error"] is False
    types = [m["type"] for m in body["messages"]]
    assert types == ["system", "assistant", "user", "user", "result"]
    blocks = body["messages"][1]["content"]
    assert [b["type"] for b in blocks] == ["thinking", "text", "tool_use"]
    assert blocks[2]["input"] == {"command": "ls"}
    assert body["messages"][2]["content"][0]["type"] == "tool_result"
    assert body["messages"][3]["content"] == "plain"
    assert runner.options is not None
    assert runner.options.cwd == str(tmp_path.resolve())
    assert runner.options.permission_mode == "acceptEdits"


def test_query_without_messages(tmp_path: Path) -> None:
    """Result-only queries omit the transcript."""
    client, _ = client_for(make_settings(tmp_path))
    response = client.post(
        "/v1/query",
        json={
            "agent": "claude",
            "sandbox": False,
            "prompt": "hi",
            "include_messages": False,
        },
        headers=AUTH,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["result"] == "done"
    assert body["session_id"] == "s1"
    assert body["model"] == "m"
    assert body["messages"] == []


def test_query_agent_error_is_200(tmp_path: Path) -> None:
    """Verify agent-level errors stay HTTP 200."""
    result = ResultMessage(
        subtype="error_max_turns",
        duration_ms=1,
        duration_api_ms=1,
        is_error=True,
        num_turns=1,
        session_id="s",
    )
    client, _ = client_for(make_settings(tmp_path), FakeRunner([result]))
    response = client.post(
        "/v1/query",
        json={"agent": "claude", "sandbox": False, "prompt": "hi"},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert response.json()["is_error"] is True


def test_auth_matrix(tmp_path: Path) -> None:
    """Verify the authentication matrix."""
    disabled = make_settings(tmp_path, api_key=None)
    client, _ = client_for(disabled)
    assert (
        client.post(
            "/v1/query", json={"agent": "claude", "sandbox": False, "prompt": "x"}
        ).status_code
        == 503
    )
    assert client.get("/v1/info").status_code == 503

    for blank in ("", "   "):
        settings = replace(make_settings(tmp_path), api_key=blank)
        client, _ = client_for(settings)
        assert (
            client.post(
                "/v1/query", json={"agent": "claude", "sandbox": False, "prompt": "x"}
            ).status_code
            == 503
        )

    open_settings = make_settings(tmp_path, api_key=None, allow_unauthenticated=True)
    client, _ = client_for(open_settings)
    assert (
        client.post(
            "/v1/query", json={"agent": "claude", "sandbox": False, "prompt": "x"}
        ).status_code
        == 200
    )

    client, _ = client_for(make_settings(tmp_path))
    assert (
        client.post(
            "/v1/query", json={"agent": "claude", "sandbox": False, "prompt": "x"}
        ).status_code
        == 401
    )
    wrong = {"Authorization": "Bearer nope"}
    assert (
        client.post(
            "/v1/query",
            json={"agent": "claude", "sandbox": False, "prompt": "x"},
            headers=wrong,
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/v1/query",
            json={"agent": "claude", "sandbox": False, "prompt": "x"},
            headers=AUTH,
        ).status_code
        == 200
    )


def test_cwd(tmp_path: Path) -> None:
    """Verify working-directory confinement."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "file.txt").write_text("x")
    client, runner = client_for(make_settings(tmp_path))

    def post(cwd: str) -> int:
        return client.post(
            "/v1/query",
            json={"agent": "claude", "sandbox": False, "prompt": "x", "cwd": cwd},
            headers=AUTH,
        ).status_code

    assert post("/") == 403
    assert post("../outside") == 403
    assert post("missing") == 400
    assert post("file.txt") == 400
    assert post("sub") == 200
    assert runner.options is not None
    assert runner.options.cwd == str((tmp_path / "sub").resolve())
    assert post(str(tmp_path / "sub")) == 200


def test_cwd_symlinks_and_malformed(tmp_path: Path) -> None:
    """Verify symlink confinement and malformed path handling."""
    (tmp_path / "inner").mkdir()
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    (tmp_path / "alias").symlink_to(tmp_path / "inner", target_is_directory=True)
    (tmp_path / "loop-a").symlink_to(tmp_path / "loop-b")
    (tmp_path / "loop-b").symlink_to(tmp_path / "loop-a")
    client, _ = client_for(make_settings(tmp_path))

    def post(cwd: str) -> int:
        return client.post(
            "/v1/query",
            json={"agent": "claude", "sandbox": False, "prompt": "x", "cwd": cwd},
            headers=AUTH,
        ).status_code

    assert post("escape") == 403
    assert post("alias") == 200
    assert post("loop-a") == 400
    assert post("bad\0path") == 400


def test_cwd_containment_precedes_directory_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject outside paths before consulting the filesystem for directory status."""
    client, _ = client_for(make_settings(tmp_path))

    def fail_is_dir(_self: Path) -> bool:
        raise OSError

    monkeypatch.setattr(Path, "is_dir", fail_is_dir)
    response = client.post(
        "/v1/query",
        json={"agent": "claude", "sandbox": False, "prompt": "x", "cwd": "../outside"},
        headers=AUTH,
    )
    assert response.status_code == 403
    assert response.json() == {"detail": "cwd is outside the workspace root"}
    response = client.post(
        "/v1/query",
        json={"agent": "claude", "sandbox": False, "prompt": "x", "cwd": "."},
        headers=AUTH,
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "cwd is not a valid path"}


def test_bypass_permissions(tmp_path: Path) -> None:
    """Verify bypassPermissions is always rejected."""
    payload = {
        "agent": "claude",
        "sandbox": False,
        "prompt": "x",
        "permission_mode": "bypassPermissions",
    }
    client, _ = client_for(make_settings(tmp_path))
    assert client.post("/v1/query", json=payload, headers=AUTH).status_code == 403


def test_validation(tmp_path: Path) -> None:
    """Verify request validation."""
    client, _ = client_for(make_settings(tmp_path, max_timeout_seconds=10))
    assert (
        client.post(
            "/v1/query",
            json={"agent": "claude", "sandbox": False, "prompt": ""},
            headers=AUTH,
        ).status_code
        == 422
    )
    stateful = {"agent": "claude", "sandbox": False, "prompt": "x", "resume": "abc"}
    assert client.post("/v1/query", json=stateful, headers=AUTH).status_code == 422
    bad_sandbox = {"agent": "claude", "prompt": "x", "sandbox": "maybe"}
    assert client.post("/v1/query", json=bad_sandbox, headers=AUTH).status_code == 422
    for key in ("system_prompt", "disallowed_tools", "max_budget_usd"):
        assert (
            client.post(
                "/v1/query",
                json={
                    "agent": "claude",
                    "sandbox": False,
                    "prompt": "x",
                    key: "unused",
                },
                headers=AUTH,
            ).status_code
            == 422
        )
    assert (
        client.post(
            "/v1/query/stream", json={"agent": "claude", "prompt": "x"}, headers=AUTH
        ).status_code
        == 404
    )
    too_long = {
        "agent": "claude",
        "sandbox": False,
        "prompt": "x",
        "timeout_seconds": 11,
    }
    assert client.post("/v1/query", json=too_long, headers=AUTH).status_code == 400


def test_runner_failures(tmp_path: Path) -> None:
    """Verify 502 and 504 mappings."""
    settings = make_settings(tmp_path)
    client, _ = client_for(settings, FakeRunner(fail_after=1))
    response = client.post(
        "/v1/query",
        json={"agent": "claude", "sandbox": False, "prompt": "x"},
        headers=AUTH,
    )
    assert response.status_code == 502
    assert response.json() == {"detail": "agent execution failed"}

    no_result = FakeRunner(canned()[:2])
    client, _ = client_for(settings, no_result)
    response = client.post(
        "/v1/query",
        json={"agent": "claude", "sandbox": False, "prompt": "x"},
        headers=AUTH,
    )
    assert response.status_code == 502
    assert response.json() == {"detail": "agent stream ended without a result message"}

    client, _ = client_for(settings, FakeRunner(delay=1.0))
    slow = {"agent": "claude", "sandbox": False, "prompt": "x", "timeout_seconds": 0.05}
    assert client.post("/v1/query", json=slow, headers=AUTH).status_code == 504
    assert not list(tmp_path.glob(".agsrv-tmp-*"))


def test_concurrency_limit(tmp_path: Path) -> None:
    """Verify 429 when no capacity is available."""
    client, _ = client_for(make_settings(tmp_path, max_concurrency=0))
    assert (
        client.post(
            "/v1/query",
            json={"agent": "claude", "sandbox": False, "prompt": "x"},
            headers=AUTH,
        ).status_code
        == 429
    )


def test_semaphore_released(tmp_path: Path) -> None:
    """Verify slots are released after success and failure."""
    client, _ = client_for(make_settings(tmp_path, max_concurrency=1))
    for _ in range(2):
        assert (
            client.post(
                "/v1/query",
                json={"agent": "claude", "sandbox": False, "prompt": "x"},
                headers=AUTH,
            ).status_code
            == 200
        )
    client, _ = client_for(
        make_settings(tmp_path, max_concurrency=1), FakeRunner(fail_after=0)
    )
    for _ in range(2):
        assert (
            client.post(
                "/v1/query",
                json={"agent": "claude", "sandbox": False, "prompt": "x"},
                headers=AUTH,
            ).status_code
            == 502
        )


class BlockingRunner:
    """Runner that blocks mid-stream until released."""

    def __init__(self) -> None:
        """Initialize the events and capture slot."""
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.options: ClaudeAgentOptions | None = None

    async def stream(
        self, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        """Yield one message, block, then yield the result.

        Args:
            prompt: Ignored beyond a non-empty check.
            options: Captured for completeness.

        Yields:
            The first and last canned messages.
        """
        assert prompt
        self.options = options
        messages = canned()
        yield messages[0]
        self.started.set()
        await self.release.wait()
        yield messages[-1]


@pytest.mark.asyncio
async def test_concurrency_in_flight(tmp_path: Path) -> None:
    """Verify a second request gets 429 while a run holds the only slot."""
    runner = BlockingRunner()
    test_app = create_app(make_settings(tmp_path, max_concurrency=1), runner)
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(
            client.post(
                "/v1/query",
                json={"agent": "claude", "sandbox": False, "prompt": "x"},
                headers=AUTH,
            )
        )
        await runner.started.wait()
        second = await client.post(
            "/v1/query",
            json={"agent": "claude", "sandbox": False, "prompt": "x"},
            headers=AUTH,
        )
        assert second.status_code == 429
        runner.release.set()
        assert (await first).status_code == 200


def test_info(tmp_path: Path) -> None:
    """Verify the info endpoint."""
    client, _ = client_for(make_settings(tmp_path, default_model="m1"))
    body = client.get("/v1/info", headers=AUTH).json()
    assert body == {
        "version": _package_version("agsrv"),
        "stateless": True,
        "default_model": "m1",
        "workspace_root": str(tmp_path.resolve()),
        "default_permission_mode": "acceptEdits",
        "default_allowed_tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        "max_concurrency": 2,
        "auth_required": True,
        "sandbox_allowed_domains": list(DEFAULT_SANDBOX_ALLOWED_DOMAINS),
    }


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default settings."""
    for name in (
        "AGSRV_API_KEY",
        "AGSRV_ALLOW_UNAUTHENTICATED",
        "AGSRV_ALLOW_UNSANDBOXED",
        "AGSRV_WORKSPACE_ROOT",
        "AGSRV_DEFAULT_MODEL",
        "AGSRV_PERMISSION_MODE",
        "AGSRV_ALLOWED_TOOLS",
        "AGSRV_MAX_CONCURRENCY",
        "AGSRV_MAX_TIMEOUT_SECONDS",
        "AGSRV_SANDBOX_ALLOWED_DOMAINS",
        "AGSRV_CODEX_BIN",
        "AGSRV_CODEX_ENABLED",
        "AGSRV_CODEX_MAX_CONCURRENCY",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.api_key is None
    assert settings.allow_unauthenticated is False
    assert settings.allow_unsandboxed is True
    assert settings.workspace_root == Path.cwd().resolve()
    assert settings.default_model is None
    assert settings.permission_mode == "acceptEdits"
    assert settings.allowed_tools == ("Read", "Write", "Edit", "Bash", "Glob", "Grep")
    assert settings.max_concurrency == 4
    assert settings.max_timeout_seconds == pytest.approx(3600.0)
    assert settings.sandbox_allowed_domains == DEFAULT_SANDBOX_ALLOWED_DOMAINS
    assert settings.codex_bin is None
    assert settings.codex_enabled is True
    assert settings.codex_max_concurrency == 2
    assert Settings().workspace_root == Path.cwd().resolve()


def test_settings_positional_workspace_root(tmp_path: Path) -> None:
    """Preserve the positional Settings constructor's workspace argument."""
    settings = Settings("token", False, tmp_path)  # ruff: ignore[boolean-positional-value-in-call]
    assert settings.api_key == "token"
    assert settings.allow_unauthenticated is False
    assert settings.workspace_root == tmp_path
    assert settings.allow_unsandboxed is True


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify explicit environment parsing."""
    monkeypatch.setenv("AGSRV_API_KEY", "k")
    monkeypatch.setenv("AGSRV_ALLOW_UNAUTHENTICATED", "TRUE")
    monkeypatch.setenv("AGSRV_ALLOW_UNSANDBOXED", "false")
    monkeypatch.setenv("AGSRV_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGSRV_DEFAULT_MODEL", "mm")
    monkeypatch.setenv("AGSRV_PERMISSION_MODE", "plan")
    monkeypatch.setenv("AGSRV_ALLOWED_TOOLS", " Read , Bash,,")
    monkeypatch.setenv("AGSRV_MAX_CONCURRENCY", "7")
    monkeypatch.setenv("AGSRV_MAX_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("AGSRV_SANDBOX_ALLOWED_DOMAINS", " GitHub.com , *.example.org,,")
    monkeypatch.setenv("AGSRV_CODEX_BIN", "/usr/bin/codex")
    monkeypatch.setenv("AGSRV_CODEX_ENABLED", "false")
    monkeypatch.setenv("AGSRV_CODEX_MAX_CONCURRENCY", "3")
    settings = Settings.from_env()
    assert settings.api_key == "k"
    assert settings.allow_unauthenticated is True
    assert settings.allow_unsandboxed is False
    assert settings.workspace_root == tmp_path.resolve()
    assert settings.default_model == "mm"
    assert settings.permission_mode == "plan"
    assert settings.allowed_tools == ("Read", "Bash")
    assert settings.max_concurrency == 7
    assert settings.max_timeout_seconds == pytest.approx(12.5)
    assert settings.sandbox_allowed_domains == ("github.com", "*.example.org")
    assert settings.codex_bin == "/usr/bin/codex"
    assert settings.codex_enabled is False
    assert settings.codex_max_concurrency == 3


@pytest.mark.parametrize("value", ["", " , ", " ,, "])
def test_settings_sandbox_domains_empty(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Verify an empty domain list means no egress."""
    monkeypatch.setenv("AGSRV_SANDBOX_ALLOWED_DOMAINS", value)
    assert Settings.from_env().sandbox_allowed_domains == ()


def test_settings_sandbox_domains_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify lowercasing, whitespace trimming, empty entries, and punycode."""
    monkeypatch.setenv(
        "AGSRV_SANDBOX_ALLOWED_DOMAINS",
        " GitHub.com ,, *.Example.org,xn--bcher-kva.example,",
    )
    assert Settings.from_env().sandbox_allowed_domains == (
        "github.com",
        "*.example.org",
        "xn--bcher-kva.example",
    )
    label63 = "a" * 63
    host253 = ".".join(["a" * 63, "a" * 63, "a" * 63, "a" * 61])
    assert len(host253) == 253
    monkeypatch.setenv("AGSRV_SANDBOX_ALLOWED_DOMAINS", f"{label63}.com,{host253}")
    assert Settings.from_env().sandbox_allowed_domains == (f"{label63}.com", host253)


@pytest.mark.parametrize(
    "value",
    [
        "https://github.com",
        "github.com/x",
        "localhost",
        "*.com.*",  # embedded wildcard
        "a.*.com",  # embedded wildcard
        "*.*.example.com",  # repeated wildcard
        "*example.com",  # wildcard not a full label
        "a..com",  # empty DNS label
        ".a.com",  # leading empty label
        "a.com.",  # trailing dot
        "*.",  # wildcard only
        "-a.com",
        "a-.com",
        "bücher.example",  # raw Unicode (needs punycode)
        "\u212a.example",  # Kelvin sign U+212A would casefold to ASCII "k"
        "\u017f.example",  # long s U+017F would casefold to ASCII "s"
        "\u0130.example",  # dotted capital I
        "a" * 64 + ".com",  # label too long
        ".".join(["a" * 63, "a" * 63, "a" * 63, "a" * 62]),  # 254 chars
    ],
)
def test_settings_sandbox_domains_invalid(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Verify malformed domains fail at startup."""
    monkeypatch.setenv("AGSRV_SANDBOX_ALLOWED_DOMAINS", value)
    with pytest.raises(ValueError, match="invalid domain"):
        Settings.from_env()


def test_settings_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify an empty or blank AGSRV_API_KEY counts as unconfigured."""
    monkeypatch.setenv("AGSRV_API_KEY", "")
    assert Settings.from_env().api_key is None
    monkeypatch.setenv("AGSRV_API_KEY", "   ")
    assert Settings.from_env().api_key is None
    monkeypatch.setenv("AGSRV_API_KEY", "  secret  ")
    assert Settings.from_env().api_key == "secret"
    monkeypatch.delenv("AGSRV_API_KEY")
    assert Settings.from_env().api_key is None


def test_build_options(tmp_path: Path) -> None:
    """Verify defaults versus overrides."""
    settings = make_settings(tmp_path, default_model="dm")
    default_request = QueryRequest(agent="claude", sandbox=False, prompt="x")
    options = build_options(default_request, settings, tmp_path)
    assert options.model == "dm"
    assert options.permission_mode == "acceptEdits"
    assert options.allowed_tools == ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
    assert options.cwd == str(tmp_path)
    assert options.system_prompt == {"type": "preset", "preset": "claude_code"}
    assert options.setting_sources == []
    assert options.extra_args == {"no-session-persistence": None}
    assert options.env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert options.strict_mcp_config is True
    assert options.max_turns is None
    assert options.resume is None
    assert options.continue_conversation is False
    assert options.settings is None
    assert options == build_options(
        default_request.model_copy(update={"fast_mode": False}), settings, tmp_path
    )

    request = QueryRequest(
        agent="claude",
        sandbox=False,
        prompt="x",
        model="o",
        allowed_tools=["Read"],
        permission_mode="plan",
        max_turns=3,
        effort="low",
    )
    options = build_options(request, settings, tmp_path)
    assert options.model == "o"
    assert options.system_prompt == {"type": "preset", "preset": "claude_code"}
    assert options.allowed_tools == ["Read"]
    assert options.permission_mode == "plan"
    assert options.max_turns == 3
    assert options.effort == "low"


@pytest.mark.parametrize("fast_mode", [True, False, None])
def test_fast_mode_request_round_trip(fast_mode: bool | None) -> None:
    """Preserve fast mode through Claude request validation and serialization."""
    body = {"agent": "claude", "prompt": "x", "fast_mode": fast_mode}
    request = QueryRequest.model_validate(body)
    assert request.fast_mode is fast_mode
    assert QueryRequest.model_validate_json(request.model_dump_json()) == request


@dataclass
class Inner:
    """Nested dataclass."""

    values: set[int]


class Weird:
    """Object with a custom string form."""

    def __str__(self) -> str:
        """Return a fixed string.

        Returns:
            The string.
        """
        return "weird"


def test_json_safe() -> None:
    """Verify recursive conversion and fallback."""
    value = {"a": (1, 2), "b": Inner({3}), "c": Weird(), 4: None, "d": [1.5, True]}
    assert json_safe(value) == {
        "a": [1, 2],
        "b": {"values": [3]},
        "c": "weird",
        "4": None,
        "d": [1.5, True],
    }
    assert json_safe(Inner) == str(Inner)


def test_generic_serialization() -> None:
    """Verify fallback serialization for unknown message and block types."""
    event = StreamEvent(
        uuid="u",
        session_id="s",
        event={"x": 1},
        user_message_uuid="prompt-1",
        user_message_uuids=["prompt-1"],
    )
    assert message_to_dict(event) == {
        "type": "stream_event",
        "data": {
            "uuid": "u",
            "session_id": "s",
            "event": {"x": 1},
            "parent_tool_use_id": None,
            "user_message_uuid": "prompt-1",
            "user_message_uuids": ["prompt-1"],
        },
    }
    block = ServerToolUseBlock(id="i", name="web_search", input={"q": "z"})
    assert block_to_dict(block)["type"] == "server_tool_use_block"


def test_protocol_stub_is_noop() -> None:
    """Verify the AgentRunner protocol body is only a placeholder."""
    runner = cast("AgentRunner", object())
    result = AgentRunner.stream(runner, "p", ClaudeAgentOptions())  # pyright: ignore[reportAbstractUsage]
    assert result is None  # pyright: ignore[reportUnnecessaryComparison]


@pytest.mark.asyncio
async def test_collect_result_missing() -> None:
    """Verify missing ResultMessage raises."""
    with pytest.raises(AgentExecutionError):
        await collect_result(FakeRunner([]).stream("x", ClaudeAgentOptions()))


@pytest.mark.asyncio
async def test_collect_result_without_assistant() -> None:
    """A result without an assistant message has no model."""
    response = await collect_result(
        FakeRunner([canned()[-1]]).stream("x", ClaudeAgentOptions()),
        include_messages=False,
    )
    assert response.model is None
    assert response.messages == []


@pytest.mark.asyncio
async def test_collect_result_last_assistant_model() -> None:
    """The last assistant message determines the response model."""
    messages = [AssistantMessage(content=[], model="first"), *canned()]
    response = await collect_result(
        FakeRunner(messages).stream("x", ClaudeAgentOptions())
    )
    assert response.model == "m"


def test_sdk_runner_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the SDK runner calls claude_agent_sdk.query."""
    calls: list[tuple[str, ClaudeAgentOptions]] = []

    def fake_query(*, prompt: str, options: ClaudeAgentOptions) -> str:
        calls.append((prompt, options))
        return "stream"

    monkeypatch.setattr("agsrv.claude.query", fake_query)
    options = ClaudeAgentOptions()
    assert SdkAgentRunner().stream("p", options) == "stream"  # pyright: ignore[reportUnnecessaryComparison]
    assert calls == [("p", options)]


class ResultErrorRunner(FakeRunner):
    """Runner that raises the SDK terminal error after its messages."""

    async def stream(
        self, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        """Yield configured messages and then fail as the SDK does."""
        async for message in super().stream(prompt, options):
            yield message
        msg = "terminal result error"
        raise ResultError(msg)


def test_result_error_responses(tmp_path: Path) -> None:
    """Preserve a terminal error result, but reject an absent result."""
    result = replace(
        cast("ResultMessage", canned()[-1]), is_error=True, subtype="error_max_turns"
    )
    messages = [canned()[0], result]
    client, _ = client_for(make_settings(tmp_path), ResultErrorRunner(messages))
    response = client.post(
        "/v1/query",
        json={"agent": "claude", "sandbox": False, "prompt": "x"},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert response.json()["is_error"] is True
    assert [m["type"] for m in response.json()["messages"]] == ["system", "result"]
    client, _ = client_for(make_settings(tmp_path), ResultErrorRunner([]))
    assert (
        client.post(
            "/v1/query",
            json={"agent": "claude", "sandbox": False, "prompt": "x"},
            headers=AUTH,
        ).status_code
        == 502
    )


class CleanupRunner(BlockingRunner):
    """Record stream cleanup after cancellation or early closure."""

    def __init__(self) -> None:
        """Initialize cleanup tracking."""
        super().__init__()
        self.closed = asyncio.Event()

    async def stream(
        self, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        """Yield then wait until cancelled, recording cleanup."""
        try:
            async for message in super().stream(prompt, options):
                yield message
        finally:
            self.closed.set()


@pytest.mark.asyncio
async def test_slot_lease_release_is_idempotent() -> None:
    """Repeated release cannot inflate concurrency capacity."""
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    lease = _SlotLease(semaphore)
    lease.release()
    lease.release()
    await semaphore.acquire()
    assert semaphore.locked()
    semaphore.release()


def test_info_blank_key(tmp_path: Path) -> None:
    """A directly constructed blank key does not require authentication."""
    settings = Settings(
        api_key="   ", allow_unauthenticated=True, workspace_root=tmp_path
    )
    client, _ = client_for(settings)
    response = client.get("/v1/info")
    assert response.status_code == 200
    assert response.json()["auth_required"] is False


@pytest.mark.asyncio
async def test_query_disconnect_cancels_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-run disconnect closes the runner and allows another query."""
    monkeypatch.setattr("agsrv.main.DISCONNECT_POLL_INTERVAL", 0.01)
    runner = CleanupRunner()
    test_app = create_app(make_settings(tmp_path, max_concurrency=1), runner)
    disconnected = False

    async def receive() -> dict[str, Any]:
        if disconnected:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    route = next(
        r for r in test_app.routes if isinstance(r, APIRoute) and r.path == "/v1/query"
    )
    endpoint = cast("Callable[..., Coroutine[Any, Any, Any]]", route.endpoint)
    request = Request({"type": "http", "app": test_app}, receive)
    first = asyncio.create_task(
        endpoint(QueryRequest(agent="claude", sandbox=False, prompt="x"), request)
    )
    await asyncio.wait_for(runner.started.wait(), 1)
    disconnected = True
    result = await asyncio.wait_for(first, 1)
    assert result.status_code == 499
    assert runner.closed.is_set()
    assert not await asyncio.to_thread(lambda: list(tmp_path.glob(".agsrv-tmp-*")))
    runner.release.set()
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/query",
            json={"agent": "claude", "sandbox": False, "prompt": "x"},
            headers=AUTH,
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_query_asgi_disconnect_returns_499(tmp_path: Path) -> None:
    """A disconnect completes the whole ASGI stack after runner cleanup."""
    runner = CleanupRunner()
    test_app = create_app(make_settings(tmp_path, max_concurrency=1), runner)
    request_sent = False
    sent: list[MutableMapping[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": b'{"agent": "claude", "sandbox": false, "prompt":"x"}',
                "more_body": False,
            }
        await runner.started.wait()
        return {"type": "http.disconnect"}

    async def send(message: MutableMapping[str, Any]) -> None:  # ruff: ignore[unused-async]
        sent.append(message)

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/query",
        "raw_path": b"/v1/query",
        "query_string": b"",
        "headers": [
            (b"authorization", b"Bearer secret"),
            (b"content-type", b"application/json"),
        ],
        "server": ("testserver", 80),
        "client": ("testclient", 1234),
    }
    await asyncio.wait_for(test_app(scope, receive, send), 1)
    assert any(m["type"] == "http.response.start" and m["status"] == 499 for m in sent)
    assert runner.closed.is_set()
    assert not test_app.state.semaphore.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", [False, True])
async def test_query_external_cancel_propagates(
    tmp_path: Path, disconnect: bool
) -> None:
    """External cancellation propagates, including alongside a disconnect."""

    class CancelDuringCleanup(CleanupRunner):
        async def stream(
            self, prompt: str, options: ClaudeAgentOptions
        ) -> AsyncIterator[Message]:
            try:
                async for message in super().stream(prompt, options):
                    yield message
            finally:
                if disconnect:
                    task = asyncio.current_task()
                    assert task is not None
                    task.cancel()

    runner = CancelDuringCleanup()
    test_app = create_app(make_settings(tmp_path, max_concurrency=1), runner)

    async def receive() -> dict[str, Any]:
        if disconnect:
            await runner.started.wait()
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    route = next(
        r for r in test_app.routes if isinstance(r, APIRoute) and r.path == "/v1/query"
    )
    endpoint = cast("Callable[..., Coroutine[Any, Any, Any]]", route.endpoint)
    request = Request({"type": "http", "app": test_app}, receive)
    task = asyncio.create_task(
        endpoint(QueryRequest(agent="claude", sandbox=False, prompt="x"), request)
    )
    await asyncio.wait_for(runner.started.wait(), 1)
    if not disconnect:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert runner.closed.is_set()
    assert not test_app.state.semaphore.locked()


@pytest.mark.asyncio
async def test_query_disconnect_coalesced_external_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An external cancel coalesced with the disconnect cancel still propagates."""
    monkeypatch.setattr("agsrv.main.DISCONNECT_POLL_INTERVAL", 0)
    runner = CleanupRunner()
    test_app = create_app(make_settings(tmp_path, max_concurrency=1), runner)
    gate = asyncio.Event()

    async def receive() -> dict[str, Any]:
        await gate.wait()
        return {"type": "http.disconnect"}

    route = next(
        r for r in test_app.routes if isinstance(r, APIRoute) and r.path == "/v1/query"
    )
    endpoint = cast("Callable[..., Coroutine[Any, Any, Any]]", route.endpoint)
    task = asyncio.create_task(
        endpoint(
            QueryRequest(agent="claude", sandbox=False, prompt="x"),
            Request({"type": "http", "app": test_app}, receive),
        )
    )
    await asyncio.wait_for(runner.started.wait(), 1)
    gate.set()
    for _ in range(50):
        await asyncio.sleep(0)
        if task.cancelling():
            break
    assert task.cancelling() == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert runner.closed.is_set()
    assert not test_app.state.semaphore.locked()


@dataclass
class SandboxEnv:
    """Isolated service home and workspace for sandbox tests."""

    home: Path
    workspace: Path
    cli: Path


@pytest.fixture
def sbx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SandboxEnv]:
    """Provide a service home, a separate workspace, and a fake bundled CLI.

    Yields:
        The isolated environment.
    """
    previous_umask = os.umask(0o022)  # deterministic modes for test directories
    home = tmp_path / "home"
    workspace = tmp_path / "ws"
    home.mkdir()
    workspace.mkdir()
    sdk = tmp_path / "sdk"
    cli = sdk / "_bundled" / "claude"
    cli.parent.mkdir(parents=True)
    cli.write_text("#!/bin/sh\n")
    cli.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("agsrv.sandbox._SDK_DIR", sdk)
    yield SandboxEnv(home, workspace, cli)
    os.umask(previous_umask)


def sbx_client(
    sbx: SandboxEnv,
    *,
    root: Path | None = None,
    **kwargs: Any,  # ruff: ignore[any-type]
) -> tuple[TestClient, FakeRunner]:
    """Create a client configured for the default-on sandbox.

    Args:
        sbx: Sandbox environment.
        root: Workspace root override.
        **kwargs: Settings overrides.

    Returns:
        Client and runner.
    """
    return client_for(make_settings(root or sbx.workspace, **kwargs))


SANDBOX_BODY = {"agent": "claude", "prompt": "x", "sandbox": True}


def test_check_sandbox_request() -> None:
    """Verify the sandbox request guard."""
    check_sandbox_request(Settings(), requested=False)
    check_sandbox_request(Settings(), requested=True)
    check_sandbox_request(Settings(), requested=True, permission_mode="plan")
    check_sandbox_request(Settings(), requested=False, permission_mode="auto")
    with pytest.raises(HTTPException) as exc:
        check_sandbox_request(Settings(), requested=True, permission_mode="auto")
    assert exc.value.status_code == 403


def test_sandbox_false_when_off(tmp_path: Path) -> None:
    """Verify an explicit sandbox=false behaves as before."""
    client, runner = client_for(make_settings(tmp_path))
    response = client.post(
        "/v1/query",
        json={"agent": "claude", "prompt": "x", "sandbox": False},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert runner.options is not None
    assert runner.options.sandbox is None
    assert runner.options.cli_path is None


@pytest.mark.parametrize(
    "fast_mode", [{}, {"fast_mode": None}, {"fast_mode": False}, {"fast_mode": True}]
)
def test_sandbox_default_enabled(
    sbx: SandboxEnv, fast_mode: dict[str, bool | None]
) -> None:
    """Verify a request omitting sandbox enables a fail-closed sandbox."""
    settings = make_settings(sbx.workspace)
    client, runner = client_for(settings)
    response = client.post(
        "/v1/query", json={"agent": "claude", "prompt": "x", **fast_mode}, headers=AUTH
    )
    assert response.status_code == 200
    assert runner.options is not None
    if fast_mode.get("fast_mode") is True:
        assert runner.options.settings is not None
        assert json.loads(runner.options.settings) == {"fastMode": True}
    else:
        assert runner.options.settings is None
    assert runner.options.sandbox == build_sandbox_settings(settings, sbx.home)
    assert runner.options.cli_path == sbx.cli
    env = runner.options.env
    assert env["XDG_CACHE_HOME"] == str(sandbox_cache_dir(sbx.home))
    assert env["npm_config_cache"]
    assert env["npm_config_store_dir"]
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    cache = sandbox_cache_dir(sbx.home)
    assert cache.is_dir()
    assert cache.stat().st_mode & 0o777 == 0o700


def test_sandbox_explicit_opt_out(sbx: SandboxEnv) -> None:
    """Verify explicit sandbox=false does not configure sandbox options."""
    client, runner = sbx_client(sbx)
    response = client.post(
        "/v1/query",
        json={"agent": "claude", "prompt": "x", "sandbox": False},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert runner.options is not None
    assert runner.options.sandbox is None
    assert runner.options.cli_path is None
    assert "XDG_CACHE_HOME" not in runner.options.env


@pytest.mark.parametrize("path", ["/v1/query"])
@pytest.mark.parametrize("mode", ["auto", "bypassPermissions"])
def test_sandbox_rejects_egress_mode_in_request(
    sbx: SandboxEnv, path: str, mode: str
) -> None:
    """Verify requested egress-expanding modes are rejected with 403."""
    client, runner = sbx_client(sbx)
    body = {**SANDBOX_BODY, "permission_mode": mode}
    response = client.post(path, json=body, headers=AUTH)
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert mode in detail
    if mode == "auto":
        assert "incompatible with sandbox" in detail
    assert runner.stream_calls == 0
    assert not sandbox_cache_dir(sbx.home).exists()


@pytest.mark.parametrize("path", ["/v1/query"])
@pytest.mark.parametrize("mode", ["auto", "bypassPermissions"])
def test_sandbox_rejects_egress_mode_inherited(
    sbx: SandboxEnv, path: str, mode: str
) -> None:
    """Verify egress-expanding server default modes are rejected."""
    client, runner = sbx_client(sbx, permission_mode=mode)
    response = client.post(path, json=SANDBOX_BODY, headers=AUTH)
    assert response.status_code == 403
    if mode == "auto":
        assert "incompatible with sandbox" in response.json()["detail"]
    assert runner.stream_calls == 0


def test_sandbox_request_mode_overrides_default(sbx: SandboxEnv) -> None:
    """Verify a safe request mode wins over an egress-expanding default."""
    client, runner = sbx_client(sbx, permission_mode="auto")
    body = {**SANDBOX_BODY, "permission_mode": "plan"}
    assert client.post("/v1/query", json=body, headers=AUTH).status_code == 200
    assert runner.stream_calls == 1


def test_unsandboxed_egress_mode_allowed(sbx: SandboxEnv) -> None:
    """Verify the egress guard applies to sandboxed runs only."""
    client, runner = sbx_client(sbx, permission_mode="auto")
    response = client.post(
        "/v1/query",
        json={"agent": "claude", "sandbox": False, "prompt": "x"},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert runner.stream_calls == 1


def test_settings_reject_workspace_containing_home(sbx: SandboxEnv) -> None:
    """Reject the service home and its ancestors, including symlink aliases."""
    alias = sbx.workspace / "alias"
    alias.symlink_to(sbx.home.parent)
    for root in (sbx.home, sbx.home.parent, alias):
        with pytest.raises(ValueError, match="must not contain the service home"):
            Settings(workspace_root=root)


def test_sandbox_rejects_shared_cache(sbx: SandboxEnv) -> None:
    """Reject the cache and non-dedicated cache subdirectories."""
    cache = sbx.home / ".cache"
    (cache / "pip").mkdir(parents=True)
    (cache / "agsrv-sandbox" / "work").mkdir(parents=True)
    client, runner = sbx_client(sbx, root=cache)
    for cwd in (".", "pip"):
        body = {**SANDBOX_BODY, "cwd": cwd}
        response = client.post("/v1/query", json=body, headers=AUTH)
        assert response.status_code == 403, cwd
    assert runner.stream_calls == 0
    ok = {**SANDBOX_BODY, "cwd": "agsrv-sandbox/work"}
    assert client.post("/v1/query", json=ok, headers=AUTH).status_code == 200
    assert runner.stream_calls == 1


def test_sandbox_rejects_symlinked_shared_cache(tmp_path: Path) -> None:
    """Resolve a relocated cache before checking the sandbox cwd."""
    home = tmp_path / "home"
    home.mkdir()
    cache = tmp_path / "workspace" / "cache"
    (cache / "pip").mkdir(parents=True)
    (cache / "agsrv-sandbox" / "work").mkdir(parents=True)
    (home / ".cache").symlink_to(cache)
    with pytest.raises(HTTPException) as exc_info:
        check_sandbox_cwd(cache / "pip", home)
    assert exc_info.value.status_code == 403
    check_sandbox_cwd(cache / "agsrv-sandbox" / "work", home)


def test_sandbox_allows_unrelated_cwd_under_home(sbx: SandboxEnv) -> None:
    """Verify a sibling directory of ~/.cache is allowed."""
    project = sbx.home / "project"
    project.mkdir()
    client, runner = sbx_client(sbx, root=project)
    body = SANDBOX_BODY
    assert client.post("/v1/query", json=body, headers=AUTH).status_code == 200
    assert runner.stream_calls == 1


def test_unsandboxed_cwd_at_cache_allowed(sbx: SandboxEnv) -> None:
    """Verify the cache guard applies to sandboxed runs only."""
    cache = sbx.home / ".cache"
    cache.mkdir()
    client, runner = sbx_client(sbx, root=cache)
    assert (
        client.post(
            "/v1/query",
            json={"agent": "claude", "sandbox": False, "prompt": "x"},
            headers=AUTH,
        ).status_code
        == 200
    )
    assert runner.stream_calls == 1


def test_sandbox_cache_hardening(sbx: SandboxEnv) -> None:
    """Verify an existing broad-mode cache directory is tightened."""
    cache = sandbox_cache_dir(sbx.home)
    cache.mkdir(parents=True)
    cache.chmod(0o777)
    assert prepare_sandbox_cache(sbx.home) == cache
    assert cache.stat().st_mode & 0o777 == 0o700


def test_sandbox_cache_chmod_ineffective(
    sbx: SandboxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify failing to tighten the mode fails closed."""
    cache = sandbox_cache_dir(sbx.home)
    cache.mkdir(parents=True)
    cache.chmod(0o777)

    def no_chmod(self: Path, mode: int) -> None:
        assert self == cache
        assert mode == 0o700

    monkeypatch.setattr(Path, "chmod", no_chmod)
    with pytest.raises(SandboxSetupError, match="cannot be set"):
        prepare_sandbox_cache(sbx.home)


def test_sandbox_cache_leaf_0500_corrected(sbx: SandboxEnv) -> None:
    """Verify a leaf the owner cannot write is corrected to exactly 0700."""
    cache = sandbox_cache_dir(sbx.home)
    cache.mkdir(parents=True)
    cache.chmod(0o500)
    assert prepare_sandbox_cache(sbx.home) == cache
    assert cache.stat().st_mode & 0o777 == 0o700


def test_sandbox_cache_leaf_0500_uncorrectable(
    sbx: SandboxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify an unusable leaf that cannot be corrected is a 503 error."""
    cache = sandbox_cache_dir(sbx.home)
    cache.mkdir(parents=True)
    cache.chmod(0o500)

    def no_chmod(self: Path, mode: int) -> None:
        """Ignore the chmod request."""

    monkeypatch.setattr(Path, "chmod", no_chmod)
    with pytest.raises(SandboxSetupError, match="cannot be set"):
        prepare_sandbox_cache(sbx.home)


@pytest.mark.parametrize("mode", [0o777, 0o775, 0o757, 0o770, 0o722])
def test_sandbox_cache_parent_writable_rejected(sbx: SandboxEnv, mode: int) -> None:
    """Verify a group/other-writable ~/.cache is rejected before the leaf."""
    parent = sbx.home / ".cache"
    parent.mkdir()
    parent.chmod(mode)
    with pytest.raises(SandboxSetupError, match="writable"):
        prepare_sandbox_cache(sbx.home)
    assert not (parent / "agsrv-sandbox").exists()


def test_sandbox_cache_parent_0755_accepted(sbx: SandboxEnv) -> None:
    """Verify a conventional 0755 parent passes."""
    parent = sbx.home / ".cache"
    parent.mkdir()
    parent.chmod(0o755)
    assert prepare_sandbox_cache(sbx.home) == parent / "agsrv-sandbox"


def test_sandbox_parent_writable_is_503(sbx: SandboxEnv) -> None:
    """Verify a writable cache parent yields a 503 on both routes."""
    parent = sbx.home / ".cache"
    parent.mkdir()
    parent.chmod(0o777)
    client, runner = sbx_client(sbx)
    for path in ("/v1/query",):
        response = client.post(path, json=SANDBOX_BODY, headers=AUTH)
        assert response.status_code == 503
    assert runner.stream_calls == 0


def test_sandbox_cache_symlink_leaf(sbx: SandboxEnv) -> None:
    """Verify a symlinked cache leaf is rejected, not followed."""
    cache = sandbox_cache_dir(sbx.home)
    cache.parent.mkdir()
    target = sbx.home / "elsewhere"
    target.mkdir()
    cache.symlink_to(target)
    with pytest.raises(SandboxSetupError, match="owned directory"):
        prepare_sandbox_cache(sbx.home)


def test_sandbox_cache_not_directory(sbx: SandboxEnv) -> None:
    """Verify a regular file in place of the cache is rejected."""
    cache = sandbox_cache_dir(sbx.home)
    cache.parent.mkdir()
    cache.write_text("x")
    with pytest.raises(SandboxSetupError, match="owned directory"):
        prepare_sandbox_cache(sbx.home)


def test_sandbox_cache_wrong_owner(
    sbx: SandboxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify a directory owned by someone else is rejected."""
    monkeypatch.setattr("agsrv.sandbox.os.geteuid", lambda: 4242424)
    with pytest.raises(SandboxSetupError, match="owned directory"):
        prepare_sandbox_cache(sbx.home)


def test_sandbox_cache_missing_home_dir(tmp_path: Path) -> None:
    """Verify OS errors become SandboxSetupError."""
    with pytest.raises(SandboxSetupError, match="setup failed"):
        prepare_sandbox_cache(tmp_path / "absent")


def test_sandbox_setup_failure_is_controlled(
    sbx: SandboxEnv,
) -> None:
    """Verify a cache setup failure is a controlled 503 and the slot is freed."""
    (sbx.home / ".cache").write_text("not a directory")
    client, runner = sbx_client(sbx, max_concurrency=1)
    for path in ("/v1/query", "/v1/query"):
        response = client.post(path, json=SANDBOX_BODY, headers=AUTH)
        assert response.status_code == 503
        assert "sandbox unavailable" in response.json()["detail"]
    assert runner.stream_calls == 0
    ok = client.post(
        "/v1/query",
        json={"agent": "claude", "sandbox": False, "prompt": "x"},
        headers=AUTH,
    )
    assert ok.status_code == 200


def test_sandbox_capacity_checked_before_setup(sbx: SandboxEnv) -> None:
    """Verify 429 is returned before any filesystem setup happens."""
    runner = FakeRunner()
    test_app = create_app(make_settings(sbx.workspace), runner)
    test_app.state.semaphore = asyncio.Semaphore(0)
    client = TestClient(test_app)
    response = client.post("/v1/query", json=SANDBOX_BODY, headers=AUTH)
    assert response.status_code == 429
    assert not (sbx.home / ".cache").exists()
    assert runner.stream_calls == 0


@pytest.mark.parametrize("path", ["/v1/query"])
def test_sandbox_429_beats_unavailable_home(
    sbx: SandboxEnv, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """Verify a saturated request gets 429 even when the home is unavailable."""
    runner = FakeRunner()
    test_app = create_app(make_settings(sbx.workspace), runner)
    test_app.state.semaphore = asyncio.Semaphore(0)
    client = TestClient(test_app)

    def boom() -> Path:
        msg = "no home"
        raise RuntimeError(msg)

    monkeypatch.setattr(Path, "home", boom)
    response = client.post(path, json=SANDBOX_BODY, headers=AUTH)
    assert response.status_code == 429
    assert runner.stream_calls == 0


def test_sandbox_missing_home(sbx: SandboxEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify an unknown or relative home is a controlled 503."""
    client, runner = sbx_client(sbx)

    def boom() -> Path:
        msg = "no home"
        raise RuntimeError(msg)

    monkeypatch.setattr(Path, "home", boom)
    response = client.post("/v1/query", json=SANDBOX_BODY, headers=AUTH)
    assert response.status_code == 503
    with pytest.raises(SandboxSetupError, match="unknown"):
        service_home()
    monkeypatch.setattr(Path, "home", lambda: Path("relative"))
    with pytest.raises(SandboxSetupError, match="not absolute"):
        service_home()
    assert runner.stream_calls == 0


def test_sandbox_missing_bundled_cli(
    sbx: SandboxEnv,
) -> None:
    """Verify a missing or non-executable bundled CLI fails closed."""
    client, runner = sbx_client(sbx)
    sbx.cli.chmod(0o644)
    response = client.post("/v1/query", json=SANDBOX_BODY, headers=AUTH)
    assert response.status_code == 503
    assert "bundled Claude CLI" in response.json()["detail"]
    sbx.cli.unlink()
    response = client.post("/v1/query", json=SANDBOX_BODY, headers=AUTH)
    assert response.status_code == 503
    assert runner.stream_calls == 0
    assert not sandbox_cache_dir(sbx.home).exists()


def test_bundled_cli_path_default_is_real_sdk() -> None:
    """Verify the real SDK bundles an executable CLI."""
    assert bundled_cli_path().parent.name == "_bundled"


def test_build_options_failure_releases_slot(
    sbx: SandboxEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify unexpected option-building errors release the slot and propagate."""

    def boom(*_a: object, **_k: object) -> ClaudeAgentOptions:
        msg = "unexpected"
        raise RuntimeError(msg)

    monkeypatch.setattr("agsrv.main.build_options", boom)
    test_app = create_app(make_settings(sbx.workspace, max_concurrency=1), FakeRunner())
    client = TestClient(test_app, raise_server_exceptions=False)
    for _ in range(2):
        response = client.post(
            "/v1/query",
            json={"agent": "claude", "sandbox": False, "prompt": "x"},
            headers=AUTH,
        )
        assert response.status_code == 500


def test_build_sandbox_settings(tmp_path: Path) -> None:
    """Verify the sandbox settings dict exactly."""
    home = tmp_path / "h"
    settings = Settings(sandbox_allowed_domains=("github.com",))
    result = build_sandbox_settings(settings, home)
    assert result == {
        "enabled": True,
        "failIfUnavailable": True,
        "autoAllowBashIfSandboxed": False,
        "allowUnsandboxedCommands": False,
        "enableWeakerNestedSandbox": False,
        "network": {"allowedDomains": ["github.com"], "allowAllUnixSockets": False},
        "filesystem": {
            "allowWrite": [str(home / ".cache" / "agsrv-sandbox")],
            "denyRead": [
                str(home / ".ssh"),
                str(home / ".gnupg"),
                str(home / ".aws"),
                str(home / ".config"),
                str(home / ".claude.json"),
                str(home / ".claude" / ".credentials.json"),
                str(home / "Dropbox"),
                "/run/user",
                "/run/dbus",
            ],
            "allowRead": [str(home / ".config" / "gh"), str(home / ".config" / "uv")],
        },
        "credentials": {
            "envVars": [
                {"name": "ANTHROPIC_API_KEY", "mode": "deny"},
                {"name": "AGSRV_API_KEY", "mode": "deny"},
                {"name": "CLAUDE_CODE_MESSAGING_TOKEN", "mode": "deny"},
                {"name": "CLAUDE_CODE_OAUTH_TOKEN", "mode": "deny"},
            ]
        },
    }
    assert "excludedCommands" not in result
    assert result["failIfUnavailable"] is True
    assert result["allowUnsandboxedCommands"] is False


def test_sandbox_env(tmp_path: Path) -> None:
    """Verify cache redirection environment."""
    cache = tmp_path / ".cache" / "agsrv-sandbox"
    assert sandbox_env(tmp_path) == {
        "XDG_CACHE_HOME": str(cache),
        "npm_config_cache": str(cache / "npm"),
        "npm_config_store_dir": str(cache / "pnpm-store"),
    }
