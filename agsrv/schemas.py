"""Request and response models."""

from __future__ import annotations

from typing import Any, Literal

# Pydantic resolves these Literal aliases at runtime, so the import cannot
# move into a TYPE_CHECKING block.
from claude_agent_sdk import EffortLevel, PermissionMode  # ruff: ignore[typing-only-third-party-import]
from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """One ephemeral agent run. Unknown (e.g. session) fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    agent: Literal["claude"]
    prompt: str = Field(min_length=1)
    model: str | None = None
    cwd: str | None = None
    allowed_tools: list[str] | None = None
    permission_mode: PermissionMode | None = None
    max_turns: int | None = Field(default=None, ge=1, le=1000)
    effort: EffortLevel | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)
    include_messages: bool = True
    sandbox: bool = True


class CodexOptions(BaseModel):
    """Codex sandbox settings."""

    model_config = ConfigDict(extra="forbid")
    sandbox: Literal["read-only", "workspace-write"] = "read-only"
    network_access: bool = False


class CodexQueryRequest(BaseModel):
    """One ephemeral Codex run."""

    model_config = ConfigDict(extra="forbid")
    agent: Literal["codex"]
    prompt: str = Field(min_length=1)
    model: str | None = None
    effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None
    cwd: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)
    include_messages: bool = True
    options: CodexOptions = Field(default_factory=CodexOptions)


class QueryResponse(BaseModel):
    """Final agent result with an optional message transcript."""

    agent: Literal["claude"] = "claude"
    result: str | None
    session_id: str
    model: str | None
    subtype: str
    is_error: bool
    num_turns: int
    duration_ms: int
    duration_api_ms: int
    total_cost_usd: float | None
    usage: dict[str, Any] | None
    stop_reason: str | None
    messages: list[dict[str, Any]]


class CodexQueryResponse(BaseModel):
    """Final Codex result and backend metadata."""

    agent: Literal["codex"]
    result: str | None
    model: str | None
    is_error: bool
    duration_ms: int
    usage: dict[str, Any] | None
    backend: dict[str, Any]


class InfoResponse(BaseModel):
    """Server capabilities and defaults."""

    version: str
    stateless: bool
    default_model: str | None
    workspace_root: str
    default_permission_mode: str
    default_allowed_tools: list[str]
    max_concurrency: int
    auth_required: bool
    sandbox_allowed_domains: list[str]
