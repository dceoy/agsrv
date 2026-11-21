"""Runtime settings loaded from environment variables."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_DEFAULT_TOOLS = "Read,Write,Edit,Bash,Glob,Grep"

DEFAULT_SANDBOX_ALLOWED_DOMAINS: tuple[str, ...] = (
    "github.com",
    "api.github.com",
    "codeload.github.com",
    "uploads.github.com",
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
)
_DOMAIN_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_MAX_HOSTNAME_LENGTH = 253
_MIN_HOSTNAME_LABELS = 2


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _env_api_key() -> str | None:
    raw = os.environ.get("AGSRV_API_KEY")
    if raw is None:
        return None
    return raw.strip() or None


def _valid_domain(domain: str) -> bool:
    """Check an allow-list entry.

    Args:
        domain: Lowercased entry: ASCII host with an optional leading ``*.``.

    Returns:
        Whether every DNS label is 1-63 chars and the host is at most 253.
    """
    host = domain.removeprefix("*.")
    labels = host.split(".")
    return (
        len(host) <= _MAX_HOSTNAME_LENGTH
        and len(labels) >= _MIN_HOSTNAME_LABELS
        and all(_DOMAIN_LABEL_RE.fullmatch(label) for label in labels)
    )


def _env_sandbox_domains() -> tuple[str, ...]:
    raw = os.environ.get("AGSRV_SANDBOX_ALLOWED_DOMAINS")
    if raw is None:
        return DEFAULT_SANDBOX_ALLOWED_DOMAINS
    entries = [d.strip() for d in raw.split(",") if d.strip()]
    for d in entries:
        # Validate as ASCII before lowercasing: casefolding can map Unicode
        # (e.g. the Kelvin sign) onto ASCII letters.
        if not d.isascii() or not _valid_domain(d.lower()):
            msg = f"invalid domain in AGSRV_SANDBOX_ALLOWED_DOMAINS: {d!r}"
            raise ValueError(msg)
    return tuple(d.lower() for d in entries)


def _env_staging_dirs() -> tuple[tuple[Path, Path], ...]:
    """Parse ``AGSRV_TMP_STAGING_DIRS`` (``mount=staging`` pairs, comma-separated).

    Returns:
        Resolved ``(mount root, staging directory)`` pairs.

    Raises:
        ValueError: If an entry is not ``mount=staging``.
    """
    raw = os.environ.get("AGSRV_TMP_STAGING_DIRS", "")
    pairs: list[tuple[Path, Path]] = []
    for entry in (e.strip() for e in raw.split(",")):
        if not entry:
            continue
        mount, sep, stage = entry.partition("=")
        if not sep or not mount.strip() or not stage.strip():
            msg = f"invalid AGSRV_TMP_STAGING_DIRS entry: {entry!r}"
            raise ValueError(msg)
        pairs.append((Path(mount.strip()).resolve(), Path(stage.strip()).resolve()))
    return tuple(pairs)


@dataclass(frozen=True)
class Settings:
    """Server configuration.

    Attributes:
        tmp_min_free_bytes: Minimum free bytes on /tmp (0 disables).
        workspace_min_free_bytes: Minimum free bytes on the workspace (0 disables).
        api_key: Bearer token required from clients, if any.
        allow_unauthenticated: Permit requests when no API key is configured.
        workspace_root: Directory that all agent working directories must stay in.
        default_model: Model used when a request does not specify one.
        permission_mode: Default permission mode for agent runs.
        allowed_tools: Default tool allow-list for agent runs.
        max_concurrency: Maximum number of simultaneous agent runs.
        max_timeout_seconds: Upper bound (and default) for run duration.
        sandbox_allowed_domains: Network egress allow-list for sandboxed Bash.
        codex_bin: Optional path to the Codex executable.
        codex_enabled: Whether Codex execution is enabled.
        codex_max_concurrency: Maximum number of simultaneous Codex runs.
        jobs_db_path: Durable SQLite WAL database (single worker).
        jobs_max_pending: Maximum queued and running jobs.
        jobs_retention_seconds: Retention for completed jobs and deduplication keys.
        jobs_max_payload_bytes: Maximum validated request size.
        jobs_max_result_bytes: Maximum persisted response size.
        tmp_staging_dirs: ``(workspace mount root, staging directory)`` pairs.
            Each staging directory must be service-owned, mode 0700, outside the
            workspace root and on the mount it serves. Turns in a mount without
            one are unavailable.
    """

    api_key: str | None = None
    allow_unauthenticated: bool = False
    workspace_root: Path = field(default_factory=lambda: Path.cwd().resolve())
    default_model: str | None = None
    permission_mode: str = "acceptEdits"
    allowed_tools: tuple[str, ...] = tuple(_DEFAULT_TOOLS.split(","))
    max_concurrency: int = 4
    max_timeout_seconds: float = 3600.0
    sandbox_allowed_domains: tuple[str, ...] = DEFAULT_SANDBOX_ALLOWED_DOMAINS
    codex_bin: str | None = None
    codex_enabled: bool = True
    codex_max_concurrency: int = 2
    jobs_db_path: Path = field(
        default_factory=lambda: Path.home() / ".local/state/agsrv/jobs.sqlite3"
    )
    jobs_max_pending: int = 32
    jobs_retention_seconds: int = 86400
    jobs_max_payload_bytes: int = 65536
    jobs_max_result_bytes: int = 1048576

    tmp_min_free_bytes: int = 536870912
    workspace_min_free_bytes: int = 1073741824
    tmp_staging_dirs: tuple[tuple[Path, Path], ...] = ()

    def __post_init__(self) -> None:
        """Validate disk thresholds and staging locations.

        Raises:
            ValueError: If a threshold is negative or a staging path is relative.
        """
        for mount, stage in self.tmp_staging_dirs:
            if not (mount.is_absolute() and stage.is_absolute()):
                msg = "AGSRV_TMP_STAGING_DIRS paths must be absolute"
                raise ValueError(msg)
        for name in ("tmp_min_free_bytes", "workspace_min_free_bytes"):
            if getattr(self, name) < 0:
                msg = f"AGSRV_{name.upper()} must be nonnegative"
                raise ValueError(msg)

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from ``AGSRV_*`` environment variables.

        Returns:
            Parsed settings with defaults applied.
        """
        tools = os.environ.get("AGSRV_ALLOWED_TOOLS", _DEFAULT_TOOLS)
        return cls(
            tmp_staging_dirs=_env_staging_dirs(),
            tmp_min_free_bytes=int(
                os.environ.get("AGSRV_TMP_MIN_FREE_BYTES", "536870912")
            ),
            workspace_min_free_bytes=int(
                os.environ.get("AGSRV_WORKSPACE_MIN_FREE_BYTES", "1073741824")
            ),
            api_key=_env_api_key(),
            allow_unauthenticated=_env_bool("AGSRV_ALLOW_UNAUTHENTICATED"),
            workspace_root=Path(
                os.environ.get("AGSRV_WORKSPACE_ROOT", str(Path.cwd()))
            ).resolve(),
            default_model=os.environ.get("AGSRV_DEFAULT_MODEL"),
            permission_mode=os.environ.get("AGSRV_PERMISSION_MODE", "acceptEdits"),
            allowed_tools=tuple(t.strip() for t in tools.split(",") if t.strip()),
            max_concurrency=int(os.environ.get("AGSRV_MAX_CONCURRENCY", "4")),
            max_timeout_seconds=float(
                os.environ.get("AGSRV_MAX_TIMEOUT_SECONDS", "3600")
            ),
            sandbox_allowed_domains=_env_sandbox_domains(),
            codex_bin=os.environ.get("AGSRV_CODEX_BIN"),
            codex_enabled=_env_bool("AGSRV_CODEX_ENABLED", default=True),
            codex_max_concurrency=int(
                os.environ.get("AGSRV_CODEX_MAX_CONCURRENCY", "2")
            ),
            jobs_db_path=Path(
                os.environ.get(
                    "AGSRV_JOBS_DB_PATH",
                    str(Path.home() / ".local/state/agsrv/jobs.sqlite3"),
                )
            )
            .expanduser()
            .resolve(),
            jobs_max_pending=int(os.environ.get("AGSRV_JOBS_MAX_PENDING", "32")),
            jobs_retention_seconds=int(
                os.environ.get("AGSRV_JOBS_RETENTION_SECONDS", "86400")
            ),
            jobs_max_payload_bytes=int(
                os.environ.get("AGSRV_JOBS_MAX_PAYLOAD_BYTES", "65536")
            ),
            jobs_max_result_bytes=int(
                os.environ.get("AGSRV_JOBS_MAX_RESULT_BYTES", "1048576")
            ),
        )
