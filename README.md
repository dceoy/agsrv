# agsrv

HTTP API for remote Claude and Codex execution using their Python SDKs.

[![CI](https://github.com/dceoy/agsrv/actions/workflows/ci.yml/badge.svg)](https://github.com/dceoy/agsrv/actions/workflows/ci.yml)

Each query is one ephemeral Claude or Codex run. There are no resumable agent
sessions or conversation state. The synchronous API cancels on disconnect; the
durable job API stores outcomes in SQLite so short-lived clients can submit,
disconnect, and retrieve the result later.

## Endpoints

| Method | Path               | Auth | Description                                |
| ------ | ------------------ | ---- | ------------------------------------------ |
| GET    | `/health`          | no   | Readiness probe: `{"status": "ok"}`        |
| GET    | `/v1/info`         | yes  | Server defaults and limits                 |
| GET    | `/v1/agents`       | yes  | Agent capabilities and health              |
| GET    | `/v1/agents/codex/rate-limits` | yes | Codex account limits              |
| POST   | `/v1/query`        | yes  | Run one agent query, return the result |
| POST   | `/v1/jobs`        | yes  | Submit a durable asynchronous job (202)  |
| GET    | `/v1/jobs/{job_id}` | yes | Poll status and terminal result         |
| POST   | `/v1/jobs/{job_id}/cancel` | yes | Request cancellation             |

See [docs/api/rest-api.md](docs/api/rest-api.md) for the full reference.

## Quickstart

Server:

```sh
uv sync
export AGSRV_API_KEY="$(openssl rand -hex 32)"
export AGSRV_WORKSPACE_ROOT=/srv/workspaces
export ANTHROPIC_API_KEY=...   # or log in with `claude` as the server user
uv run uvicorn agsrv.main:app --host 127.0.0.1 --port 8000
```

Client (another machine, via TLS or a private network):

```sh
curl -sS https://agsrv.example.net/v1/query \
  -H "Authorization: Bearer $AGSRV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"agent": "claude", "prompt": "List the files in this directory", "cwd": "myproject", "max_turns": 5}'

curl -sS https://agsrv.example.net/v1/jobs \
  -H "Authorization: Bearer $AGSRV_API_KEY" \
  -H "Idempotency-Key: muse-vm-task-001" \
  -H "Content-Type: application/json" \
  -d '{"agent": "claude", "prompt": "Summarize README.md", "cwd": "myproject"}'
# Poll GET /v1/jobs/<job_id> from a replacement VM using the same bearer key.
```

## Configuration

| Variable                          | Default                          | Description                                              |
| --------------------------------- | -------------------------------- | -------------------------------------------------------- |
| `AGSRV_CODEX_BIN`                | unset (SDK bundled binary)       | Optional Codex executable path                           |
| `AGSRV_CODEX_ENABLED`            | `true`                           | Enable Codex                                             |
| `AGSRV_CODEX_MAX_CONCURRENCY`    | `2`                              | Simultaneous Codex runs                                  |
| `AGSRV_API_KEY`                  | unset                            | Bearer token clients must send                           |
| `AGSRV_ALLOW_UNAUTHENTICATED`    | `false`                          | Allow requests with no key configured (local dev only)   |
| `AGSRV_WORKSPACE_ROOT`           | current directory                | All request `cwd` values must resolve inside this path   |
| `AGSRV_DEFAULT_MODEL`            | unset (SDK default)              | Model used when a request omits `model`                  |
| `AGSRV_PERMISSION_MODE`          | `acceptEdits`                    | Default permission mode                                  |
| `AGSRV_ALLOWED_TOOLS`            | `Read,Write,Edit,Bash,Glob,Grep` | Default auto-approve list (not a restriction; see below) |
| `AGSRV_MAX_CONCURRENCY`          | `4`                              | Simultaneous runs; excess requests get HTTP 429          |
| `AGSRV_MAX_TIMEOUT_SECONDS`      | `3600`                           | Default and maximum run duration                         |
| `AGSRV_JOBS_DB_PATH`             | `~/.local/state/agsrv/jobs.sqlite3` | Persistent SQLite database (single worker only) |
| `AGSRV_JOBS_MAX_PENDING`         | `32`                             | Maximum queued and running jobs (429 when full) |
| `AGSRV_JOBS_RETENTION_SECONDS`   | `86400`                          | Terminal record and idempotency-key retention (24 h) |
| `AGSRV_JOBS_MAX_PAYLOAD_BYTES`   | `65536`                          | Maximum canonical request size |
| `AGSRV_JOBS_MAX_RESULT_BYTES`    | `1048576`                        | Maximum persisted result size |
| `AGSRV_SANDBOX_ALLOWED_DOMAINS`  | GitHub, PyPI, and npm hosts (see below) | Comma-separated egress allow-list for sandboxed Bash; set-but-empty means no egress |

The default sandbox domains are `github.com`, `api.github.com`,
`codeload.github.com`, `uploads.github.com`, `raw.githubusercontent.com`,
`objects.githubusercontent.com`, `release-assets.githubusercontent.com`,
`pypi.org`, `files.pythonhosted.org`, and `registry.npmjs.org`.

(including an empty string, `OFF`, `Opt-In`, padded values, or `required`)
aborts startup. Domain entries are trimmed and lowercased, and empty
comma-separated entries are dropped (`a.com,,b.com` is two entries; a set but
empty or blank list means no egress). Each entry must be an ASCII hostname with
at least two DNS labels, each 1-63 characters (letters, digits, inner hyphens),
and at most 253 characters in total. A single leading `*.` wildcard is allowed;
embedded wildcards, empty labels (`a..com`), trailing dots, and raw Unicode
names (use punycode, `xn--...`) abort startup. Entries are checked as ASCII
before lowercasing, so look-alike characters such as the Kelvin sign are rejected.

Codex requests use `"agent": "codex"` and may set `options.sandbox` to
`"read-only"` (default) or `"workspace-write"`, and `options.network_access`
to a boolean (default `false`). Codex threads are ephemeral, approvals are
denied, and explicit network access uses the SDK’s low-level sandbox policy.
Rate limits come from `account/rateLimits/read`.

## Security model

- **The API key is a shell.** Holding the key is equivalent to running a shell
  as the service user (with the workspace as the starting directory). Protect
  it accordingly; the default Claude Bash sandbox is only one part of defence in depth.
- **API key required by default.** Without `AGSRV_API_KEY` (or with an empty
  one), execution endpoints return 503 unless
  `AGSRV_ALLOW_UNAUTHENTICATED=true` is set explicitly. Keys are compared in
  constant time.
- **`allowed_tools` is an auto-approve list, not a restriction.** It names the
  tools a run may use without a permission decision; it does not remove other
  tools, and under the default `acceptEdits` mode file edits are approved
  regardless. The server-side `AGSRV_ALLOWED_TOOLS` is only a default — any
  key holder can send a different list per request.
- **The agent can read the server's secrets.** A run inherits the service
  environment, including `ANTHROPIC_API_KEY`, the Claude login credentials,
  and `AGSRV_API_KEY` itself. Prompt-injected workspace content can therefore
  exfiltrate them; do not point the service at untrusted workspaces. The
  default-on sandbox only narrows the Bash path (see below).
- **Default-on Bash sandbox.** Claude requests default to `sandbox: true` and
  run Bash and its child processes inside the CLI sandbox. Set `sandbox: false`
  only for a deliberate control operation that needs access to the host:
  writes are limited to the working directory, temp, and a dedicated cache
  directory; egress is limited to the allow-list; the systemd/D-Bus user bus is
  unreachable; common secret directories are unreadable; and the API keys are
  unset in the Bash environment. The sandbox is fail-closed
  (`failIfUnavailable`): if it cannot start, the run fails rather than running
  unsandboxed. A sandboxed request is also rejected with 403 when its
  effective permission mode (request value, else `AGSRV_PERMISSION_MODE`) is
  `auto` or `bypassPermissions`, because those modes can approve hosts outside
  the allow-list; and when the resolved `cwd` is `~/.cache`, an ancestor of it
  (including the service home directory), or inside `~/.cache` outside
  `~/.cache/agsrv-sandbox`, because the CLI makes the whole working directory
  writable. Sandboxed runs use only the CLI bundled with the SDK (they fail
  closed if it is missing) and verify that the cache directory is a real
  directory owned by the service user, tightening it to mode 0700. Limits to be
  aware of: only Bash is covered, not
  Read/Edit/Write, Glob/Grep, WebFetch, or MCP. Because `Read` is auto-approved,
  secret files such as the service env file can still be read, and anything
  readable can be exfiltrated to allow-listed hosts (GitHub). The `gh` token
  stays readable so push and PR creation work; use a minimal-scope token.

- **Workspace confinement.** Request `cwd` is resolved (symlinks included) and
  must stay inside `AGSRV_WORKSPACE_ROOT`. Note this confines the agent's
  starting directory, not what its tools (e.g. `Bash`) can reach; run the server
  as an unprivileged user or in a container for real isolation.
  It is not a meaningful boundary on its own: with `Bash` auto-approved, the
  default posture is already shell-equivalent.
- **Client disconnects.** Claude `POST /v1/query` promptly cancels the run and frees
  its concurrency slot when the client disconnects. Durable `/v1/jobs` runs
  are owned by the server and continue after the client disconnects.
- **Bind and TLS.** Bind to loopback or a private interface and put TLS in front
  (reverse proxy, Tailscale, or an SSH tunnel). The API key travels in a header.
- **Claude statelessness.** Request bodies containing `resume`, `session_id`, or any
  other unknown field are rejected with 422. Runs are hermetic: CLI session
  persistence is disabled, so no resumable transcript is written, and host
  and project settings are not loaded (`setting_sources` is empty), so hooks
  or MCP servers from those settings cannot leak into a run. Auto memory is
  disabled (`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`), and MCP configuration is
  strict and isolated. Managed policy settings and `~/.claude.json` still load.
  The default system prompt is the Claude Code preset. What still persists after
  a run: files the agent creates or edits in the workspace, and ordinary CLI
  bookkeeping under `~/.claude`.

## Durable job contract

`POST /v1/jobs` requires the same validated Claude/Codex request as
`POST /v1/query`. Provide a stable `Idempotency-Key` on retries: an identical
request reuses its original `job_id`, while conflicting content returns 409.
Poll `GET /v1/jobs/{job_id}` until `succeeded`, `failed`, `cancelled`, or
`interrupted`; cancellation is requested via POST to `/cancel`.
A result with `is_error: true` is recorded as `failed`. Error codes are
sanitized. Non-terminal jobs are bounded, and queue saturation returns 429
with `Retry-After`. Expired terminal records and idempotency keys are removed
after 24 hours by default. Jobs run in-process and **do not survive process
termination**: after restart nonterminal records become `interrupted`
without replaying filesystem or Git side effects. Only one uvicorn worker may
use a given SQLite database. Jobs for overlapping workspace directories are
serialized. Shared API keys represent one caller and authorize access to the
same job records; use a separate server/key for caller isolation.

## Development

Requirements: Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/dceoy/agsrv.git
cd agsrv
uv sync
uv run uvicorn agsrv.main:app --host 127.0.0.1 --port 8000
```

Swagger UI is at `http://127.0.0.1:8000/docs`.

```sh
uv run ruff format --check .
uv run ruff check .
uv run pyright .
uv run pytest
uv run mkdocs build --strict
```

Dependency versions are tracked in the committed `uv.lock`.

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
