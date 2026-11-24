# REST API

All `/v1/*` endpoints require `Authorization: Bearer <AGSRV_API_KEY>`.
FastAPI also serves `/docs` and `/openapi.json`.

## Statelessness contract

Every request is one ephemeral Claude, Codex or Muse run. The server keeps no sessions
and offers no resume, continue, or fork. The `session_id` in responses is
informational only and cannot be sent back: request bodies with unknown fields
(such as `resume` or `session_id`) are rejected with `422`. To continue work,
include the needed context in the next prompt.

Claude runs are hermetic. CLI session persistence is disabled, so the CLI writes no
resumable transcript, and host and project settings are not loaded
(`setting_sources` is empty): hooks, MCP servers, and permission rules from
the host's or the workspace's `.claude/settings.json` never enter a run. Auto
memory is disabled (`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`), and MCP configuration
is strict and isolated. Managed policy settings and `~/.claude.json` still load.
What persists after a run is only what the agent itself writes — files in the
workspace and ordinary CLI bookkeeping under `~/.claude`.

## `GET /health`

No authentication.

```json
{ "status": "ok" }
```

## `GET /v1/info`

```json
{
  "version": "0.5.3",
  "stateless": true,
  "default_model": null,
  "workspace_root": "/srv/workspaces",
  "default_permission_mode": "acceptEdits",
  "default_allowed_tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
  "max_concurrency": 4,
  "auth_required": true,
  "sandbox_allowed_domains": ["github.com", "api.github.com", "codeload.github.com", "uploads.github.com", "raw.githubusercontent.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com", "pypi.org", "files.pythonhosted.org", "registry.npmjs.org"]
}
```

## `POST /v1/query`

Claude request (`agent` and `prompt` are required):

```json
{
  "agent": "claude",
  "prompt": "List the Python files",
  "model": null,
  "cwd": "myproject",
  "allowed_tools": ["Read", "Glob"],
  "permission_mode": "default",
  "max_turns": 5,
  "effort": "low",
  "timeout_seconds": 300,
  "include_messages": true,
  "sandbox": true
}
```

- `include_messages`: when `false`, the response omits the transcript (`messages`
  is `[]`); intended for result-only clients.
- `sandbox`: defaults to `true`, running Bash inside the fail-closed CLI
  sandbox. Only explicit `false` opts out for host-control operations. Never
  retry a failed sandboxed run with `sandbox: false`. Requests are rejected
  with 403 if the effective permission mode (request value, else the server
  default) is `auto` or `bypassPermissions` (incompatible with the sandbox's
  egress allow-list), or if the resolved `cwd` overlaps the shared cache: it is
  `~/.cache`, an ancestor of it (such as the service home directory), or inside
  `~/.cache` outside `~/.cache/agsrv-sandbox`. If sandbox prerequisites cannot
  be established (unknown home, unusable cache directory, missing bundled CLI)
  the response is 503.
- `cwd`: relative to `AGSRV_WORKSPACE_ROOT`, or absolute inside it.
- `allowed_tools`: an auto-approve list, not a restriction. It names tools the
  run may use without a permission decision; it does not remove other tools,
  and file edits are still approved under `acceptEdits`. When omitted, the
  server default (`AGSRV_ALLOWED_TOOLS`) applies — a default, not a ceiling.
- `permission_mode`: `default`, `acceptEdits`, `plan`, `bypassPermissions`
  (always rejected with 403), `dontAsk`, `auto`.
- `max_turns`: 1–1000. `timeout_seconds` must be > 0;
  `timeout_seconds` may not exceed `AGSRV_MAX_TIMEOUT_SECONDS`.

If a Claude client disconnects, the run is promptly cancelled and its concurrency
slot is freed. The server internally answers 499 without logging an error; the
response is not delivered to the disconnected client.

Response:

```json
{
  "agent": "claude",
  "result": "There are 3 Python files.",
  "session_id": "0b0e...",
  "model": "...",
  "subtype": "success",
  "is_error": false,
  "num_turns": 2,
  "duration_ms": 4210,
  "duration_api_ms": 3900,
  "total_cost_usd": 0.0123,
  "usage": { "input_tokens": 100, "output_tokens": 40 },
  "stop_reason": "end_turn",
  "messages": [
    { "type": "system", "subtype": "init", "data": {} },
    {
      "type": "assistant",
      "model": "...",
      "content": [
        { "type": "thinking", "thinking": "...", "signature": "..." },
        { "type": "tool_use", "id": "t1", "name": "Glob", "input": { "pattern": "*.py" } }
      ],
      "error": null,
      "usage": null,
      "stop_reason": null,
      "session_id": null,
      "uuid": null
    },
    {
      "type": "user",
      "content": [
        { "type": "tool_result", "tool_use_id": "t1", "content": "a.py", "is_error": false }
      ],
      "uuid": null,
      "parent_tool_use_id": null
    },
    {
      "type": "result",
      "subtype": "success",
      "is_error": false,
      "result": "...",
      "session_id": "0b0e...",
      "num_turns": 2,
      "duration_ms": 4210,
      "duration_api_ms": 3900,
      "total_cost_usd": 0.0123,
      "usage": null,
      "stop_reason": "end_turn",
      "errors": null
    }
  ]
}
```

The top-level `model` is the model reported by the run's assistant messages (the last one wins) and is `null` when the run produced no assistant message, so clients can verify the model that actually ran without parsing the transcript.

An agent-level failure (`is_error: true`, e.g. `subtype: "error_max_turns"`) is
still HTTP `200`. Message types not listed above are emitted as
`{"type": "<snake_case_class>", "data": {...}}`.

### Codex requests

```json
{ "agent": "codex", "prompt": "Summarize this project", "cwd": ".",
  "model": null, "effort": "high", "timeout_seconds": 300,
  "include_messages": false,
  "options": { "sandbox": "read-only", "network_access": false } }
```

`options.sandbox` is `read-only` (default) or `workspace-write`;
`network_access` defaults to `false`. `cwd` is confined to the same workspace
root as Claude. `include_messages` is accepted for consistency; Codex returns
a turn summary rather than a transcript. The response contains `agent`,
`result`, `model`, `is_error`, `duration_ms`, `usage`, and `backend` with the
turn ID, status and executed commands. The model comes from thread metadata,
or is null when unavailable.

Codex threads use `ephemeral=true` (D1) and approval policy `never`, with an
additional handler that declines all approval requests. Rate limits use the
typed `account/rateLimits/read` request (D2). For network access the adapter
uses an explicit low-level sandbox policy; otherwise it uses SDK presets (D3).
Set `AGSRV_CODEX_ENABLED` (default `true`), `AGSRV_CODEX_BIN` (optional SDK
binary override), and `AGSRV_CODEX_MAX_CONCURRENCY` (default `2`). A disabled
or unavailable backend returns 503 without affecting Claude.

### Muse requests

```json
{ "agent": "muse", "prompt": "Summarize this project", "cwd": ".",
  "model": null, "effort": "high", "timeout_seconds": 300,
  "include_messages": false, "sandbox": true }
```

`effort` is one of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`,
`ultra`. Muse uses `muse serve --no-session-log --contributor-mode off`, a
deny-all `on_approval` handler and approval mode `onRequest` (never
`--trust-workspace`).

`sandbox` (boolean, default `true`) applies to both `POST /v1/query` and
`POST /v1/jobs`. An explicit opt-out is the only way to disable the sandbox:
that turn's `muse serve` host is spawned with `--disable-sandbox` and the
sandbox availability check is skipped for that turn. Omitting the field or
sending `true` keeps the sandboxed posture and the fail-closed availability
check. `false` (or any value Pydantic's boolean parsing accepts as false, such
as `"false"`, `"0"`, `"no"`, `"off"` or `0`) opts out for that turn; other
non-boolean values, including `null` and strings such as `"maybe"`, are
rejected with 422. The opt-out is per request; there is no server-wide
setting or separate control endpoint, and the health probe and
`GET /v1/agents/muse/rate-limits` always use the sandboxed posture. The
effective value is recorded as `backend.sandbox` in the response. Sessions are
memory-only; if `client.durability` is not ephemeral the backend fails closed
and `GET /v1/agents` reports unhealthy. `include_messages` is accepted for
parity but no transcript is returned in v1. The backend is disabled by default
(`AGSRV_MUSE_ENABLED=false`) and requires a logged-in `muse` host matching the
SDK version (`muse-code-sdk==1.4.24` with `muse 1.4.24`); upgrade them
together. There is no streaming variant for any agent; all queries are
request/response.

## `GET /v1/agents`

Returns `claude`, `codex` and `muse` objects with `enabled`, `healthy`, `default_model`,
and `supports` flags for `sandbox`, `network_access`, `rate_limits`, and
`streaming`. For Muse, `supports.sandbox` means the backend honours the
per-request `sandbox` field (sandboxed by default, opt-out with `false`); it
does not mean every turn is sandboxed. Claude and Codex health probes are cached for approximately 30 seconds. Muse initialization follows the same cached (~30s) probe pattern, checking ephemeral durability and sandbox availability. Claude uses the local `/usage` command, without consuming tokens or allowance.

Claude is `healthy` when the CLI is spawnable, the ephemeral `/usage` run does not
return an error result, and the bundled CLI's `auth status` reports
`loggedIn: true`. The probe never uses the sandbox. This proves a configured,
spawnable CLI whose `/usage` run succeeds; it does not prove the credential is
still valid server-side (an expired or revoked token can pass until a real turn
runs). While the Claude concurrency slot is busy, this endpoint returns the last
cached health value (even if expired) and never responds 429 because of the
Claude probe. The Codex entry is unaffected: with Codex enabled and its
concurrency slot busy and its cached probe has expired, this endpoint can
still respond 429. The same applies to the Muse entry when Muse is enabled,
its slot is busy, and its cached probe has expired.

## `GET /v1/agents/claude/rate-limits`

Runs the local Claude CLI's `/usage` command and returns the normalized
rate-limit windows together with the raw text. If the output cannot be
parsed, `available` is false. The `/usage` command consumes no tokens or
subscription allowance. The route spawns a Claude child process, so it
receives the same admission, disk-space preflight and per-turn `TMPDIR`
handling as a query — unlike the Codex route below, which starts no
workspace turn.

## `GET /v1/agents/muse/rate-limits`

Actively refreshes subscription usage via the experimental Muse host
`usage/refresh` API without a model call or a session. `five_hour` and `seven_day` contain `used_percent` and an ISO 8601
`resets_at`, or null when unavailable. `raw` and `backend` preserve the SDK
payload. Results are cached for 60 seconds, so repeated polls can return the same
windows and `fetched_at` even if account usage changes; the timestamp is the read
time of the cached observation. If the refresh fails or has no usable windows, `available` is false. Disabled or unavailable Muse returns 503;
backend failures return 502 and timeouts return 504.

## `GET /v1/agents/codex/rate-limits`

Returns `plan`, `primary` and `secondary` windows (`used_percent` and
`resets_at`), `reached_type`, `reset_credits`, and the raw SDK payload in
`backend`. Missing windows are null.

## Durable asynchronous jobs

`POST /v1/jobs` accepts the same discriminated Claude/Codex/Muse JSON body as
`/v1/query`. Unlike synchronous queries, the server owns execution after the
202 response and continues across HTTP disconnects. Example:

```http
POST /v1/jobs
Authorization: Bearer <AGSRV_API_KEY>
Idempotency-Key: vm-task-123
Content-Type: application/json

{"agent":"claude","prompt":"Analyze the repository","cwd":"myproject"}
```

Returns 202: `{"job_id":"<opaque UUID>","status_url":"/v1/jobs/<id>"}`.
The optional `Idempotency-Key` (1–128 printable ASCII chars) is scoped to the
bearer principal. Identical validated requests reuse the job ID even on retries
after a VM is destroyed. A conflicting payload returns 409; once the terminal
record expires, key reuse can create a new job.

`GET /v1/jobs/{job_id}` returns `job_id`, `agent`, `status`,
`created_at`, `started_at`, `finished_at`, `result`, and `error`.
Statuses are `queued`, `running`, `succeeded`, `failed`, `cancelled`,
or `interrupted`. `result` contains the normalized Claude, Codex or Muse response
only on completion. Agent-level errors become `failed` with a retained result;
transport failures and deadlines have sanitized `error.code` values.
`POST /v1/jobs/{job_id}/cancel` is idempotent: queued jobs stop immediately,
running jobs remain nonterminal until their child processes are cleaned up.

The job database is SQLite WAL; it must be on durable storage accessible to a
**single uvicorn worker**. An exclusive lock file alongside the database
prevents concurrent workers or overlapping instances from taking ownership. The default path is
`~/.local/state/agsrv/jobs.sqlite3` (overridden by `AGSRV_JOBS_DB_PATH`).
Nonterminal jobs are limited by `AGSRV_JOBS_MAX_PENDING=32`, canonical
requests by `AGSRV_JOBS_MAX_PAYLOAD_BYTES=65536`, stored results by
`AGSRV_JOBS_MAX_RESULT_BYTES=1048576`; saturation gives 429 and
`Retry-After: 5`. Terminal jobs and idempotency keys are retained for
`AGSRV_JOBS_RETENTION_SECONDS=86400` (24 hours). Garbage collection is
performed during API access. Synchronous queries and jobs targeting overlapping
workspaces cannot execute concurrently. When a result exceeds the configured
cap, message history is omitted with `messages_truncated: true` while keeping
the final answer. If the core result is still oversized, a minimal summary is
stored with `result_truncated: true` so the original execution outcome is not
misreported. Extremely small caps that cannot fit this summary result in
`failed` / `result_too_large`. The bearer token is the caller identity: **all clients sharing
the same API key can read the same jobs**; use separate server credentials and
instances if isolation between clients is needed. No raw internal errors or
credentials are included in job status.

The in-process scheduler **does not guarantee completion across a server
restart**. On restart both queued and running jobs become `interrupted`
without replaying side effects; completed results remain until expiry.
The old SSE stream endpoint is deliberately removed.

## Errors

| Status | Meaning                                                                |
| ------ | ---------------------------------------------------------------------- |
| 400    | `cwd` missing or not a directory; `timeout_seconds` above server max   |
| 401    | Missing or incorrect bearer token                                      |
| 403    | `cwd` outside workspace root; `bypassPermissions` always forbidden; with an egress-expanding permission mode (`auto`, `bypassPermissions`), or with a `cwd` overlapping the shared cache (`~/.cache` outside `~/.cache/agsrv-sandbox`, or an ancestor such as the home directory) |
| 409    | Idempotency-Key reused with a different job payload                  |
| 413    | Durable job payload too large                                         |
| 404    | Job not found or not authorized for this caller                       |
| 422    | Invalid body (empty prompt, unknown/stateful fields, bad ranges)       |
| 429    | Concurrency or pending-job capacity reached (jobs include Retry-After) |
| 502    | SDK/CLI failure, or the run ended without a result message             |
| 503    | `AGSRV_API_KEY` not configured and unauthenticated access not allowed; sandbox prerequisites unavailable; Claude, Codex or Muse disabled/unavailable; or the turn could not be prepared — the disk-space preflight found free space on `/tmp`, the workspace root or the working directory below its threshold; no staging directory is configured for the working directory's mount; the working directory failed acceptance revalidation (a missing component, a symlink, or a substituted directory); or the per-turn directory could not be created (for example an owner-clearing `umask`, or `EXDEV`) |
| 504    | Run exceeded its timeout                                               |

Errors use FastAPI's `{"detail": "..."}` body.
