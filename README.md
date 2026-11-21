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
| GET    | `/v1/agents/claude/rate-limits` | yes | Claude subscription quota |
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
| `AGSRV_TMP_MIN_FREE_BYTES`      | `536870912`                      | Minimum free bytes on /tmp; 0 disables the check |
| `AGSRV_WORKSPACE_MIN_FREE_BYTES`| `1073741824`                     | Minimum free bytes on the workspace filesystem; 0 disables the check |
| `AGSRV_WORKSPACE_ROOT`           | current directory                | All request `cwd` values must resolve inside this path   |
| `AGSRV_TMP_STAGING_DIRS`         | unset (turns unavailable)        | `mount_root=staging_dir` pairs, comma-separated; see below |
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

Each turn uses a private `.agsrv-tmp-<random>` directory in its working
directory. Claude receives it as `TMPDIR` in the SDK environment. For Codex, the
per-turn thread configuration carries
`shell_environment_policy.set.TMPDIR`; this is verified at the request-payload
level only (a mocked `thread_start` test). A live Codex child observing the value
has **not** been verified, because bubblewrap's socket-directory constraint
blocks that observation. Codex read-only sandboxes still enforce their normal
write restrictions. Before a turn, agsrv checks free space on `/tmp`, the
workspace root and the working directory (measured on the held descriptors, so
a working directory on a separate mount must pass its own check) and returns
HTTP 503 when either enabled threshold is unmet. Queued work is checked again,
after its last wait, immediately before creation. The Claude rate-limits route
spawns a Claude child process and gets the same admission, preflight and
`TMPDIR` handling; the Codex rate-limits route starts no workspace turn.

### Temporary directory design and threat model

**Required deployment property.** Turns need one pre-provisioned,
service-controlled *staging* directory per workspace mount, configured with
`AGSRV_TMP_STAGING_DIRS` (comma-separated `mount_root=staging_dir` pairs; the most
specific mount root containing the working directory wins). Each staging
directory must be owned by the service, mode `0700`, outside the workspace root,
on the same filesystem as its mount root, and unreachable by workspace writers.
They are opened and validated at startup (startup fails on an unsafe one) and the
descriptors are held for the life of the server. A mount without a staging
directory runs no turns (HTTP 503). A local filesystem with atomic rename and
`RENAME_NOREPLACE` support is required; `EXDEV`, unsupported flags or a missing
`renameat2` fail closed. Example: `AGSRV_WORKSPACE_ROOT=/srv/ws`
`AGSRV_TMP_STAGING_DIRS=/srv/ws=/var/lib/agsrv/stage`.

Other writers may share the working directory and can list it, rename or remove
entries, substitute real directories or symlinks, and change ancestors. Random
names and the workspace lease do not establish ownership of an entry, so
ownership is decided by an open descriptor, never by a name. The protected object
is the top-level tree agsrv created for the turn (its contents are disposable);
a different entry substituted for it must survive, even if empty or owned by the
service user.

- **Admission.** The accepted working directory is opened once by walking from
  the workspace-root descriptor, one component at a time, with
  `O_DIRECTORY | O_NOFOLLOW`, and that descriptor is held by the request or job
  (including through the job queue and detached Codex execution). After the last
  wait the walk is repeated and compared with `os.path.samestat`; a missing
  component, symlink or different directory fails the turn (HTTP 503) before
  anything is published or launched. All later server mutations use the held
  descriptor, so a later symlink swap cannot redirect them.
- **Create privately, publish atomically.** The directory is created in staging
  (`mkdir` under a random name, `chmod 0700` while still private, opened and
  verified through its descriptor) and then published into the working directory
  with `renameat2(RENAME_NOREPLACE)`, which never overwrites an existing entry,
  even an empty directory. `O_CREAT | O_EXCL | O_DIRECTORY` is not used (it fails
  with `EINVAL` on Linux 6.4 and later).
- **Reclaim before delete.** Cleanup renames the public entry from the held
  working-directory descriptor back into staging (no-replace), checks it against
  the held descriptor, and only on a match runs `shutil.rmtree(name, dir_fd=...)`
  (symlink-attack resistant; required at startup). All deletion, including the
  final `rmdir`, happens under the protected staging parent. On a mismatch
  nothing is traversed or deleted: the entry is renamed back if the public name
  is free; otherwise it stays in staging and its recovery name is logged.
- **Off the event loop, shielded.** Cleanup runs in one worker thread. If the
  owning task is cancelled (repeatedly, too) the same worker is awaited to the
  end, and only then is cancellation propagated and the workspace lease released.
  At shutdown turn owners are drained before the shared descriptors are closed.
- **Rollback.** A failure after the private `mkdir` removes the staged directory
  by its private name; a failure after publication runs the full
  reclaim-and-verify sequence. The original error, result or cancellation is
  preserved; uncertain or partly removed entries are retained and logged.

Residual risks:

- Children receive a `TMPDIR` path *string* and resolve it (and their cwd) in
  their own namespace at the time of use. A writer can substitute the leaf or an
  ancestor, so a child can fail, write into a replacement, or follow a symlink if
  its sandbox permits. agsrv's own operations stay on the held descriptors and
  cannot be redirected, but give the child's path lookup no identity guarantee.
- A descriptor identifies its object even if an ancestor is moved; continuous
  containment against reparenting needs additional namespace or permission
  constraints.
- Turn contents are disposable: anyone able to put data *inside* the genuine turn
  tree puts it in cleanup scope. A writer that repeatedly moves entries, a
  surviving writer that keeps modifying contents, filesystem errors or process
  death can prevent reclamation. A crash can leave a public or staged entry;
  there is deliberately no name-based scavenger. Subdirectories made
  non-listable by a child may keep a partially removed entry in staging (logged).
- `0700` is also a deployment property: an unrestricted same-UID child that can
  reach staging (for example through the service's `/proc` descriptors) defeats
  it, and such a deployment needs credential or namespace isolation. Mount
  administration, service compromise, malicious filesystems and the trusted
  anchors (workspace root, staging directory and their ancestors) are outside
  the model.
- A live Codex child observing `TMPDIR` is unverified, as noted above.

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
Rate limits come from `account/rateLimits/read`. The Claude rate-limits route
runs the local `/usage` command and returns normalized windows and the raw text.
If the output cannot be parsed, it returns `available=false`.

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
use a given SQLite database; the server rejects concurrent owners via a lock
file alongside the database. Synchronous queries and jobs targeting overlapping
workspace directories are serialized. Oversized transcripts are omitted with
`messages_truncated: true` while retaining the final answer. If even the core
result exceeds the size limit, `result_truncated: true` preserves a minimal
outcome summary. Shared API keys represent one caller and authorize access to the
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
