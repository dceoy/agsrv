# Deployment

## Local development

```sh
uv sync
AGSRV_ALLOW_UNAUTHENTICATED=true uv run uvicorn agsrv.main:app --host 127.0.0.1 --port 8000
```

## Remote server

1. Install dependencies on the server:

   ```sh
   git clone https://github.com/dceoy/agsrv.git /opt/agsrv
   cd /opt/agsrv && uv sync --no-dev
   ```

2. Authenticate Claude on the server. The SDK spawns the Claude Code CLI, so
   either set `ANTHROPIC_API_KEY` in the environment file below, or log in once
   with `claude` as the user that runs the service.

3. Create `/etc/agsrv.env` (mode `0600`):

   ```sh
   AGSRV_API_KEY=<long random secret, e.g. openssl rand -hex 32>
   AGSRV_WORKSPACE_ROOT=/srv/workspaces
   ANTHROPIC_API_KEY=sk-ant-...
   ```

4. Run uvicorn on loopback or a private interface, and expose it only through
   TLS (a reverse proxy) or a private network such as Tailscale or an SSH
   tunnel (`ssh -L 8000:127.0.0.1:8000 server`).

5. Run it as an unprivileged user that owns only the workspace directory.
   Workspace confinement limits the starting `cwd`, not what `Bash` can reach.
   The agent inherits this environment file, so a run can read
   `AGSRV_API_KEY` and `ANTHROPIC_API_KEY` from its own environment — the
   API key is effectively a shell as this user (see the README security
   model). The concurrency limit is per process: running uvicorn with
   `--workers N` multiplies `AGSRV_MAX_CONCURRENCY` by N. The durable job
   queue currently requires **exactly one worker** for a database. Store
   `AGSRV_JOBS_DB_PATH` on persistent storage with restricted permissions.

### systemd unit

```ini
[Unit]
Description=agsrv
After=network-online.target

[Service]
User=agsrv
WorkingDirectory=/opt/agsrv
EnvironmentFile=/etc/agsrv.env
ExecStart=/opt/agsrv/.venv/bin/uvicorn agsrv.main:app --host 127.0.0.1 --port 8000
Restart=on-failure
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

### Bash sandbox (enabled by default)

Claude runs use the fail-closed CLI Bash sandbox by default. Explicit
`sandbox: false` is reserved for host-control tasks that must bypass it.

- Prerequisites on the host: `bwrap` (bubblewrap), `socat`, and `rg` (ripgrep).
- Sandboxed runs are pinned to the CLI bundled with the installed
  `claude-agent-sdk` (`cli_path`), never any `claude` on `PATH`; if the bundled
  executable is missing or not executable the request fails (503). Its version decides how sandbox settings are
  interpreted, so redo the probe below after every SDK update.
- Sandboxed requests are rejected (403) when the effective permission mode is
  `auto` or `bypassPermissions`, and when the working directory overlaps
  `~/.cache` (for example a workspace root equal to the service home with no
  `cwd`). Keep `AGSRV_WORKSPACE_ROOT` away from the service home, or ensure
  clients send a `cwd` below a project directory.
- `~/.cache/agsrv-sandbox` is created with mode 0700 on first use; it must be
  a real directory owned by the service user (symlinks are rejected). Its mode
  is set to and verified as exactly 0700, and `~/.cache` itself must not be
  group- or other-writable; otherwise requests fail with 503.
- The sandbox is fail-closed. On Ubuntu releases that ship the
  `bwrap-userns-restrict` AppArmor profile, the sandbox's seccomp helper cannot
  start, so `sandbox: true` requests fail until the host policy is adjusted
  (for example with a dedicated, AppArmor-profiled copy of `bwrap` placed first
  on the service `PATH`). Do not weaken this by disabling the sandbox checks.

Before deployment, run an acceptance probe with the default (sandboxed) run:

1. Verify `AGSRV_SANDBOX_ALLOWED_DOMAINS` permits only required hosts.
2. Run a request without a `sandbox` field: `echo ok` succeeds; writes
   outside the working directory and the sandbox cache are denied; secret
   directories such as `~/.ssh` are unreadable; `env` shows no
   `AGSRV_API_KEY`; `curl` to a non-allow-listed host is denied; and
   `systemctl --user status` fails.

Clients must never retry a failed default sandbox request with
`sandbox: false`; that would reintroduce fail-open behaviour.

### Durable jobs

Use the same long-lived API key on replacement Muse VMs and submit jobs with a
stable `Idempotency-Key`. Poll `GET /v1/jobs/{job_id}` without holding open a
connection. Back up the SQLite WAL database (including its WAL file) and avoid
multi-worker uvicorn: an exclusive sibling lock file rejects concurrent
processes sharing the database. Synchronous queries share the same workspace
lease as queued jobs. Jobs are in-process: a server restart marks uncompleted
jobs `interrupted` and never retries them automatically. Default completed-job
retention is 24 hours. All holders of the same bearer token can inspect jobs
created with that token; do not share credentials across untrusted tenants.

### Client smoke test

From the client machine (here via an SSH tunnel on `localhost:8000`):

```sh
curl -sS http://localhost:8000/health
curl -sS http://localhost:8000/v1/info -H "Authorization: Bearer $AGSRV_API_KEY"
curl -sS http://localhost:8000/v1/query \
  -H "Authorization: Bearer $AGSRV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"agent": "claude", "prompt": "Reply with the single word: pong", "max_turns": 1}'
```

A `200` response with `"result": "pong"` confirms authentication, Claude
credentials, and the SDK/CLI path all work.
