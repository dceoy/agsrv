# agsrv

Stateless HTTP API for remote Claude, Codex and Muse execution using their
Python SDKs.

## Capabilities

- `POST /v1/query`: run one ephemeral agent query and receive the result as
  JSON, with an optional Claude message transcript.
- `POST /v1/jobs`: durable asynchronous jobs with idempotency keys; poll
  `GET /v1/jobs/{job_id}` and cancel with `POST /v1/jobs/{job_id}/cancel`.
- `GET /v1/info` and `GET /v1/agents` (capabilities and cached health), plus a
  rate-limits route per agent (`/v1/agents/claude/rate-limits`,
  `/v1/agents/codex/rate-limits`, `/v1/agents/muse/rate-limits`).
  `GET /health` needs no authentication.
- Bearer-token authentication, workspace confinement, and per-agent
  concurrency and timeout limits.
- No sessions, resume, or server-side conversation state, and no streaming
  endpoint: every request is one ephemeral run.

## Development

```sh
uv sync
uv run uvicorn agsrv.main:app --host 127.0.0.1 --port 8000
```

- [API reference](api/index.md)
- [Deployment](api/deployment.md)

Licensed under AGPL-3.0-only.
