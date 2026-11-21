# API Reference

agsrv exposes a stateless REST API around the Claude, Codex and Muse SDKs:

- `GET /health`: readiness probe (no auth).
- `GET /v1/info`: server defaults and limits.
- `GET /v1/agents`: agent availability and capabilities.
- `GET /v1/agents/claude/rate-limits`: Claude subscription quota.
- `GET /v1/agents/codex/rate-limits`: Codex account limits.
- `GET /v1/agents/muse/rate-limits`: Muse last-observed subscription usage.
- `POST /v1/query`: one agent run, JSON result with an optional Claude transcript.
- `POST /v1/jobs`: submit a durable asynchronous job (202).
- `GET /v1/jobs/{job_id}`: poll a job's status and terminal result.
- `POST /v1/jobs/{job_id}/cancel`: request cancellation.

- [REST API](rest-api.md)
- [Deployment](deployment.md)
