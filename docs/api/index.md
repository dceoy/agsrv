# API Reference

agsrv exposes a stateless REST API around the Claude and Codex SDKs:

- `GET /health`: readiness probe (no auth).
- `GET /v1/info`: server defaults and limits.
- `GET /v1/agents`: agent availability and capabilities.
- `GET /v1/agents/codex/rate-limits`: Codex account limits.
- `POST /v1/query`: one agent run, JSON result with an optional Claude transcript.
- `POST /v1/query/stream`: one Claude run, streamed as SSE.

- [REST API](rest-api.md)
- [Deployment](deployment.md)
