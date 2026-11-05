# Deployment validation — 2026-11-05

## Executed

| Check | Actual result |
|---|---|
| `docker compose config --quiet` | Passed |
| `docker compose --profile ollama config --quiet` | Passed |
| Backend full pytest suite | **425 passed, 5 failed**; same MCP `ps` sandbox PermissionError failures as before this change |
| Frontend Node tests | **17 passed**, including four new streaming/proxy tests |
| Frontend lint | Passed |
| Next.js production build | Passed with Node **24.21.0**, Next.js **16.3.6**, empty public API base and standalone output |
| Production HTTP smoke, outside Docker | Passed: real FastAPI/Uvicorn + real Next standalone server, same-origin health, conversation creation, Direct date answer, SSE event replay and Last-Event-ID |
| Backend process restart, outside Docker | Same Run/conversation recovered from isolated temporary data; checkpoint SQLite present; no changes to the real research library |
| Host Ollama | `/api/tags` reachable and `qwen3.5:2b` present; no model generation needed |
| Telemetry off | Production smoke ran with `OBSERVABILITY_ENABLED=false`; no trace export or paid model call |

The new backend tests exercise explicit Router host permission and readiness model validation. Frontend tests exercise streaming before response completion, cancellation propagation, multipart upload forwarding, error/status handling and same-origin redirects. Existing Run recovery/HITL tests were included in the full suite. No failing assertions were deleted, skipped or weakened.

The production smoke used a trusted `workflow=direct` override for the date query so it did not require paid answering inference; it is not a fresh semantic Router accuracy evaluation. The SSE smoke replayed a completed real Run; continuous streaming/cancellation were covered separately by the proxy tests. The temporary servers were stopped after verification.

## Not verified

Docker CLI **29.5.3** and Compose **5.1.4** are installed, but the Docker Engine socket is absent. Consequently:

- Backend/frontend **container image builds were not executed**.
- Container-to-host and container-to-Ollama-profile connectivity were not exercised.
- Named-volume persistence across container recreation was not exercised.
- CPU Torch/Linux wheel installation, container MCP startup and optional visual dependencies need a real image build/smoke check.
- No full PDF ingestion, vision inference, paid Deep Research or Langfuse export was run as part of this deployment task.

A Linux-target Python dependency resolution check completed, but it is not an installed Linux environment or Docker build. Core packages are constrained to the observed working versions; Python transitive packages/base-image tags remain a reproducibility limitation. The local production build and HTTP smoke validate application/proxy behavior, not the unbuilt container layers.

Use the [deployment acceptance commands](docker.md#operations-and-verification) after starting Docker Engine. Existing SQLite, checkpoint and model files should be backed up before any separate data migration; this work did not perform such a migration.

## Files changed

- Deployment: `docker-compose.yml`, `.dockerignore`, `.env.docker.example`, `deploy/backend.Dockerfile`, `deploy/frontend.Dockerfile`, `deploy/backend-constraints.txt`, `deploy/entrypoint.py`, `deploy/prepare_models.py`, `deploy/configs/mcp_servers.json`, and `deploy/mcp/package.json` / `package-lock.json`.
- Configuration boundary only: `sage_scholar/config.py`, `sage_scholar/routing/classifier.py` (explicit trusted Ollama hosts; prompt/classification unchanged).
- Frontend deployment boundary: `web/next.config.ts`, `web/lib/api.ts`, `web/lib/backend-proxy.ts`, `web/app/api/[...path]/route.ts`.
- Verification: `tests/test_deployment.py`, `web/tests/backend-proxy.test.mjs`, `.github/workflows/quality.yml` (Compose config and frontend checks, no paid inference).
- Documentation: `README.md` reduced from 494 to 115 lines; `.gitignore` permits only the intended documentation/example files. `docs/implementation-notes.md` preserves the old homepage. New local-development/configuration/architecture/RAG/Deep-Research/Router-evaluation/observability/FAQ guides and this deployment folder provide current entry points. Existing `docs/repository-audit.md` and `docs/migration-map.md` are now included instead of linking to ignored files.
- Local, ignored setup: `.env.docker` was created from the blank example; host `.env` and research data were not modified.

Additional dependency checks: the CPU Torch index lists CPython 3.12 wheels for both Linux x86_64 and aarch64 at the pinned version. Fetch, pdfmux and paper-search pinned dependency sets resolve for the Linux target. The resolver used the available Python 3.13 interpreter for any build metadata because local 3.12 is absent; these checks are not a substitute for running the Python 3.12 container image.
