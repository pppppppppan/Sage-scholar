# Docker Compose deployment

## What runs

| Service | Runtime | Network |
|---|---|---|
| backend | Python 3.12, single Uvicorn worker; no reload | `backend:8000`, internal only |
| frontend | Node 24 / Next.js standalone production server | Published on `127.0.0.1:3000` by default |
| ollama | Optional `ollama` profile, CPU default | `ollama:11434`, internal only |

The backend runs the existing FastAPI application. Next.js streams browser `/api/*` requests to `http://backend:8000` at runtime, including PDF uploads, SSE and cursor headers. Internal Docker addresses and provider credentials are not embedded in browser JavaScript. `NEXT_PUBLIC_API_BASE` is compiled as an empty string; changing it requires rebuilding. Existing host development keeps its localhost API default.

Healthchecks verify backend HTTP readiness and frontend-to-backend reachability. Backend startup also waits up to `OLLAMA_WAIT_SECONDS` for `/api/tags` to contain the configured Router model; this does not run model inference or prove all provider keys/tools work. Missing Ollama yields a clear startup failure and `unless-stopped` restart; it is not silently replaced with cloud classification. There is exactly one backend worker. Do not scale it with this deployment.

## First setup: existing host Ollama

1. Start Docker Desktop/Engine. Verify `docker info` and `docker compose version`.
2. Prepare the host Router: `ollama pull qwen3.5:2b` (an already downloaded model is reused).
3. Make the host Ollama listener reachable from Docker. For a manually launched Ollama, quit any existing instance first, then run:

   ```bash
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```

   For the macOS app, `launchctl setenv OLLAMA_HOST "0.0.0.0:11434"`, then quit/reopen Ollama. This can expose an unauthenticated model endpoint to the LAN; restrict access with the host firewall. Docker Desktop's `host.docker.internal` and Linux's Compose `host-gateway` mapping target the host, but they cannot make an unreachable listener reachable. Keep the default host Router on native macOS if you want MPS/Metal acceleration.
4. In the repository root:

   ```bash
   cp .env.docker.example .env.docker  # First setup only; keep existing .env untouched.
   # Edit .env.docker: LLM_MODEL_ID + provider key/base URL; Tavily key if used.
   # Default Router URL: http://host.docker.internal:11434
   docker compose config --quiet
   docker compose build
   docker compose run --rm --no-deps backend python -m deploy.prepare_models
   docker compose up -d --build
   ```

Open http://localhost:3000. Model preparation is explicit and potentially large; the final `up` command assumes it and credentials are ready. Preparation downloads into the persistent model cache, resumes via Hugging Face's cache, and does not invoke any paid LLM. It prepares text embedding/reranker, not Ollama. Optional visual models are separate below.

The base service can answer without loading RAG models; document ingestion and QA require the embedding. `EMBEDDING_LOCAL_ONLY=true` and `RERANKER_LOCAL_ONLY=true` make missing caches visible. The reranker can fall back to RRF, with reduced/changed retrieval quality, while a missing embedding prevents indexing.

## Optional container Ollama

This is an alternative to host Ollama, not an additional mandatory dependency. No GPU is requested; CPU inference is slower and macOS Docker cannot use Metal/MPS.

```bash
# In .env.docker set ROUTER_BASE_URL=http://ollama:11434
# ROUTER_ALLOWED_HOSTS already permits exactly ollama and host.docker.internal.
docker compose --profile ollama up -d ollama
docker compose exec ollama ollama pull qwen3.5:2b
docker compose --profile ollama up -d --build
```

The model is stored in `ollama-models`. Host Ollama's cache is not automatically shared into that volume. Prefer the host path to avoid downloading again. `ollama list` healthchecks confirm the service is alive; the backend startup check confirms the required model exists. The backend readiness loop handles either host/profile startup without a mandatory dependency on a disabled profile.

## Model cache and optional vision

`model-cache` persists all Hugging Face/XDG caches at `/home/sage/.cache`. The image contains model libraries, not weights. CPU is the portable default; allocating more Docker memory improves reliability, but model sizes and document lengths still limit throughput. GPU execution requires an operator-specific NVIDIA runtime/device configuration and is not included here.

To reuse an existing Hugging Face cache instead of downloading again, create a local override:

```yaml
# docker-compose.override.yml (local machine-specific configuration)
services:
  backend:
    volumes:
      - type: bind
        source: /absolute/path/to/your/.cache/huggingface
        target: /home/sage/.cache/huggingface
```

The cache must contain the model IDs/revisions in `.env.docker`, including usable snapshot symlinks and blobs, and be writable by container UID 10001 (Docker Desktop sharing rules differ from native Linux). Mount an exact local model directory and set `EMBEDDING_LOCAL_MODEL`/`RERANKER_MODEL` if you used a standalone model folder. Do not point containers at host-only `/Users/...` paths without mounting them.

Visual retrieval is optional and defaults off:

```bash
# Edit .env.docker: PDF_VISUAL_ENABLED=true, PDF_VISUAL_PARSER=page
SAGE_DOCKER_VISUAL=true docker compose build backend
# Downloads the visual embedding as well as normal RAG models:
docker compose run --rm --no-deps backend python -m deploy.prepare_models --visual
SAGE_DOCKER_VISUAL=true docker compose up -d --build
```

The `SAGE_DOCKER_VISUAL` build argument comes from the shell, not service `env_file`. Alternatively place it in `.env.docker` and use `docker compose --env-file .env.docker ...` consistently. `page` uses PyMuPDF page renders; `docling` additionally loads layout/OCR models and can require first-use network access and more memory. Docling caches share the persisted XDG/HF location. A vision-capable reading provider is still required to interpret original images; embedding images is not a replacement for that model.

## Persistent data

Compose project name is fixed to `sage-scholar` so the default volume names are stable.

| Volume | Container path | Contents |
|---|---|---|
| `sage-data` | `/app/data` | `runtime.sqlite3` (Runs, events, conversations, journals), `checkpoints.sqlite3`, `memory.json`, `memory_vectors.json`, memory jobs, catalog `index.json`, versioned PDFs/converted Markdown/structure, Evidence, RAG SQLite/FAISS artifacts and visual indexes |
| `model-cache` | `/home/sage/.cache` | Hugging Face snapshots/blobs and other model/XDG caches |
| `sage-logs` | `/app/logs` | Existing application logs |
| `ollama-models` | `/root/.ollama` in optional Ollama | Ollama model blobs/manifests |

All existing application data formats and relative layout are retained. Temporary tool downloads are under `/app/data/tmp` via `TMPDIR`; the existing tool lifecycle may clean these disposable copies. Uploaded/committed original PDFs are stored in document versions, not solely that temporary directory.

`docker compose down` / rebuild / recreate keep named volumes. **`down -v` deletes them.** To back up SQLite consistently, stop backend writes first, then archive the entire data volume (including related files); restore to the same `/app/data` root and use a compatible application/checkpoint version. Do not mount the same data directory into a running host process and container simultaneously.

This creates a separate deployment workspace; it does not automatically import existing macOS `data/`. Existing catalog/visual records can contain absolute paths. Simply copying `/Users/.../data` to `/app/data` does not migrate those references. For new Docker libraries re-upload originals; preserving historical host Runs requires a separately reviewed path-compatible migration. No implicit path rewriting is performed here.

## MCP and network dependencies

Docker uses `deploy/configs/mcp_servers.json`; the host's `configs/mcp_servers.json` stays unchanged. `configs/agents.json` and built-in Skills are copied into the image.

| MCP | Container execution |
|---|---|
| fetch | Pinned `mcp-server-fetch` in `/opt/fetch`, isolated because it requires older httpx |
| paper-search | Pinned `paper-search-mcp` in the app venv; MCP v1 retained |
| pdfmux | Pinned Python `pdfmux[serve]` in `/opt/pdfmux`; launches the same server directly, without npx's first-run pip installation |
| Tavily | Pinned npm package + committed lockfile; Node executes the installed entrypoint with proxy support |
| GitHub / custom host commands | Not enabled in the container config. Host Docker/uvx binaries and stdio servers cannot be reached over HTTP merely by changing an address. |

No Docker socket or host executable paths are mounted. Custom MCP executables need their dependencies installed in a derived image and an explicit config mount. Optional missing keys are skipped by the existing loader; command-level failures still produce tool-unavailable status.

To use a host HTTP proxy, configure `http_proxy`, `https_proxy` and uppercase equivalents in `.env.docker` with `http://host.docker.internal:7897`, and make that proxy reachable from Docker. Compose supplies `NO_PROXY` for backend/frontend/Ollama/local traffic. Container loopback is not the host proxy. Build-time package downloads use Docker's build/proxy configuration separately; do not embed credentials in image layers.

Langfuse is off by default. The image includes OTel packages; configure `OBSERVABILITY_ENABLED=true`, region-correct `LANGFUSE_BASE_URL` and keys in `.env.docker` to export safe metadata. No Langfuse stack is started. See [observability](../observability.md).

## Operations and verification

```bash
docker compose ps
docker compose logs --tail=100 backend frontend
docker compose logs -f backend
curl -f http://localhost:3000/api/health
# Verify the Router address/model from inside the backend without paid inference:
docker compose exec backend python -c 'import os; from deploy.entrypoint import check_ollama; check_ollama(os.environ["ROUTER_BASE_URL"], os.environ["ROUTER_MODEL"]); print("Ollama/model reachable")'
docker compose exec backend python -m sage_scholar.mcp.check --server fetch
# Consumes one Tavily search request, if configured:
docker compose exec backend python -m sage_scholar.mcp.check --server tavily
```

Create a conversation/query in the UI; watch the Network panel for streamed `/api/runs/{id}/events`. To inspect the stream directly, substitute a real Run ID:

```bash
curl -N -H 'Last-Event-ID: 0' http://localhost:3000/api/runs/RUN_ID/events
docker compose restart backend
curl -f http://localhost:3000/api/runs/RUN_ID
```

Verify that the same conversation, Run, document catalog and Memory survive. Clarify should remain waiting; a compatible interrupted checkpoint should resume according to existing recovery rules. A missing/incompatible checkpoint must not claim success. Disconnecting this curl session must not stop the Run. These are business acceptance checks, not performed by `/api/health`.

```bash
docker compose stop                          # Stop while retaining containers/volumes
docker compose start
docker compose down                          # Remove containers/network, retain volumes
SAGE_WEB_PORT=3001 docker compose up -d       # Alternative local UI port
```

`.env.docker` changes require `docker compose up -d --force-recreate backend`, not just `restart`. This application has no multi-user authentication: the default published interface is loopback. External hosting requires a separate access-controlled HTTPS ingress; it is not configured here.

## Reproducibility and validation limits

Node/Python/Ollama base tags, core application package versions and MCP versions are explicit; npm dependency trees use committed lockfiles. Python transitive packages and base-image/OS package content are not fully digest-locked. Preserve built image digests for immutable deployment and review updates explicitly. No local model weights, databases, `.env`, uploads or history are included in the allowlisted build context.

The Docker Engine was unavailable in the implementation environment. See [validation results](validation.md) for checks actually executed; static Compose validation and a local production build are not a claim that container images were built or that container persistence was exercised.

References: [Compose networking](https://docs.docker.com/compose/how-tos/networking/), [profiles](https://docs.docker.com/compose/how-tos/profiles/), [Next.js standalone output](https://nextjs.org/docs/app/api-reference/config/next-config-js/output).
