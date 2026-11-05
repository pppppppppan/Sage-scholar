# Troubleshooting

- **Docker cannot connect to its socket:** start Docker Desktop/Engine first. `docker compose config` is only a configuration check and does not prove images build or containers run.
- **Backend waits for Ollama:** inspect `docker compose logs backend`; check model name and reachable bind address. Container `localhost` is not the host. Use the deployment guide's host/profile instructions.
- **Browser cannot resolve `backend`:** Docker DNS belongs on the Next.js server. The shipped image compiles an empty `NEXT_PUBLIC_API_BASE`, then proxies `/api` to runtime `BACKEND_INTERNAL_URL`. Rebuild if you changed a public build-time variable.
- **SSE appears delayed:** use `curl -N` and inspect response headers. The Next route streams without collecting the response body. Any additional reverse proxy must also disable SSE buffering; disconnection is not Run cancellation.
- **PDF embedding/model unavailable:** explicitly prepare weights into the cache volume. Local-only settings intentionally prevent a hidden first-use download. Confirm model identity before rebuilding an index.
- **PDF visual parsing unavailable:** build optional visual dependencies, prepare visual weights, and select `page` or prepare Docling assets. Linux containers on macOS cannot use Apple's MPS acceleration.
- **Search/fetch TLS or proxy failure:** the container config inherits reachable HTTP(S) proxy settings. A host proxy bound only to 127.0.0.1 may need a reachable interface; do not copy that loopback URL into a container.
- **Langfuse returns 401:** check the project's region and key pair. Use `.env.docker` for containers; changing the host `.env` does not change container credentials.
- **Data looks empty after switching to Docker:** named volumes are separate from the host development `data/`. Existing catalog paths may be absolute; copying the directory to a different root is not a supported migration.
- **Port 3000 occupied:** stop the old development UI or set `SAGE_WEB_PORT=3001 docker compose up -d`. Backend port 8000 is not published by default.
- **Image rebuild changes dependencies:** core versions and npm trees are pinned, but Python transitive dependencies and OS/base-image tag contents are not a complete immutable lock. Preserve the built image/digest for repeatable deployment.

See [Docker deployment](deployment/docker.md) for exact commands and persistence safeguards.
