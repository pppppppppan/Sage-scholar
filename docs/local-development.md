# Local development

The Docker additions do not change `server.py`, `setup.sh` or the host MCP configuration.

Use Python 3.12 (project allows 3.11–3.14), Node 24.21.0 (`.nvmrc`), and Ollama. In the repository root:

```bash
conda create -n sage-scholar python=3.12 -y
conda activate sage-scholar
nvm install
nvm use
cp .env.example .env  # Only on first setup; do not overwrite existing secrets.
ollama pull qwen3.5:2b
# Configure .env, then:
./setup.sh
```

The script installs Python/frontend dependencies, checks ports 8000/3000, and starts both development servers. Ctrl+C stops its child processes. It does not install Ollama or all optional MCP/visual packages.

To run separately after installation:

```bash
python server.py                   # Uvicorn with reload, development only
npm --prefix web run dev           # Separate terminal
```

For an unproxied local browser, the API defaults to `http://localhost:8000`. Docker builds set `NEXT_PUBLIC_API_BASE` to an empty string for same-origin `/api`; `BACKEND_INTERNAL_URL` is server-only. Next public variables are build-time values.

Optional integrations:

```bash
python -m pip install -r requirements-observability.txt
python -m pip install -r requirements-visual.txt  # Only for PDF visual retrieval
```

`configs/mcp_servers.json` is the host configuration: `fetch` uses uvx, Tavily/pdfmux use npx, paper-search needs `paper-search-mcp`, and GitHub uses a host Docker executable. Review proxy addresses and credentials before running them. Docker uses its own configuration and installed programs instead.

```bash
python -m pytest -q
python -m compileall -q sage_scholar tests
npm --prefix web test
npm --prefix web run lint
npm --prefix web run build
```

See [configuration](configuration.md), [Docker deployment](deployment/docker.md), and [historical notes](implementation-notes.md) for detailed subsystem commands.
