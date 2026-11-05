"""Container readiness only; the application's workflow and state stay unchanged."""
import os
from pathlib import Path
import time

import httpx


def check_ollama(url, model):
    # Same URL allowlist as the application; no credentials or query data are sent.
    from sage_scholar.config import RouterConfig
    from sage_scholar.routing.classifier import LocalClassifier
    LocalClassifier(RouterConfig(base_url=url, model=model))
    with httpx.Client(timeout=3, trust_env=False) as client:
        response = client.get(url.rstrip("/") + "/api/tags")
        response.raise_for_status()
        names = {m.get("name") for m in response.json().get("models", [])}
    if model not in names and model + ":latest" not in names:
        raise RuntimeError("Router model not installed; run ollama pull for ROUTER_MODEL")


def main():
    for directory in (os.getenv("SAGE_DATA_DIR", "/app/data"), os.getenv("TMPDIR", "/app/data/tmp"), "logs"):
        Path(directory).mkdir(parents=True, exist_ok=True)
    if os.getenv("OLLAMA_CHECK_ON_START", "true").lower() == "true":
        deadline = time.monotonic() + float(os.getenv("OLLAMA_WAIT_SECONDS", "60"))
        while True:
            try:
                check_ollama(os.getenv("ROUTER_BASE_URL", "http://host.docker.internal:11434"),
                             os.getenv("ROUTER_MODEL", "qwen3.5:2b"))
                break
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                if time.monotonic() >= deadline:
                    raise SystemExit("Ollama readiness failed (" + type(exc).__name__ +
                                     "). Check ROUTER_BASE_URL, host bind address and ollama list.") from None
                print("Waiting for Ollama and the configured Router model...", flush=True)
                time.sleep(2)
    # One worker: RunManager and MCP ownership are intentionally single-process.
    os.execvp("python", ["python", "-m", "uvicorn", "sage_scholar.api.app:app", "--host", "0.0.0.0",
                         "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "45"])


if __name__ == "__main__":
    main()
