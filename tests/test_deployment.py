"""Deployment-boundary tests; no model inference and no Docker daemon required."""
import httpx
import pytest

from deploy.entrypoint import check_ollama
from sage_scholar.config import RouterConfig
from sage_scholar.routing.classifier import LocalClassifier


def test_router_container_hosts_are_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("ROUTER_ALLOWED_HOSTS", raising=False)
    for host in ("ollama", "host.docker.internal", "cloud.example"):
        with pytest.raises(ValueError):
            LocalClassifier(RouterConfig(base_url=f"http://{host}:11434"))
    monkeypatch.setenv("ROUTER_ALLOWED_HOSTS", "ollama,host.docker.internal")
    for host in ("ollama", "host.docker.internal", "127.0.0.1"):
        LocalClassifier(RouterConfig(base_url=f"http://{host}:11434"))
    for url in ("http://ollama.evil:11434", "https://cloud.example", "http://user:secret@ollama:11434"):
        with pytest.raises(ValueError):
            LocalClassifier(RouterConfig(base_url=url))


def test_readiness_checks_model_not_just_http(monkeypatch):
    def response(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen3.5:2b"}]})
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(response), **kwargs))
    check_ollama("http://127.0.0.1:11434", "qwen3.5:2b")
    with pytest.raises(RuntimeError, match="not installed"):
        check_ollama("http://127.0.0.1:11434", "missing:2b")
