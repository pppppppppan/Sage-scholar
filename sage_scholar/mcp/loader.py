"""Capability-scoped optional MCP startup; public statuses contain no exception text."""
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .client import MCPClient, MCPServerConfig

logger = logging.getLogger(__name__)
RESEARCH_SERVERS = frozenset({"brave-search", "tavily", "fetch", "paper-search"})
CAPABILITY_SERVERS = {"research": RESEARCH_SERVERS, "pdf": frozenset({"pdfmux"}),
                      "paper": frozenset({"paper-search"})}
_LABELS = {"brave-search": "Brave 搜索", "tavily": "Tavily 搜索", "fetch": "网页读取",
           "paper-search": "论文搜索与读取", "pdfmux": "PDF 解析"}


def _status(name, available, reason=None):
    label = _LABELS.get(name, name)
    return {"server": name, "status": "available" if available else "unavailable", "reason": reason,
            "message": f"{label}已就绪" if available else f"{label}不可用，将继续使用其他来源"}


def iter_mcp_clients(json_path, *, server_names, connect_timeout=12):
    """Yield each availability result as soon as ready; services start in parallel."""
    try:
        with open(json_path, encoding="utf-8") as file:
            configs = json.load(file)
            if not isinstance(configs, dict):
                raise ValueError("MCP configuration must be an object")
    except (OSError, ValueError):
        logger.warning("[MCP] server configuration is unavailable")
        configs = {}
    selected = []
    for name in sorted(server_names):
        config = configs.get(name)
        if config is None:
            yield _status(name, False, "not_configured"), None
            continue
        if not isinstance(config, dict) or not isinstance(config.get("env") or {}, dict):
            yield _status(name, False, "invalid_config"), None
            continue
        env, missing = {}, []
        for key, value in (config.get("env") or {}).items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                value = os.getenv(value[2:-1])
            if value is None or (isinstance(value, str) and not value.strip()):
                if key not in (config.get("optional_env") or []):
                    missing.append(key)
                continue
            env[key] = value
        if missing:
            logger.warning("[MCP] skipping %s: missing configuration %s", name, ", ".join(missing))
            yield _status(name, False, "missing_config"), None
            continue
        try:
            selected.append(MCPServerConfig(name=name, command=config["command"], args=config["args"],
                                           env=env, connect_timeout=connect_timeout))
        except (ValueError, KeyError):
            logger.warning("[MCP] skipping invalid server configuration: %s", name)
            yield _status(name, False, "invalid_config"), None

    def connect(config):
        client = MCPClient(config)
        try:
            client.connect()
            return _status(config.name, True), client
        except Exception as exc:
            client.disconnect()
            logger.warning("[MCP] optional service %s unavailable: %s", config.name, exc)
            return _status(config.name, False, "timeout" if isinstance(exc, TimeoutError) else "connection_failed"), None

    if selected:
        handed_over = set()
        futures = []
        try:
            with ThreadPoolExecutor(max_workers=len(selected), thread_name_prefix="mcp-init") as pool:
                futures = [pool.submit(connect, config) for config in selected]
                for future in as_completed(futures):
                    handed_over.add(future)
                    yield future.result()
        finally:
            # Closing the initialization generator must not orphan successful peers
            # whose results were not adopted by the runtime yet.
            for future in futures:
                if future not in handed_over and future.done() and not future.cancelled():
                    _, client = future.result()
                    if client is not None:
                        client.disconnect()


def create_mcp_clients(json_path, *, server_names=RESEARCH_SERVERS, connect_timeout=12):
    return [client for _, client in iter_mcp_clients(
        json_path, server_names=server_names, connect_timeout=connect_timeout,
    ) if client is not None]
