import json
import os
import signal
import subprocess
import sys
import time

import pytest

from sage_scholar.mcp.client import MCPClient, MCPServerConfig
from sage_scholar.mcp.loader import iter_mcp_clients


@pytest.fixture
def server(tmp_path):
    script = tmp_path / "server.py"
    script.write_text('''
import json, os, signal, subprocess, sys, time
mode, pidfile = sys.argv[1:]
child = None
if mode == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, "-c", "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"])
with open(pidfile, "w") as f:
    f.write(json.dumps([os.getpid()] + ([child.pid] if child else [])))
if mode == "stubborn":
    time.sleep(60)
for line in sys.stdin:
    data = json.loads(line)
    method = data.get("method")
    if "id" not in data: continue
    if mode == "initialize_hang": continue
    if method == "initialize":
        result = {"protocolVersion":"2025-03-26", "capabilities":{"tools":{}}, "serverInfo":{"name":"test","version":"1"}}
    elif method == "tools/list":
        if mode == "list_hang": continue
        result = {"tools":[{"name":"echo","description":"test", "inputSchema":{"type":"object","properties":{}}}]}
    elif method == "tools/call":
        if mode == "call_hang": continue
        result = {"content":[{"type":"text","text":"source text"}]}
    else: continue
    print(json.dumps({"jsonrpc":"2.0","id":data["id"],"result":result}), flush=True)
''')
    return script


def alive(pid):
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return result.returncode == 0 and bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


def client_for(server, tmp_path, mode, **kwargs):
    pidfile = tmp_path / f"{mode}.json"
    client = MCPClient(MCPServerConfig(name=mode, command=sys.executable,
        args=[str(server), mode, str(pidfile)], connect_timeout=0.4, **kwargs))
    return client, pidfile


@pytest.mark.parametrize("mode", ["initialize_hang", "list_hang", "stubborn"])
def test_startup_timeout_reaps_subprocess_and_descendants(server, tmp_path, mode):
    client, pidfile = client_for(server, tmp_path, mode)
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            client.connect()
        assert time.monotonic() - start < 7
        assert client.status == "unavailable" and not client._thread.is_alive()
        pids = json.loads(pidfile.read_text())
        deadline = time.monotonic() + 2
        while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(alive(pid) for pid in pids), "timed-out server/child survived"
    finally:
        client.disconnect()
        if pidfile.exists():
            for pid in json.loads(pidfile.read_text()):
                if alive(pid):
                    os.kill(pid, signal.SIGKILL)


def test_working_server_remains_usable_then_disconnects(server, tmp_path):
    client, pidfile = client_for(server, tmp_path, "healthy")
    try:
        assert client.connect()[0].name == "echo"
        assert client.status == "available"
        assert client.call_tool("echo", {}) == "source text"
    finally:
        client.disconnect()
    assert not client._thread.is_alive()
    assert not alive(json.loads(pidfile.read_text())[0])


def test_tool_call_wait_is_also_bounded(server, tmp_path):
    client, pidfile = client_for(server, tmp_path, "call_hang", call_timeout=0.2)
    try:
        client.connect()
        start = time.monotonic()
        from sage_scholar.tools import ToolCallError
        with pytest.raises(ToolCallError, match="timed out"):
            client.call_tool("echo", {})
        assert time.monotonic() - start < 7
    finally:
        client.disconnect()
    assert not client._thread.is_alive()
    assert not alive(json.loads(pidfile.read_text())[0])


def test_optional_failure_does_not_delay_available_peer_or_start_missing_key(server, tmp_path, monkeypatch):
    monkeypatch.delenv("SAGE_TEST_MISSING_KEY", raising=False)
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({
        "fetch": {"command": sys.executable, "args": [str(server), "initialize_hang", str(tmp_path / "bad.json")]},
        "paper-search": {"command": sys.executable, "args": [str(server), "healthy", str(tmp_path / "good.json")]},
        "brave-search": {"command": "should-never-run", "args": [], "env": {"KEY": "${SAGE_TEST_MISSING_KEY}"}},
        "github": {"command": "should-never-run", "args": []},
    }))
    statuses, clients = [], []
    try:
        for status, client in iter_mcp_clients(str(config), server_names={"fetch", "paper-search", "brave-search"}, connect_timeout=0.4):
            statuses.append(status)
            if client:
                clients.append(client)
        assert len(clients) == 1 and clients[0].call_tool("echo", {}) == "source text"
        by_name = {s["server"]: s for s in statuses}
        assert by_name["fetch"]["reason"] == "timeout"
        assert by_name["brave-search"]["reason"] == "missing_config"
        assert list(by_name).index("paper-search") < list(by_name).index("fetch")
        assert "github" not in by_name
        assert "Traceback" not in json.dumps(statuses)
    finally:
        for client in clients:
            client.disconnect()


def test_invalid_or_missing_placeholder_config_is_skipped_before_spawn(tmp_path, monkeypatch):
    monkeypatch.delenv("SAGE_TEST_MISSING_KEY", raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid or missing credentials must not start a process")
    monkeypatch.setattr(MCPClient, "connect", forbidden)
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({
        "fetch": "invalid entry",
        "brave-search": {"command": "unused", "args": [], "env": {"BRAVE_API_KEY": "${SAGE_TEST_MISSING_KEY}"}},
    }))
    results = list(iter_mcp_clients(config, server_names={"fetch", "brave-search"}))
    assert {status["reason"] for status, _ in results} == {"invalid_config", "missing_config"}
    assert all(client is None for _, client in results)


@pytest.mark.parametrize("resolved", [None, "", "   "])
def test_loader_only_treats_empty_placeholders_as_missing(tmp_path, monkeypatch, resolved):
    from sage_scholar.mcp import loader
    if resolved is None:
        monkeypatch.delenv("SAGE_TEST_EMPTY", raising=False)
    else:
        monkeypatch.setenv("SAGE_TEST_EMPTY", resolved)
    monkeypatch.setenv("all_proxy", "http://inherited.invalid:8080")
    received = []
    class Client:
        def __init__(self, config):
            self.config = config
        def connect(self):
            received.append(self.config.to_stdio_params().env)
        def disconnect(self):
            pass
    monkeypatch.setattr(loader, "MCPClient", Client)
    path = tmp_path / "env.json"
    path.write_text(json.dumps({
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"],
                  "env": {"all_proxy": "", "LITERAL_SPACES": "  ", "OPTIONAL": "${SAGE_TEST_EMPTY}"},
                  "optional_env": ["OPTIONAL"]},
        "tavily": {"command": "unused", "args": [], "env": {"KEY": "${SAGE_TEST_EMPTY}"}},
    }))
    results = list(loader.iter_mcp_clients(path, server_names={"fetch", "tavily"}))
    statuses = {status["server"]: status for status, _ in results}
    assert statuses["fetch"]["status"] == "available"
    assert statuses["tavily"]["reason"] == "missing_config"
    assert len(received) == 1
    assert received[0]["all_proxy"] == "" and received[0]["LITERAL_SPACES"] == "  "
    assert "OPTIONAL" not in received[0]


def test_project_configuration_disables_brave_and_preserves_fetch():
    from pathlib import Path
    from sage_scholar.mcp.loader import RESEARCH_SERVERS
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/mcp_servers.json").read_text())
    assert "brave-search" not in config and "brave-search" not in RESEARCH_SERVERS
    assert config["fetch"]["env"]["all_proxy"] == ""
