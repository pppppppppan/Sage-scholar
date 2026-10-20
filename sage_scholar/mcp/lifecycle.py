"""Process-local, per-server lifecycle shared by research and ingestion.

One connection attempt per server per ensure, exponential cooldown capped at five
minutes. A healthy connection is retained until close; failures never cache a tool.
"""
from dataclasses import dataclass
import os
import logging
import threading
import time

from .loader import iter_mcp_clients, _status
from .tool_adapter import register_mcp_tools
from sage_scholar.tools.registry import ToolRegistry
from sage_scholar.runtime.control import check_control, ResearchBudgetReached
from sage_scholar.runtime.journal import WorkflowInterrupted

logger = logging.getLogger(__name__)


@dataclass
class ServerState:
    status: str = 'uninitialized'
    failures: int = 0
    retry_at: float = 0
    public: dict | None = None


class MCPLifecycle:
    def __init__(self, path, timeout=12, *, clock=time.monotonic):
        self.path, self.timeout, self.clock = path, timeout, clock
        self.retry_base = max(1., float(os.getenv('MCP_RETRY_BASE_SECONDS', '5')))
        self.retry_max = max(self.retry_base, float(os.getenv('MCP_RETRY_MAX_SECONDS', '300')))
        self.states = {}
        self.clients = []
        self.registry = ToolRegistry()
        self.lock = threading.RLock()
        self.closed = False

    def ensure(self, names, connector=None):
        with self.lock:
            if self.closed:
                raise RuntimeError('MCP lifecycle is closed')
            check_control()
            for client in list(self.clients):
                if getattr(client, "status", "available") in {"closed", "unavailable"}:
                    client.disconnect()
                    self.clients.remove(client)
                    for key in list(self.registry.tools):
                        if key.startswith(f"mcp__{client.name}__"):
                            del self.registry.tools[key]
                    self._failed(client.name)
            due = set()
            for name in names:
                state = self.states.setdefault(name, ServerState())
                if state.status != 'ready' and self.clock() >= state.retry_at:
                    state.status = 'retrying' if state.failures else 'uninitialized'
                    due.add(name)
            pending = set(due)
            iterator = None
            try:
                iterator = iter((connector or iter_mcp_clients)(self.path, server_names=due, connect_timeout=self.timeout) if due else [])
                for public, client in iterator:
                    name = public['server']
                    state = self.states[name]
                    if client is not None:
                        try:
                            # Build before adopting: malformed tool metadata must not leak a client.
                            temporary = ToolRegistry()
                            register_mcp_tools(temporary, [client])
                            self.registry.tools.update(temporary.tools)
                        except Exception:
                            client.disconnect()
                            public = _status(name, False, 'invalid_tools')
                            client = None
                    if client is not None:
                        self.clients.append(client)
                        state.status, state.failures, state.retry_at = 'ready', 0, 0
                    else:
                        self._failed(name)
                    state.public = public
                    pending.discard(name)
                    check_control()
            except (WorkflowInterrupted, ResearchBudgetReached):
                raise
            except Exception as exc:
                logger.warning("Optional MCP startup batch failed (%s)", type(exc).__name__)
            finally:
                if hasattr(iterator, "close"):
                    try:
                        iterator.close()
                    except Exception as exc:
                        logger.warning("MCP startup cleanup failed (%s)", type(exc).__name__)
                # Cancellation or malformed/partial connector output cannot strand retrying states.
                for name in pending:
                    self._failed(name)
            return [self.states[name].public for name in sorted(names)]

    def _failed(self, name):
        state = self.states[name]
        state.status = 'unavailable'
        state.failures += 1
        state.retry_at = self.clock() + min(self.retry_max, self.retry_base * 2 ** min(state.failures - 1, 12))
        state.public = _status(name, False, 'connection_failed')

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for client in self.clients:
                try:
                    client.disconnect()
                except Exception:
                    pass  # Continue closing the remaining independent subprocesses.
            self.clients.clear()
            self.registry.tools.clear()
