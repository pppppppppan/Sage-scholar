import asyncio
import logging
import os
import threading
import time
from concurrent.futures import Future
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, Tool
from pydantic import BaseModel, Field

from ..tools import ToolCallError

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    connect_timeout: float = Field(default=12, gt=0)
    call_timeout: float = Field(default=120, gt=0)
    cleanup_timeout: float = Field(default=6, gt=0)

    def to_stdio_params(self) -> StdioServerParameters:
        return StdioServerParameters(command=self.command, args=self.args,
                                     env={**os.environ, **(self.env or {})})


class MCPClient:
    """One owning task for the stdio/session contexts, with bounded startup and shutdown."""
    def __init__(self, config: MCPServerConfig) -> None:
        self.name = config.name
        self.config = config
        self.server_config = config.to_stdio_params()
        self._thread = self._loop = self._queue = self._main_task = None
        self._ready_event = threading.Event()
        self._stop_requested = threading.Event()
        self._connect_error = None
        self._pending: set[Future] = set()
        self._pending_lock = threading.Lock()
        self.tools = []
        self.status = "not_started"

    def connect(self) -> list[Tool]:
        if self.status == "available":
            return self.tools
        if self._thread is not None:
            raise RuntimeError("MCP client cannot be reconnected; create a new client")
        self.status = "initializing"
        self._thread = threading.Thread(target=self._run_background, name=f"mcp-{self.name}", daemon=True)
        self._thread.start()
        # Async timeout covers spawn + initialize + list_tools. The extra allowance
        # lets the SDK close stdin and terminate the process group before returning.
        if not self._ready_event.wait(self.config.connect_timeout + self.config.cleanup_timeout):
            self._connect_error = TimeoutError(f"MCP {self.name} initialization timed out")
        if self._connect_error is not None or self.status != "available":
            self.status = "unavailable"
            self.disconnect()
            raise self._connect_error or RuntimeError(f"MCP {self.name} unavailable")
        return self.tools

    def _run_background(self):
        try:
            asyncio.run(self._async_main())
        finally:
            self._ready_event.set()

    async def _async_main(self):
        self._loop = asyncio.get_running_loop()
        self._main_task = asyncio.current_task()
        self._queue = asyncio.Queue()
        try:
            if self._stop_requested.is_set():
                return
            # Keep these nested contexts in the same task. Cancelling only a
            # detached initialize task would leave the stdio subprocess alive.
            async with asyncio.timeout(self.config.connect_timeout) as deadline:
                async with stdio_client(self.server_config) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self.tools = (await session.list_tools()).tools
                        deadline.reschedule(None)
                        self.status = "available"
                        self._ready_event.set()
                        while True:
                            name, args, future = await self._queue.get()
                            if future.cancelled():
                                continue
                            operation = asyncio.create_task(session.call_tool(name=name, arguments=args))
                            def cancelled_call(done, operation=operation):
                                if done.cancelled() and not self._loop.is_closed():
                                    self._loop.call_soon_threadsafe(operation.cancel)
                            future.add_done_callback(cancelled_call)
                            try:
                                result = await asyncio.wait_for(operation, self.config.call_timeout)
                                if not future.done():
                                    future.set_result(result)
                            except asyncio.CancelledError:
                                if self._main_task.cancelling():
                                    raise
                                future.cancel()
                            except Exception as exc:
                                if not future.done():
                                    future.set_exception(exc)
        except asyncio.CancelledError:
            self._connect_error = RuntimeError(f"MCP {self.name} stopped")
        except Exception as exc:
            self._connect_error = exc
        finally:
            self.status = "unavailable" if self._connect_error else "closed"
            with self._pending_lock:
                for future in self._pending:
                    if not future.done():
                        future.set_exception(ToolCallError(f"MCP {self.name} connection closed", self.name))
            self._ready_event.set()

    def call_tool(self, name: str, args: dict[str, Any], _max_retries: int = 3):
        if self.status != "available" or self._loop is None:
            raise ToolCallError(f"MCP {self.name} unavailable", name)
        from sage_scholar.runtime.control import check_control
        from sage_scholar.runtime.journal import WorkflowInterrupted
        for attempt in range(_max_retries + 1):
            check_control()
            future = Future()
            with self._pending_lock:
                self._pending.add(future)
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, (name, args, future))
                deadline = time.monotonic() + self.config.call_timeout + self.config.cleanup_timeout
                while True:
                    check_control()
                    try:
                        result: CallToolResult = future.result(timeout=min(.2, max(.001, deadline - time.monotonic())))
                        break
                    except TimeoutError:
                        if time.monotonic() >= deadline:
                            raise
            except WorkflowInterrupted:
                future.cancel()  # Cancel this request, never the shared MCP server.
                raise
            except TimeoutError:
                future.cancel()
                raise ToolCallError(f"MCP {self.name} tool call timed out", name) from None
            except Exception as exc:
                if "already borrowed" in str(exc).lower() and attempt < _max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
            finally:
                with self._pending_lock:
                    self._pending.discard(future)
            if result.isError:
                error_text = "\n".join(item.text for item in result.content if hasattr(item, "text"))
                if "already borrowed" in error_text.lower() and attempt < _max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise ToolCallError(f"工具调用失败: {error_text}", name)
            return "\n".join(item.text for item in result.content if hasattr(item, "text"))
        raise ToolCallError("工具调用失败: already borrowed 重试次数用尽", name)

    def disconnect(self):
        self._stop_requested.set()
        if self._thread is None:
            return
        def cancel_owner():
            if self._main_task and not self._main_task.done() and not self._main_task.cancelling():
                self._main_task.cancel()
        if self._loop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(cancel_owner)
            except RuntimeError:
                pass  # The owning thread may have finished between the checks.
        self._thread.join(timeout=self.config.cleanup_timeout)
        if self._thread.is_alive():
            logger.error("[MCP] %s cleanup exceeded deadline", self.name)
        else:
            self._loop = None
        if self.status == "available":
            self.status = "closed"
