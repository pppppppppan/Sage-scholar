from __future__ import annotations

import asyncio
from collections.abc import Callable

from .ports import EventEmitter


class ResearchService:
    """Adapts the synchronous SAGE Research graph to the async RunManager."""

    def __init__(self, runtime_provider):
        self._runtime_provider = runtime_provider

    async def run(self, query: str, emit: EventEmitter, cancelled: Callable[[], bool]) -> dict:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def produce() -> None:
            try:
                runtime = self._runtime_provider.get()
                usage_before = runtime.usage_snapshot()
                for raw_event in runtime.run_research(query):
                    if cancelled():
                        break
                    event = runtime.to_public_event(raw_event)
                    if event is not None:
                        asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
                usage_after = runtime.usage_snapshot()
                stats = {
                    key: max(0, value - usage_before.get(key, 0))
                    for key, value in usage_after.items()
                }
                asyncio.run_coroutine_threadsafe(queue.put(("stats", stats)), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

        producer = asyncio.create_task(asyncio.to_thread(produce))
        final_report: str | None = None
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                event_type, data = item
                if event_type == "write":
                    final_report = data.get("report")
                await emit(event_type, data)
        finally:
            await producer

        return {"report": final_report} if final_report else {}
