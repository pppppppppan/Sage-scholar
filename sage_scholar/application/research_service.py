from __future__ import annotations

import asyncio
from collections.abc import Callable

from .ports import EventEmitter
from sage_scholar.domain.execution import ExecutionRequest, RunOutcome


class ResearchService:
    """Adapts the synchronous SAGE Research graph to the async RunManager."""

    def __init__(self, runtime_provider):
        self._runtime_provider = runtime_provider

    async def run(self, request: ExecutionRequest, emit: EventEmitter, cancelled: Callable[[], bool]) -> RunOutcome:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def produce() -> None:
            try:
                runtime = self._runtime_provider.get(require_external=request.source_policy.permits_web)
                usage_before = runtime.usage_snapshot()
                for raw_event in runtime.run_research(request):
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
        remaining_gaps = []
        evidence_ids = []
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
                if event_type == "outcome":
                    remaining_gaps = data.get("remaining_gaps", [])
                if event_type == "evidence":
                    evidence_ids = data.get("evidence_ids", [])
                await emit(event_type, data)
        finally:
            await producer

        if not final_report:
            remaining_gaps.append("未生成研究报告")
        return RunOutcome(
            status="partial" if remaining_gaps else "completed",
            result={"report": final_report, "evidence_ids": evidence_ids},
            reason="研究仍有未覆盖项" if remaining_gaps else None,
            remaining_gaps=remaining_gaps,
        )
