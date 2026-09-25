from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import uuid

from sage_scholar.application.ports import LongTaskRunner
from sage_scholar.routing import Workflow

from .models import RunEvent, RunSnapshot, RunStatus, TERMINAL_STATUSES, utc_now


@dataclass
class _RunState:
    snapshot: RunSnapshot
    events: list[RunEvent] = field(default_factory=list)
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    cancel_requested: bool = False


class RunManager:
    """Owns task lifecycle; workflow state remains inside LangGraph."""

    def __init__(self, max_concurrent_runs: int = 2):
        self._runs: dict[str, _RunState] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_runs)

    async def start(self, workflow: Workflow, query: str, runner: LongTaskRunner) -> RunSnapshot:
        run_id = uuid.uuid4().hex
        now = utc_now()
        state = _RunState(
            snapshot=RunSnapshot(
                run_id=run_id,
                workflow=workflow,
                query=query,
                status=RunStatus.CREATED,
                created_at=now,
                updated_at=now,
            )
        )
        self._runs[run_id] = state
        await self._append_event(state, "run_created", {"workflow": workflow.value})
        task = asyncio.create_task(self._execute(state, runner), name=f"sage-run-{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda _task, rid=run_id: self._tasks.pop(rid, None))
        return state.snapshot.model_copy(deep=True)

    def get(self, run_id: str) -> RunSnapshot | None:
        state = self._runs.get(run_id)
        return state.snapshot.model_copy(deep=True) if state else None

    async def cancel(self, run_id: str) -> RunSnapshot | None:
        state = self._runs.get(run_id)
        if state is None:
            return None
        if state.snapshot.status in TERMINAL_STATUSES:
            return state.snapshot.model_copy(deep=True)
        state.cancel_requested = True
        self._set_status(state, RunStatus.CANCEL_REQUESTED)
        await self._append_event(state, "cancel_requested", {})
        return state.snapshot.model_copy(deep=True)

    async def events(self, run_id: str, after: int = 0) -> AsyncIterator[RunEvent]:
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(run_id)
        cursor = max(0, after)
        while True:
            async with state.changed:
                await state.changed.wait_for(
                    lambda: cursor < len(state.events)
                    or state.snapshot.status in TERMINAL_STATUSES
                )
                pending = [event.model_copy(deep=True) for event in state.events[cursor:]]
                cursor = len(state.events)
                terminal = state.snapshot.status in TERMINAL_STATUSES
            for event in pending:
                yield event
            if terminal and cursor >= len(state.events):
                return

    async def shutdown(self) -> None:
        for state in self._runs.values():
            if state.snapshot.status not in TERMINAL_STATUSES:
                state.cancel_requested = True
        if self._tasks:
            done, pending = await asyncio.wait(self._tasks.values(), timeout=5)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _execute(self, state: _RunState, runner: LongTaskRunner) -> None:
        try:
            async with self._semaphore:
                if state.cancel_requested:
                    self._set_status(state, RunStatus.CANCELLED)
                    await self._append_event(state, "run_cancelled", {})
                    return

                self._set_status(state, RunStatus.RUNNING)
                await self._append_event(state, "run_started", {})

                async def emit(event_type: str, data: dict) -> None:
                    await self._append_event(state, event_type, data)

                result = await runner.run(
                    state.snapshot.query,
                    emit,
                    lambda: state.cancel_requested,
                )
                state.snapshot.result = result
                if state.cancel_requested:
                    self._set_status(state, RunStatus.CANCELLED)
                    await self._append_event(state, "run_cancelled", {})
                else:
                    self._set_status(state, RunStatus.COMPLETED)
                    await self._append_event(state, "run_finished", {"status": "completed"})
        except asyncio.CancelledError:
            self._set_status(state, RunStatus.INTERRUPTED)
            await self._append_event(state, "run_interrupted", {})
            raise
        except Exception as exc:
            state.snapshot.error = str(exc)
            self._set_status(state, RunStatus.FAILED)
            await self._append_event(state, "error", {"message": str(exc)})

    def _set_status(self, state: _RunState, status: RunStatus) -> None:
        state.snapshot.status = status
        state.snapshot.updated_at = utc_now()

    async def _append_event(self, state: _RunState, event_type: str, data: dict) -> None:
        event = RunEvent(sequence=len(state.events) + 1, type=event_type, data=data)
        state.events.append(event)
        state.snapshot.updated_at = utc_now()
        async with state.changed:
            state.changed.notify_all()
