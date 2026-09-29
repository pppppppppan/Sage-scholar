from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage_scholar.application.ports import LongTaskRunner
import uuid

from sage_scholar.context.manager import request_context
from sage_scholar.domain.execution import ExecutionRequest
from .journal import WorkflowInterrupted
from .models import RunEvent, RunSnapshot, RunStatus, TERMINAL_STATUSES, utc_now


@dataclass
class _RunState:
    snapshot: RunSnapshot
    events: list[RunEvent] = field(default_factory=list)
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    cancel_requested: bool = False
    shutdown_requested: bool = False


class RunManager:
    """Durable lifecycle and SSE; never tracks a graph's current/next node."""
    def __init__(self, max_concurrent_runs=2, *, store=None, checkpoints=None):
        self.store, self.checkpoints = store, checkpoints
        self._runs = {snapshot.run_id: _RunState(snapshot, events) for snapshot, events in store.load()} if store else {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_runs)
        self._runners = {}
        self._closing = False

    def register_runners(self, runners):
        self._runners = runners

    async def start(self, request: ExecutionRequest, runner: LongTaskRunner):
        if self._closing:
            raise ValueError("服务正在关闭，请稍后重试")
        run_id, now = uuid.uuid4().hex, utc_now()
        request = request.model_copy(update={"run_id": run_id}, deep=True)
        state = _RunState(RunSnapshot(run_id=run_id, workflow=request.workflow, query=request.query,
            execution=request, status=RunStatus.PENDING, created_at=now, updated_at=now))
        self._runs[run_id] = state
        await self._append_event(state, "run_created", {"workflow": request.workflow.value})
        self._launch(state, runner)
        return self.get(run_id)

    def _launch(self, state, runner):
        run_id = state.snapshot.run_id
        task = asyncio.create_task(self._execute(state, runner), name=f"sage-run-{run_id}")
        self._tasks[run_id] = task
        def completed(done):
            if self._tasks.get(run_id) is done:
                self._tasks.pop(run_id, None)
        task.add_done_callback(completed)

    def get(self, run_id):
        state = self._runs.get(run_id)
        if not state:
            return None
        result = state.snapshot.model_copy(deep=True)
        if self.store:
            result.consumed_budget = self.store.budget(run_id)
        return result

    def list(self):
        return sorted((self.get(identity) for identity in self._runs), key=lambda r: r.created_at, reverse=True)

    async def recover(self):
        for state in list(self._runs.values()):
            if state.snapshot.status in {RunStatus.PENDING, RunStatus.RUNNING, RunStatus.CANCEL_REQUESTED, RunStatus.INTERRUPTED}:
                if state.snapshot.status == RunStatus.CANCEL_REQUESTED:
                    self._set_status(state, RunStatus.CANCELLED)
                    await self._append_event(state, "run_cancelled", {"reason": "cancel requested before restart"})
                    continue
                self._set_status(state, RunStatus.INTERRUPTED)
                issue = self._recovery_issue(state)
                if issue:
                    state.snapshot.restart_required = True
                    state.snapshot.reason = issue
                    await self._append_event(state, "run_interrupted", {"reason": issue, "restart_required": True})
                else:
                    await self.resume(state.snapshot.run_id)

    def _recovery_issue(self, state):
        run = state.snapshot
        if run.restart_required:
            return run.reason or "restart_required：需要明确重新开始。"
        unknown = self.store.unknown_calls(run.run_id, mark=True) if self.store else []
        if unknown:
            return "unknown_execution_state：存在未确认结果的外部调用；不能自动重放，请核查后明确重新开始。"
        if not self.checkpoints or not self.checkpoints.compatible(run):
            return "restart_required：checkpoint 缺失或 schema 不兼容。"
        if run.workflow not in self._runners:
            return "restart_required：当前版本没有该 workflow runner。"
        return None

    async def resume(self, run_id):
        state = self._runs.get(run_id)
        if not state:
            raise KeyError(run_id)
        if run_id in self._tasks:
            return self.get(run_id)
        if state.snapshot.status not in {RunStatus.INTERRUPTED, RunStatus.FAILED}:
            raise ValueError("只有中断或失败的任务可以恢复")
        if self._closing:
            raise ValueError("服务正在关闭")
        issue = self._recovery_issue(state)
        if issue:
            state.snapshot.restart_required = True
            state.snapshot.reason = issue
            await self._append_event(state, "run_interrupted", {"reason": issue, "restart_required": True})
            raise ValueError(issue)
        state.cancel_requested = state.shutdown_requested = False
        state.snapshot.error = state.snapshot.reason = None
        state.snapshot.restart_required = False
        self._set_status(state, RunStatus.PENDING)
        await self._append_event(state, "run_resumed", {"same_run": True})
        self._launch(state, self._runners[state.snapshot.workflow])
        return self.get(run_id)

    async def restart(self, run_id):
        state = self._runs.get(run_id)
        if not state:
            raise KeyError(run_id)
        if state.snapshot.status not in TERMINAL_STATUSES:
            raise ValueError("请先取消正在运行的任务")
        request = state.snapshot.execution.model_copy(update={"run_id": None, "metadata": {
            **state.snapshot.execution.metadata, "restarted_from": run_id}})
        runner = self._runners.get(request.workflow)
        if not runner:
            raise ValueError("workflow unavailable")
        return await self.start(request, runner)

    async def cancel(self, run_id):
        state = self._runs.get(run_id)
        if state is None:
            return None
        if state.snapshot.status not in TERMINAL_STATUSES:
            state.cancel_requested = True
            self._set_status(state, RunStatus.CANCEL_REQUESTED)
            await self._append_event(state, "cancel_requested", {})
        return self.get(run_id)

    async def events(self, run_id: str, after=0) -> AsyncIterator[RunEvent]:
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(run_id)
        cursor = max(0, after)
        while True:
            async with state.changed:
                await state.changed.wait_for(lambda: any(e.sequence > cursor for e in state.events) or state.snapshot.status in TERMINAL_STATUSES)
                pending = [event.model_copy(deep=True) for event in state.events if event.sequence > cursor]
                terminal = state.snapshot.status in TERMINAL_STATUSES
            for event in pending:
                cursor = event.sequence
                yield event
            if terminal:
                return

    async def shutdown(self):
        self._closing = True
        for state in self._runs.values():
            if state.snapshot.status not in TERMINAL_STATUSES:
                state.shutdown_requested = True
        if self._tasks:
            _, pending = await asyncio.wait(list(self._tasks.values()), timeout=5)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _execute(self, state, runner):
        try:
            async with self._semaphore:
                if state.cancel_requested or state.shutdown_requested:
                    raise WorkflowInterrupted("任务停止")
                self._set_status(state, RunStatus.RUNNING)
                await self._append_event(state, "run_started", {})
                async def emit(event_type, data):
                    await self._append_event(state, event_type, data)
                with request_context(state.snapshot.execution):
                    outcome = await runner.run(state.snapshot.execution, emit, lambda: state.cancel_requested or state.shutdown_requested)
                if state.cancel_requested or state.shutdown_requested:
                    raise WorkflowInterrupted("任务停止")
                state.snapshot.result, state.snapshot.reason = outcome.result, outcome.reason
                state.snapshot.remaining_gaps = outcome.remaining_gaps
                self._set_status(state, RunStatus(outcome.status))
                await self._append_event(state, "run_finished", {"status": outcome.status, "reason": outcome.reason,
                                                                "remaining_gaps": outcome.remaining_gaps})
        except (asyncio.CancelledError, WorkflowInterrupted) as exc:
            status = RunStatus.CANCELLED if state.cancel_requested else RunStatus.INTERRUPTED
            self._set_status(state, status)
            state.snapshot.reason = str(exc) or "服务关闭，任务中断"
            await self._append_event(state, "run_cancelled" if status == RunStatus.CANCELLED else "run_interrupted", {"reason": state.snapshot.reason})
        except Exception as exc:
            state.snapshot.error = str(exc)
            self._set_status(state, RunStatus.FAILED)
            await self._append_event(state, "error", {"message": str(exc)})

    @staticmethod
    def _set_status(state, status):
        state.snapshot.status = status
        state.snapshot.updated_at = utc_now()

    async def _append_event(self, state, event_type, data):
        event = RunEvent(sequence=state.snapshot.event_cursor + 1, type=event_type, data=data)
        state.snapshot.event_cursor = event.sequence
        state.snapshot.updated_at = utc_now()
        if self.store:
            state.snapshot.consumed_budget = self.store.budget(state.snapshot.run_id)
        if self.checkpoints and not state.snapshot.restart_required:
            try:
                state.snapshot.checkpoint = self.checkpoints.reference(state.snapshot.run_id, state.snapshot.workflow)
            except Exception:
                pass  # Preserve the recorded schema; recovery will reject incompatible data.
        if self.store:
            self.store.append(state.snapshot, event)
        state.events.append(event)
        async with state.changed:
            state.changed.notify_all()
