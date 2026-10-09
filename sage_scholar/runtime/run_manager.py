from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage_scholar.application.ports import LongTaskRunner
import uuid
import logging

logger = logging.getLogger(__name__)

from sage_scholar.context.manager import request_context
from sage_scholar.domain.execution import ExecutionRequest
from .journal import WorkflowInterrupted, is_safe_retry_intent
from .control import RunControl, RunBudgetExceeded, run_control
from .human import RunWaiting, HumanResponse, public_interrupts
from .limits import ExecutionLimits, execution_budget
from .models import RunEvent, RunSnapshot, RunStatus, TERMINAL_STATUSES, utc_now


@dataclass
class _RunState:
    snapshot: RunSnapshot
    events: list[RunEvent] = field(default_factory=list)
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    cancel_requested: bool = False
    shutdown_requested: bool = False
    control: object = None


class RunManager:
    """Durable lifecycle and SSE; never tracks a graph's current/next node."""
    def __init__(self, max_concurrent_runs=2, *, store=None, checkpoints=None, on_completed=None):
        self.store, self.checkpoints = store, checkpoints
        self.on_completed = on_completed
        self._runs = {snapshot.run_id: _RunState(snapshot, events) for snapshot, events in store.load()} if store else {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_runs)
        self._runners = {}
        self._closing = False

    def register_runners(self, runners):
        self._runners = runners

    def _assert_conversation_idle(self, request, excluding=None):
        if not self.store or not request.conversation_id:
            return
        self.store.conversations.assert_open(request.conversation_id)
        if any(r.snapshot.run_id != excluding and r.snapshot.execution.conversation_id == request.conversation_id
               and r.snapshot.status not in TERMINAL_STATUSES for r in self._runs.values()):
            raise ValueError("当前对话已有运行中或等待用户的任务")

    async def start(self, request: ExecutionRequest, runner: LongTaskRunner):
        if self._closing:
            raise ValueError("服务正在关闭，请稍后重试")
        self._assert_conversation_idle(request)
        run_id, now = uuid.uuid4().hex, utc_now()
        request = request.model_copy(update={"run_id": run_id, "budget": execution_budget(request.workflow, request.budget)}, deep=True)
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
            if self.store:
                self.store.clear_reservations(state.snapshot.run_id)
                if self.store.conversations.source_deleted(state.snapshot.execution.conversation_id):
                    await self.remove(state.snapshot.run_id)
                    continue
            if state.snapshot.status == RunStatus.WAITING_FOR_USER:
                continue
            if state.snapshot.status in {RunStatus.PENDING, RunStatus.RUNNING, RunStatus.CANCEL_REQUESTED, RunStatus.INTERRUPTED}:
                if state.snapshot.status == RunStatus.CANCEL_REQUESTED:
                    self._set_status(state, RunStatus.CANCELLED)
                    await self._append_event(state, "run_cancelled", {"reason": "cancel requested before restart"})
                    continue
                if self.checkpoints and not state.snapshot.resume_input:
                    pending = self.checkpoints.pending_interrupts(state.snapshot.run_id)
                    if pending:
                        state.snapshot.interrupts = public_interrupts(pending)
                        self._set_status(state, RunStatus.WAITING_FOR_USER)
                        await self._append_event(state, "waiting_for_user", {"interrupts": state.snapshot.interrupts})
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
        stale_unknown_lock = run.restart_required and (
            "unknown_execution_state" in (run.reason or "")
            or "外部工具执行状态未知" in (run.reason or "")
            or "未确认结果的外部调用" in (run.reason or "")
        )
        if run.restart_required and not stale_unknown_lock:
            return run.reason or "restart_required：需要明确重新开始。"
        unknown = self.store.unknown_call_records(run.run_id, mark=True) if self.store else []
        unsafe = [record for record in unknown if not is_safe_retry_intent(record["intent"])]
        if unsafe:
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
        if run_id in self._tasks and not self._tasks[run_id].done():
            return self.get(run_id)
        self._assert_conversation_idle(state.snapshot.execution, excluding=run_id)
        if state.snapshot.status not in {RunStatus.INTERRUPTED, RunStatus.FAILED, RunStatus.PARTIAL, RunStatus.CANCELLED}:
            raise ValueError("只有中断、停止、失败或部分完成的任务可以恢复")
        if self._closing:
            raise ValueError("服务正在关闭")
        issue = self._recovery_issue(state)
        if issue:
            state.snapshot.restart_required = True
            state.snapshot.reason = issue
            await self._append_event(state, "run_interrupted", {"reason": issue, "restart_required": True})
            raise ValueError(issue)
        if state.snapshot.status in {RunStatus.PARTIAL, RunStatus.CANCELLED}:
            old = execution_budget(state.snapshot.workflow, state.snapshot.execution.budget)
            ceiling = ExecutionLimits.model_fields
            extension = execution_budget(state.snapshot.workflow)
            for key in ("max_total_tokens", "max_tool_calls", "max_seconds"):
                maximum = next(m.le for m in ceiling[key].metadata if hasattr(m, "le"))
                old[key] = min(maximum, old[key] + extension[key])
            metadata = {**state.snapshot.execution.metadata, "continuation": int(state.snapshot.execution.metadata.get("continuation", 0)) + 1}
            state.snapshot.execution = state.snapshot.execution.model_copy(update={"budget": old, "metadata": metadata})
            if self.store:
                self.store.reset_blocked(run_id)
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
        request = state.snapshot.execution.model_copy(update={"run_id": None, "request_id": uuid.uuid4().hex, "metadata": {
            **state.snapshot.execution.metadata, "restarted_from": run_id}})
        runner = self._runners.get(request.workflow)
        if not runner:
            raise ValueError("workflow unavailable")
        return await self.start(request, runner)

    async def answer(self, run_id, response: HumanResponse):
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(run_id)
        if response.run_id != run_id or response.conversation_id != state.snapshot.execution.conversation_id:
            raise ValueError("回答与当前对话/任务不匹配")
        if state.snapshot.status != RunStatus.WAITING_FOR_USER:
            raise ValueError("该问题已回答、已取消或已过期")
        question = next((q for q in state.snapshot.interrupts if q["interrupt_id"] == response.interrupt_id), None)
        if not question or (response.skip and not question.get("allow_skip")) or (not response.skip and not response.answer.strip()):
            raise ValueError("无效的回答或不可跳过的问题")
        if self.store:
            self.store.conversations.assert_open(response.conversation_id)
        # Consume the question durably before scheduling; duplicate submissions see PENDING.
        issue = self._recovery_issue(state)
        if issue:
            raise ValueError(issue)
        state.snapshot.resume_input = {response.interrupt_id: response.model_dump()}
        state.snapshot.interrupts = []
        self._set_status(state, RunStatus.PENDING)
        await self._append_event(state, "human_responded", response.model_dump())
        previous = self._tasks.get(run_id)
        if previous and not previous.done():
            await asyncio.shield(previous)
        self._launch(state, self._runners[state.snapshot.workflow])
        return self.get(run_id)

    async def cancel(self, run_id):
        state = self._runs.get(run_id)
        if state is None:
            return None
        if state.snapshot.status not in TERMINAL_STATUSES and not state.cancel_requested:
            state.cancel_requested = True
            waiting = state.snapshot.status in {RunStatus.WAITING_FOR_USER, RunStatus.PENDING}
            self._set_status(state, RunStatus.CANCELLED if waiting else RunStatus.CANCEL_REQUESTED)
            state.snapshot.interrupts = []
            await self._append_event(state, "run_cancelled" if waiting else "cancel_requested", {})
            task = self._tasks.get(run_id)
            if task and not task.done():
                task.cancel()
            elif not waiting:
                self._set_status(state, RunStatus.CANCELLED)
                await self._append_event(state, "run_cancelled", {})
        return self.get(run_id)

    async def remove(self, run_id):
        await self.cancel(run_id)
        task = self._tasks.get(run_id)
        if task:
            await asyncio.shield(asyncio.gather(task, return_exceptions=True))
        if self.checkpoints:
            self.checkpoints.delete(run_id)
        if self.store:
            self.store.purge_run(run_id)
        self._runs.pop(run_id, None)

    async def events(self, run_id: str, after=0) -> AsyncIterator[RunEvent]:
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(run_id)
        cursor = max(0, after)
        while True:
            async with state.changed:
                await state.changed.wait_for(lambda: any(e.sequence > cursor for e in state.events) or state.snapshot.status in TERMINAL_STATUSES or state.snapshot.status == RunStatus.WAITING_FOR_USER)
                pending = [event.model_copy(deep=True) for event in state.events if event.sequence > cursor]
                terminal = state.snapshot.status in TERMINAL_STATUSES or state.snapshot.status == RunStatus.WAITING_FOR_USER
            for event in pending:
                cursor = event.sequence
                yield event
            if terminal:
                return

    async def shutdown(self):
        self._closing = True
        for state in self._runs.values():
            if state.snapshot.status not in TERMINAL_STATUSES and state.snapshot.status != RunStatus.WAITING_FOR_USER:
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
                loop = asyncio.get_running_loop()
                def emit_sync(kind, data):
                    asyncio.run_coroutine_threadsafe(emit(kind, data), loop).result()
                control = RunControl(state.snapshot.execution, self.store,
                    lambda: state.cancel_requested or state.shutdown_requested, emit_sync) if self.store else None
                state.control = control
                request = state.snapshot.execution
                if state.snapshot.resume_input:
                    request = request.model_copy(update={"metadata": {**request.metadata, "_resume_input": state.snapshot.resume_input}})
                with request_context(request), run_control(control):
                    outcome = await runner.run(request, emit, lambda: state.cancel_requested or state.shutdown_requested)
                if control:
                    control.meter()
                state.snapshot.resume_input = {}
                if state.cancel_requested or state.shutdown_requested:
                    raise WorkflowInterrupted("任务停止")
                state.snapshot.result, state.snapshot.reason = outcome.result, outcome.reason
                state.snapshot.remaining_gaps = outcome.remaining_gaps
                self._set_status(state, RunStatus(outcome.status))
                await self._append_event(state, "run_finished", {"status": outcome.status, "reason": outcome.reason,
                                                                "remaining_gaps": outcome.remaining_gaps})
                if self.on_completed:
                    try:
                        self.on_completed(state.snapshot.model_copy(deep=True), tuple(state.events))
                    except Exception:
                        logger.exception("Post-run memory scheduling failed; Run result retained")
        except RunWaiting as exc:
            if state.control:
                state.control.meter()
            state.snapshot.interrupts = exc.interrupts
            state.snapshot.resume_input = {}
            self._set_status(state, RunStatus.WAITING_FOR_USER)
            await self._append_event(state, "waiting_for_user", {"interrupts": exc.interrupts})
        except RunBudgetExceeded as exc:
            self._set_status(state, RunStatus.PARTIAL)
            state.snapshot.reason = str(exc)
            state.snapshot.remaining_gaps = list(dict.fromkeys([*state.snapshot.remaining_gaps, str(exc)]))
            await self._append_event(state, "run_finished", {"status": "partial", "reason": str(exc)})
        except (asyncio.CancelledError, WorkflowInterrupted) as exc:
            status = RunStatus.CANCELLED if state.cancel_requested else RunStatus.INTERRUPTED
            self._set_status(state, status)
            state.snapshot.reason = str(exc) or "服务关闭，任务中断"
            await self._append_event(state, "run_cancelled" if status == RunStatus.CANCELLED else "run_interrupted", {"reason": state.snapshot.reason})
        except Exception as exc:
            state.snapshot.error = str(exc)
            self._set_status(state, RunStatus.FAILED)
            await self._append_event(state, "error", {"message": str(exc)})
        finally:
            state.control = None  # Waiting/idle time must never consume execution budget.

    @staticmethod
    def _set_status(state, status):
        state.snapshot.status = status
        state.snapshot.updated_at = utc_now()

    async def _append_event(self, state, event_type, data):
        if state.control and state.snapshot.status != RunStatus.WAITING_FOR_USER:
            state.control.meter()
        if event_type in {"initializing_tools", "tool_unavailable", "research_started"}:
            state.snapshot.progress["message"] = data.get("message", "")
        elif event_type == "research_planned":
            tasks = {t["task_id"]: t for t in state.snapshot.progress.get("tasks", [])}
            tasks.update({t["task_id"]: t for t in data.get("tasks", [])})
            state.snapshot.progress["tasks"] = list(tasks.values())
        elif event_type == "research_task_completed":
            note = data.get("note", {})
            for task in state.snapshot.progress.get("tasks", []):
                if task["task_id"] == note.get("task_id"):
                    task["status"] = "failed" if note.get("metadata", {}).get("task_error") else "completed"
        elif event_type in {"research_synthesizing", "report_drafting", "report_reviewing", "report_revising"}:
            state.snapshot.progress["message"] = {"research_synthesizing": "正在综合研究发现…", "report_drafting": "正在分章节撰写报告…", "report_reviewing": "正在审查报告内容…", "report_revising": "正在局部修订报告…"}[event_type]
        if event_type == "answer_delta":
            state.snapshot.result = {**(state.snapshot.result or {}), "answer": (state.snapshot.result or {}).get("answer", "") + data.get("text", "")}
        elif event_type in {"answer", "write"}:
            state.snapshot.result = {**(state.snapshot.result or {}), **data}
        event = RunEvent(sequence=state.snapshot.event_cursor + 1, type=event_type, data=data)
        state.snapshot.event_cursor = event.sequence
        state.snapshot.updated_at = utc_now()
        if self.store:
            state.snapshot.consumed_budget = self.store.budget(state.snapshot.run_id)
            state.snapshot.stage_usage = self.store.stages(state.snapshot.run_id)
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
