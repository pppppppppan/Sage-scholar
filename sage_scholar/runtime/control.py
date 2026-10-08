"""Per-Run cooperative cancellation and execution budgets, separate from Context size."""
import asyncio
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from .journal import WorkflowInterrupted

CURRENT_CONTROL = ContextVar('sage_run_control', default=None)


class RunBudgetExceeded(WorkflowInterrupted):
    pass


class ResearchBudgetReached(Exception):
    """Stop gathering sources while retaining the report/verification allocation."""


class RunControl:
    def __init__(self, request, store, cancelled, emit=None):
        self.request, self.store, self.cancelled, self.emit = request, store, cancelled, emit
        self.lock = threading.RLock()
        self.last_meter = time.monotonic()
        self.elapsed = 0.0
        self.stopped = threading.Event()
        self.budget_error = None
        self.research_stopped = threading.Event()

    def meter(self):
        with self.lock:
            now = time.monotonic()
            delta, self.last_meter = now - self.last_meter, now
            self.elapsed += delta
            if self.store:
                self.store.add_seconds(self.request.run_id, delta)

    def check(self):
        if self.stopped.is_set() or self.cancelled():
            raise WorkflowInterrupted('用户已停止任务')
        self.meter()
        elapsed = self.store.budget(self.request.run_id).get('active_seconds', 0) if self.store else self.elapsed
        if elapsed >= self.request.budget.get('max_seconds', float('inf')):
            raise RunBudgetExceeded('运行时间预算已耗尽，已有进度已保存')

    @property
    def research_limit(self):
        maximum = self.request.budget.get('max_total_tokens', 2_000_000)
        reserve = self.request.budget.get('report_reserve_tokens', 500_000)
        return maximum - min(reserve, maximum // 2)

    def research_exhausted(self):
        return self.request.workflow == 'deep_research' and (self.research_stopped.is_set() or
            bool(self.store and self.store.model_load(self.request.run_id) >= self.research_limit))

    def reserve_model(self, tokens, purpose=""):
        self.check()
        if self.budget_error:
            raise RunBudgetExceeded(self.budget_error)
        if not self.store:
            return None
        limit = self.request.budget.get('max_total_tokens', 10_000_000)
        gathering = self.request.workflow == 'deep_research' and (purpose.endswith(':react') or purpose == 'supervisor:plan')
        if gathering:
            if self.research_stopped.is_set():
                raise ResearchBudgetReached('停止继续检索：已进入报告预算预留区')
            limit = self.research_limit
        lease = self.store.reserve_model(self.request.run_id, tokens, limit)
        if lease is None and gathering:
            self.research_stopped.set()
            raise ResearchBudgetReached('已达到资料收集 token 预算，保留剩余额度用于笔记、报告与核验')
        if lease is None:
            self.budget_error = 'Run 累计模型 token 预算已耗尽（不是单次 Context Window 上限）'
            raise RunBudgetExceeded(self.budget_error)
        return lease

    def release_model(self, lease):
        if lease and self.store:
            self.store.release_model(self.request.run_id, lease)


def check_control():
    control = CURRENT_CONTROL.get()
    if control:
        control.check()


@contextmanager
def run_control(control):
    token = CURRENT_CONTROL.set(control)
    try:
        yield
    finally:
        CURRENT_CONTROL.reset(token)


async def cooperative_thread(fn, *args, **kwargs):
    """Do not abandon a thread that may still write/checkpoint after cancellation."""
    check_control()
    task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
    try:
        result = await asyncio.shield(task)
        check_control()
        return result
    except asyncio.CancelledError:
        control = CURRENT_CONTROL.get()
        if control:
            control.stopped.set()
        try:
            await asyncio.shield(task)
        except BaseException:
            pass
        raise
