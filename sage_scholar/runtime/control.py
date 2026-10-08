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


class RunControl:
    def __init__(self, request, store, cancelled, emit=None):
        self.request, self.store, self.cancelled, self.emit = request, store, cancelled, emit
        self.lock = threading.RLock()
        self.last_meter = time.monotonic()
        self.elapsed = 0.0
        self.stopped = threading.Event()
        self.budget_error = None

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
        if self.budget_error or elapsed >= self.request.budget.get('max_seconds', float('inf')):
            raise RunBudgetExceeded(self.budget_error or '运行时间预算已耗尽，已有进度已保存')

    def reserve_model(self, tokens):
        self.check()
        if not self.store:
            return None
        lease = self.store.reserve_model(self.request.run_id, tokens, self.request.budget.get('max_total_tokens', 10_000_000))
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
