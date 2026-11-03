"""Opt-in, context-local evaluation trace; never attached to ordinary API responses."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import re

_TRACE = ContextVar('router_evaluation_trace', default=None)


def redact(value):
    if isinstance(value, dict):
        return {k: ('[redacted]' if re.search(r'api.?key|password|secret|authorization', k, re.I)
                    else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'(?i)\b(?:sk-[\w-]+|bearer\s+\S+)', '[redacted]', value)
        return re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[email]', value)
    return value


def record(**values):
    trace = _TRACE.get()
    if trace is not None:
        trace.update(redact(deepcopy(values)))


@contextmanager
def capture():
    trace = {}
    token = _TRACE.set(trace)
    try:
        yield trace
    finally:
        _TRACE.reset(token)
