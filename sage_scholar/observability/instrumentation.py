"""Narrow adapters for our custom model/tool/node boundaries, without payload capture."""
from contextlib import contextmanager
from functools import wraps
import time

from .tracing import GENERATION, TOOL, ACTIVE, span, label


def stage(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            state = args[0] if args and isinstance(args[0], dict) else {}
            task = state.get("task")
            attributes = {"sage.task_id": label(task.task_id)} if task is not None else {}
            with span(name, "agent", attributes=attributes):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


def model_call(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        model = self.model if hasattr(self, "model") else self.config.model
        purpose = label(kwargs.get("tag", "router:classify" if fn.__name__ == "classify" else "model"))
        with span("LLM." + purpose, "generation", attributes={
            "gen_ai.request.model": label(model), "langfuse.observation.model.name": label(model),
            "sage.usage.source": "unknown", "sage.cost.source": "unknown",
            "sage.purpose": purpose, "sage.retry_count.source": "unreported",
        }) as observation:
            if fn.__name__ == "classify":
                observation.set(**{"sage.retry_count": 0, "sage.retry_count.source": "httpx_no_retries"})
            token = GENERATION.set(observation)
            try:
                return fn(self, *args, **kwargs)
            finally:
                GENERATION.reset(token)
    return wrapped


def provider_request(request):
    """SDK request hook observes actual automatic retries without changing retry policy."""
    observation = GENERATION.get()
    if observation is not None:
        try:
            retries = int(request.headers.get("x-stainless-retry-count", "0"))
            observation.set(**{"sage.retry_count": retries, "sage.retry_count.source": "sdk_request_header"})
            request.extensions["sage_observation_started"] = time.monotonic()
        except Exception:
            pass


def provider_response(response):
    observation = GENERATION.get()
    if observation is not None:
        try:
            attributes = {"http.response.status_code": response.status_code}
            started = response.request.extensions.get("sage_observation_started")
            if started is not None:
                attributes["sage.http.response_headers_seconds"] = time.monotonic() - started
            observation.event("provider.response", attributes)
        except Exception:
            pass


def first_token(seconds):
    observation = GENERATION.get()
    if observation is not None:
        observation.set(**{"sage.llm.ttft_seconds": max(0, seconds), "sage.llm.ttft.source": "first_text_delta"})


@contextmanager
def tool_span(name, *, retry_count=0):
    with span("Tool." + label(name), "tool", attributes={"gen_ai.tool.name": label(name),
              "sage.tool.execution_count": 1, "sage.retry_count": retry_count,
              "sage.retry_count.source": "consecutive_tool_failures"}) as observation:
        token = TOOL.set(observation)
        try:
            yield observation
        finally:
            TOOL.reset(token)


def execute_tool(tool, parameters, *, retry_count=0):
    from sage_scholar.tools.results import is_error_result
    with tool_span(tool.name, retry_count=retry_count) as observation:
        result = tool.run_tool(parameters)
        # No arguments, captions, URLs, or source text leave the process.
        if isinstance(result, str):
            observation.set(**{"sage.tool.result_chars": len(result)})
            if is_error_result(result):
                observation.status("failed", "ToolErrorEnvelope")
        return result


def journal_result(*, replayed=False, retry=False):
    observation = TOOL.get()
    if observation is not None:
        observation.set(**{"sage.tool.replayed": replayed, "sage.tool.recovery_retry": retry,
                            "sage.tool.execution_count": 0 if replayed else 1})


def mcp_attempt(attempt, *, timed_out=False):
    for observation in (TOOL.get(), ACTIVE.get()):
        if observation is not None:
            observation.set(**{"sage.mcp.retry_count": attempt})
            if timed_out:
                observation.status("timeout", "TimeoutError")


def failure(exc):
    # For nodes that intentionally recover from an error before returning a partial result.
    from .tracing import CURRENT, error_category
    if CURRENT.get() is not None:
        with span("Recovered.Failure", attributes={"error.type": label(type(exc).__name__)}) as observation:
            observation.status(error_category(exc), type(exc).__name__)
