"""Safe structured spans. OTel is optional and never owns business state."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import re
import threading

from .config import ObservabilityConfig

logger = logging.getLogger(__name__)
CURRENT = ContextVar("sage_observability", default=None)
GENERATION = ContextVar("sage_generation", default=None)
TOOL = ContextVar("sage_tool", default=None)
ACTIVE = ContextVar("sage_observation", default=None)


def label(value):
    text = str(value)
    return text if re.fullmatch(r"[\w.:/-]{1,100}", text, re.ASCII) else "redacted"


def error_category(exc):
    # Never export exception text/stack: provider errors can contain prompts/keys.
    names = {type(exc).__name__}
    cause = exc.__cause__
    for _ in range(3):
        if cause is None:
            break
        names.add(type(cause).__name__)
        cause = cause.__cause__
    if any("Timeout" in n for n in names):
        return "timeout"
    if names & {"CancelledError", "WorkflowInterrupted"}:
        return "interrupted"
    if names & {"RunWaiting", "GraphInterrupt", "GraphBubbleUp"}:
        return "waiting_for_user"
    if names & {"RunBudgetExceeded", "ResearchBudgetReached", "ContextBudgetExceeded"}:
        return "budget_exhausted"
    if any("Connect" in n for n in names):
        return "connection"
    return "error"


class Observation:
    def __init__(self, span=None):
        self.span = span
        self.usage_recorded = False
        self.outcome = "completed"

    def set(self, **attributes):
        if self.span is not None:
            try:
                for key, value in attributes.items():
                    if value is not None:
                        self.span.set_attribute(key, value)
            except Exception:
                pass

    def status(self, value, error_type=None):
        if value == "error" and self.outcome == "timeout":
            return
        self.outcome = value
        self.set(**{"sage.status": value, "error.type": label(error_type) if error_type else None})
        if self.span is not None and value in {"error", "failed", "timeout", "connection"}:
            try:
                from opentelemetry.trace import Status, StatusCode
                self.span.set_status(Status(StatusCode.ERROR))
                self.set(**{"langfuse.observation.level": "ERROR",
                            "langfuse.observation.status_message": label(error_type or value)})
            except Exception:
                pass

    def event(self, name, attributes):
        if self.span is not None:
            try:
                self.span.add_event(name, attributes)
            except Exception:
                pass


class Observer:
    def __init__(self, config=None, *, exporter=None):
        self.config, self.provider, self.tracer = config or ObservabilityConfig(), None, None
        if self.config.enabled:
            try:
                from .backend import build_provider
                self.provider = build_provider(self.config, exporter)
                self.tracer = self.provider.get_tracer("sage_scholar.observability", "1")
            except Exception:
                logger.warning("Observability disabled: configuration or optional SDK unavailable")

    @classmethod
    def from_env(cls):
        try:
            return cls(ObservabilityConfig.from_env())
        except Exception:
            logger.warning("Observability disabled: invalid configuration")
            return cls()

    @contextmanager
    def span(self, name, kind="span", *, attributes=None, context=None, links=()):
        observation, cm = Observation(), None
        if self.tracer is not None:
            try:
                cm = self.tracer.start_as_current_span(label(name), context=context, links=links,
                    attributes={"langfuse.observation.type": kind, "sage.status": "completed", **(attributes or {})},
                    record_exception=False, set_status_on_exception=False)
                observation = Observation(cm.__enter__())
            except Exception:
                cm = None
        try:
            token = ACTIVE.set(observation)
            try:
                yield observation
            finally:
                ACTIVE.reset(token)
        except BaseException as exc:
            observation.status(error_category(exc), type(exc).__name__)
            raise
        finally:
            if cm:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass

    @contextmanager
    def run(self, state):
        """Same trace after restart, new attempt span, parent/link IDs persisted in SQLite."""
        token = CURRENT.set(self)
        trace_token = None
        try:
            if self.tracer is None:
                yield Observation()
                return
            from opentelemetry import trace
            from opentelemetry.context import Context
            from .backend import TRACE_ID
            snapshot = state.snapshot
            identity = int(hashlib.sha256(("sage-run:" + snapshot.run_id).encode()).hexdigest()[:32], 16) or 1
            trace_token = TRACE_ID.set(identity)
            saved = snapshot.execution.metadata.get("_observability", {})
            parent, links = Context(), []
            try:
                if not isinstance(saved, dict) or saved.get("trace_id") != f"{identity:032x}":
                    saved = {}
                if saved:
                    def reference(span_id):
                        number = int(span_id, 16)
                        if not 0 < number < 2**64:
                            raise ValueError("Invalid span id")
                        return trace.SpanContext(identity, number, True, trace.TraceFlags(1))
                    parent = trace.set_span_in_context(trace.NonRecordingSpan(reference(saved["root_span_id"])), Context())
                    links = [trace.Link(reference(saved["last_span_id"]))]
                    int(saved.get("attempt", 0))
            except (ValueError, KeyError, TypeError):
                saved, parent, links = {}, Context(), []
            attempt = int(saved.get("attempt", 0)) + 1
            with self.span("Agent.Run" if not saved else "Agent.Run.Resume", "agent", context=parent, links=links,
                attributes={"langfuse.trace.name": "Sage Scholar Run", "langfuse.trace.metadata.run_id": snapshot.run_id,
                            "sage.run_id": snapshot.run_id, "sage.attempt": attempt,
                            "sage.workflow": str(snapshot.workflow)}) as observation:
                try:
                    if observation.span is not None:
                        span_id = f"{observation.span.get_span_context().span_id:016x}"
                        association = {"trace_id": f"{identity:032x}", "root_span_id": saved.get("root_span_id", span_id),
                                       "last_span_id": span_id, "attempt": attempt}
                        snapshot.execution = snapshot.execution.model_copy(update={"metadata": {
                            **snapshot.execution.metadata, "_observability": association}})
                except Exception:
                    pass
                try:
                    yield observation
                finally:
                    observation.status(str(snapshot.status))
                    observation.set(**{"sage.workflow": str(snapshot.workflow)})
                    try:
                        created = datetime.fromisoformat(str(snapshot.created_at).replace("Z", "+00:00"))
                        observation.set(**{"sage.run.wall_seconds": max(0, (datetime.now(timezone.utc) - created).total_seconds())})
                    except (ValueError, TypeError):
                        pass
        finally:
            if trace_token is not None:
                TRACE_ID.reset(trace_token)
            CURRENT.reset(token)

    def lifecycle(self, snapshot, event_type):
        """Record actions made while no execution span is live (e.g. cancel during HITL)."""
        if self.tracer is None or CURRENT.get() is self:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.context import Context
            saved = snapshot.execution.metadata.get("_observability", {})
            parent = trace.SpanContext(int(saved["trace_id"], 16), int(saved["root_span_id"], 16), True, trace.TraceFlags(1))
            with self.span("Lifecycle." + label(event_type), context=trace.set_span_in_context(trace.NonRecordingSpan(parent), Context()),
                           attributes={"sage.run_id": snapshot.run_id}) as observation:
                observation.status(str(snapshot.status))
        except Exception:
            pass

    def flush(self):
        try:
            return self.provider.force_flush(timeout_millis=int(self.config.timeout_seconds * 1000)) if self.provider else True
        except Exception:
            return False

    async def close(self):
        if self.provider:
            # SDK shutdown can drain many batches. A daemon avoids leaving a long
            # default-executor job that asyncio would join again at loop shutdown.
            def shutdown():
                try:
                    self.provider.shutdown()
                except Exception:
                    pass
            worker = threading.Thread(target=shutdown, name="sage-trace-shutdown", daemon=True)
            try:
                worker.start()
                await asyncio.to_thread(worker.join, self.config.timeout_seconds + 1)
            except Exception:
                pass


@contextmanager
def span(name, kind="span", **kwargs):
    observer = CURRENT.get()
    if observer is None:
        yield Observation()
    else:
        with observer.span(name, kind, **kwargs) as observation:
            yield observation


def usage(model, input_tokens, output_tokens, *, source="provider", cached=None, cache_write=None, cost_usd=None, elapsed=None):
    """Only generation leaves own usage; parent nodes rely on Langfuse aggregation."""
    observation, observer = GENERATION.get(), CURRENT.get()
    if observation is None or observer is None or observation.usage_recorded:
        return
    try:
        observation.usage_recorded = True
        observation.set(**{"sage.usage.source": source if input_tokens is not None and output_tokens is not None else "unknown", "sage.cost.source": "unknown",
                            "sage.llm.provider_seconds": elapsed})
        if input_tokens is None or output_tokens is None:
            return
        incoming, outgoing = max(0, int(input_tokens)), max(0, int(output_tokens))
        read = min(incoming, max(0, int(cached or 0)))
        write = min(incoming - read, max(0, int(cache_write or 0)))
        # Langfuse flat usage fields must be EXCLUSIVE, cache is part of input.
        details = {"input": incoming - read - write, "output": outgoing}
        if cached is not None:
            details["cache_read"] = read
        if cache_write is not None:
            details["cache_write"] = write
        observation.set(**{"langfuse.observation.usage_details": json.dumps(details),
            "gen_ai.usage.input_tokens": incoming, "gen_ai.usage.output_tokens": outgoing,
            "sage.usage.total_tokens": incoming + outgoing})
        if isinstance(cost_usd, (int, float)) and not isinstance(cost_usd, bool) and math.isfinite(cost_usd) and cost_usd >= 0:
            observation.set(**{"langfuse.observation.cost_details": json.dumps({"total": cost_usd}),
                                "sage.cost.source": "provider"})
        else:
            prices = observer.config.prices.get(model)
            if prices and all(k in prices for k, n in details.items() if n):
                # Discounted cache pricing is only reliable when the provider reports cache usage.
                if "cache_read" in prices and cached is None:
                    return
                costs = {k: n * prices.get(k, 0) / 1_000_000 for k, n in details.items()}
                observation.set(**{"langfuse.observation.cost_details": json.dumps(costs),
                                    "sage.cost.source": "configured_estimate"})
    except Exception:
        pass  # Telemetry must never alter provider results or budget accounting.
