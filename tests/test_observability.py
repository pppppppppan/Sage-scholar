"""Real OTel SDK + real graphs/checkpoints, in-memory exporter, no external services."""
import asyncio
from contextlib import contextmanager
import json
import threading
from types import SimpleNamespace

import httpx
import pytest
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sage_scholar.observability import Observer, span, usage, model_call, execute_tool
from sage_scholar.observability.config import ObservabilityConfig
from sage_scholar.observability.tracing import CURRENT, GENERATION
from sage_scholar.runtime.human import HumanResponse
from sage_scholar.application.query_service import QueryCommand
from sage_scholar.runtime.models import RunStatus
from test_semantic_query_graph import setup, Classifier, intent


@pytest.fixture
def telemetry():
    exporter = InMemorySpanExporter()
    observer = Observer(ObservabilityConfig(enabled=True, timeout_seconds=1), exporter=exporter)
    assert observer.provider is not None
    yield observer, exporter
    observer.provider.shutdown()


@contextmanager
def active(observer):
    token = CURRENT.set(observer)
    try:
        with span("test.root"):
            yield
    finally:
        CURRENT.reset(token)


def exported(telemetry):
    observer, exporter = telemetry
    assert observer.flush()
    return exporter.get_finished_spans()


def test_disabled_and_bad_configuration_never_build_exporter(monkeypatch):
    from sage_scholar.observability import backend
    monkeypatch.setattr(backend, "build_provider", lambda *a: pytest.fail("disabled exporter initialized"))
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")
    monkeypatch.setenv("OBSERVABILITY_MODEL_PRICES_JSON", "invalid")
    assert Observer.from_env().tracer is None
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    assert Observer.from_env().tracer is None


@pytest.mark.parametrize("workflow", ["direct", "qa", "deep_research"])
def test_real_entry_graph_routes_trace_and_sse(tmp_path, telemetry, workflow):
    from sage_scholar.routing.classifier import SemanticIntent
    from sage_scholar.domain.execution import RunOutcome
    class Docs:
        async def catalog(self):
            return [{"document_id": "d", "title": "PRIVATE_PAPER", "ready": True}]
        async def resolve(self, *args):
            return SimpleNamespace(message=None)
    class Research:
        async def run(self, request, emit, cancelled):
            return RunOutcome(result={"report": "PRIVATE_RESULT"})
    async def scenario():
        scope = {"direct": "conversation", "qa": "selected_documents", "deep_research": "public"}[workflow]
        classifier = Classifier(SemanticIntent(workflow=workflow, source_scope=scope, confidence=.9))
        c = setup(tmp_path, classifier, Research(), Docs())
        c.manager.observability = telemetry[0]
        initial = await c.service.execute(QueryCommand(query="PRIVATE_QUERY", document_ids=["d"]))
        await c.manager._tasks[initial.run_id]
        run = c.manager.get(initial.run_id)
        assert run.status == RunStatus.COMPLETED
        events = [e async for e in c.manager.events(run.run_id)]
        assert events[0].type == "run_created" and events[-1].type == "run_finished"
        assert run.execution.metadata["_observability"]["attempt"] == 1
        spans = exported(telemetry)
        root = next(s for s in spans if s.name == "Agent.Run")
        branch = next(s for s in spans if s.name == workflow)
        assert branch.parent.span_id == root.context.span_id
        assert {s.context.trace_id for s in spans} == {root.context.trace_id}
        assert root.attributes["sage.status"] == "completed"
        assert root.attributes["sage.workflow"] == workflow
        assert root.attributes["sage.run.wall_seconds"] >= 0
        assert all("PRIVATE_" not in str(s.attributes) for s in spans)
        await c.close()
    asyncio.run(scenario())


def test_parallel_graph_and_thread_context_isolation(telemetry):
    from research_fakes import FakeSupervisor, FakeResearcher, FakeWriter, research_request
    from sage_scholar.domain.research import SubQuestion
    from sage_scholar.graph.graph_builder import build_graph
    barrier = threading.Barrier(2, timeout=5)
    class Model:
        model = "fake-model"
        @model_call
        def invoke(self, **kwargs):
            usage(self.model, 20, 5)
    model = Model()
    class Supervisor(FakeSupervisor):
        def plan(self, request):
            return [SubQuestion(objective="methods"), SubQuestion(objective="experiments")]
    class Researcher(FakeResearcher):
        def run(self, task, request):
            barrier.wait()
            model.invoke(tag=task.task_id + ":react")
            return super().run(task, request)
    async def scenario():
        with active(telemetry[0]):
            graph = build_graph(Supervisor(), lambda _: Researcher(), FakeWriter())
            result = await asyncio.to_thread(graph.invoke, {"execution": research_request()})
            assert len(result["research_notes"]) == 2
    asyncio.run(scenario())
    spans = exported(telemetry)
    researchers = {s.attributes["sage.task_id"]: s for s in spans if s.name == "Researcher"}
    for name in ("T1", "T2"):
        child = next(s for s in spans if s.name == "LLM." + name + ":react")
        assert child.parent.span_id == researchers[name].context.span_id
    assert len({s.context.trace_id for s in spans}) == 1
    assert {"Supervisor", "Supervisor.Review", "Writer"} <= {s.name for s in spans}
    assert sum(s.attributes.get("sage.usage.total_tokens", 0) for s in spans) == 50


def test_clarify_restart_multiple_resumes_and_waiting_cancel(tmp_path, telemetry):
    async def scenario():
        c = setup(tmp_path, Classifier(intent("clarify", clarification="topic?")))
        c.manager.observability = telemetry[0]
        conversation = c.store.conversations.create().conversation_id
        first = await c.service.execute(QueryCommand(query="PRIVATE", conversation_id=conversation))
        await c.manager._tasks[first.run_id]
        initial = c.manager.get(first.run_id)
        association = initial.execution.metadata["_observability"]
        assert initial.status == RunStatus.WAITING_FOR_USER
        await c.close()
        # New provider represents a restarted process; only persisted IDs cross this boundary.
        second_exporter = InMemorySpanExporter()
        restarted = Observer(ObservabilityConfig(enabled=True), exporter=second_exporter)
        c = setup(tmp_path, Classifier(intent("clarify", clarification="which paper?"), intent()))
        c.manager.observability = restarted
        await c.manager.recover()
        for answer in ("PRIVATE_TOPIC", "PRIVATE_PAPER"):
            run = c.manager.get(first.run_id)
            await c.manager.answer(run.run_id, HumanResponse(run_id=run.run_id, conversation_id=conversation,
                interrupt_id=run.interrupts[0]["interrupt_id"], answer=answer))
            await c.manager._tasks[run.run_id]
        final = c.manager.get(first.run_id)
        assert final.status == RunStatus.COMPLETED
        assert final.execution.metadata["_observability"]["attempt"] == 3
        restarted.flush()
        resumes = [s for s in second_exporter.get_finished_spans() if s.name == "Agent.Run.Resume"]
        assert len(resumes) == 2
        assert all(f"{s.context.trace_id:032x}" == association["trace_id"] for s in resumes)
        assert all(f"{s.parent.span_id:016x}" == association["root_span_id"] for s in resumes)
        assert resumes[-1].links[0].context.span_id == resumes[0].context.span_id
        await c.close()
        await restarted.close()
    asyncio.run(scenario())


def test_cancel_one_run_and_exporter_failure_do_not_break_sse(tmp_path):
    class BrokenExporter(SpanExporter):
        def export(self, spans):
            raise RuntimeError("secret exporter error")
    observer = Observer(ObservabilityConfig(enabled=True, timeout_seconds=.1), exporter=BrokenExporter())
    from sage_scholar.runtime import RunManager
    from research_fakes import research_request
    from sage_scholar.domain.execution import RunOutcome
    started = asyncio.Event()
    class Runner:
        async def run(self, request, emit, cancelled):
            if request.query == "slow":
                started.set()
                await asyncio.sleep(60)
            return RunOutcome(result={"answer": "ok"})
    async def scenario():
        manager = RunManager(observability=observer)
        slow = await manager.start(research_request().model_copy(update={"query": "slow"}), Runner())
        await started.wait()
        fast = await manager.start(research_request(), Runner())
        await manager.cancel(slow.run_id)
        await asyncio.gather(*list(manager._tasks.values()))
        assert manager.get(slow.run_id).status == RunStatus.CANCELLED
        assert manager.get(fast.run_id).status == RunStatus.COMPLETED
        assert [e.type async for e in manager.events(slow.run_id)][-1] == "run_cancelled"
        observer.flush()  # Broken async export cannot change authoritative state.
        await manager.shutdown()
        await observer.close()
    asyncio.run(scenario())


def test_tool_retries_timeout_envelopes_and_journal_replay(tmp_path, telemetry):
    from sage_scholar.runtime.store import RunStore
    from sage_scholar.runtime.journal import JournaledTool
    from sage_scholar.tools.base_tool import BaseTool, ToolCallError
    from sage_scholar.observability.instrumentation import mcp_attempt
    class Tool(BaseTool):
        source_access = "local"
        replay_policy = "safe_retry"
        def __init__(self):
            super().__init__("test_tool", "", [])
            self.calls = 0
        def run_tool(self, parameters):
            self.calls += 1
            if parameters.get("timeout"):
                mcp_attempt(2, timed_out=True)
                raise ToolCallError("secret timeout body")
            if parameters.get("error"):
                return '{"error":"secret tool payload"}'
            return "private source content"
    store, tool = RunStore(tmp_path), Tool()
    with active(telemetry[0]):
        durable = JournaledTool(tool, store, "r", "t")
        assert execute_tool(durable, {}) == execute_tool(durable, {})
        with pytest.raises(ToolCallError):
            execute_tool(tool, {"timeout": True}, retry_count=1)
        execute_tool(tool, {"error": True})
    store.close()
    tools = [s for s in exported(telemetry) if s.name == "Tool.test_tool"]
    assert tool.calls == 3 and len(tools) == 4
    assert sum(s.attributes["sage.tool.execution_count"] for s in tools) == 3
    assert any(s.attributes.get("sage.tool.replayed") for s in tools)
    timed = next(s for s in tools if s.attributes["sage.status"] == "timeout")
    assert timed.attributes["sage.mcp.retry_count"] == 2 and timed.attributes["sage.retry_count"] == 1
    assert tools[-1].attributes["sage.status"] == "failed"
    assert all("secret" not in str(s.attributes) and not s.events for s in tools)


def test_usage_cost_exclusive_cache_and_unknowns(telemetry):
    observer = telemetry[0]
    observer.config.prices.update({"priced": {"input": 2, "output": 4, "cache_read": .2}})
    class Model:
        model = "priced"
        @model_call
        def invoke(self, **kwargs):
            usage(self.model, 100, 20, cached=40)
            usage(self.model, 100, 20, cached=40)  # Duplicate callback must not double-charge.
    with active(observer):
        model = Model()
        model.invoke()
        model.model = "no-price"
        model.invoke()
        with span("unknown", "generation") as observation:
            token = GENERATION.set(observation)
            usage("no-price", None, None)
            GENERATION.reset(token)
    spans = exported(telemetry)
    generations = [s for s in spans if s.attributes.get("langfuse.observation.type") == "generation"]
    details = json.loads(generations[0].attributes["langfuse.observation.usage_details"])
    assert details == {"input": 60, "output": 20, "cache_read": 40}
    costs = json.loads(generations[0].attributes["langfuse.observation.cost_details"])
    assert sum(costs.values()) == pytest.approx(.000208)
    assert generations[0].attributes["sage.cost.source"] == "configured_estimate"
    assert "langfuse.observation.cost_details" not in generations[1].attributes
    assert "langfuse.observation.usage_details" not in generations[2].attributes
    assert generations[2].attributes["sage.usage.source"] == "unknown"
    assert sum(s.attributes.get("sage.usage.total_tokens", 0) for s in spans) == 240


def test_real_llm_sdk_retry_usage_and_privacy(tmp_path, telemetry):
    from openai import OpenAI
    from sage_scholar.base.llm_client import LLMClient
    from sage_scholar.observability.instrumentation import provider_request, provider_response
    attempts = []
    def respond(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, json={"error": {"message": "SECRET_PROVIDER_MESSAGE"}}, headers={"retry-after-ms": "1"})
        return httpx.Response(200, json={"id": "id", "object": "chat.completion", "created": 1, "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "PRIVATE_ANSWER"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35}})
    class Context:
        def prepare(self, messages, **kwargs):
            return SimpleNamespace(messages=messages, accounting={"total_tokens": 50, "input_tokens": 30})
    client = LLMClient(model="test-model", api_key="SECRET_KEY", base_url="https://test.invalid/v1", context_manager=Context())
    client.client.close()
    client.client = OpenAI(api_key="SECRET_KEY", base_url="https://test.invalid/v1", max_retries=1,
        http_client=httpx.Client(transport=httpx.MockTransport(respond), event_hooks={"request": [provider_request], "response": [provider_response]}))
    with active(telemetry[0]):
        assert client.invoke([{"role": "user", "content": "PRIVATE_PROMPT"}], max_tokens=10).content == "PRIVATE_ANSWER"
    client.client.close()
    generation = next(s for s in exported(telemetry) if s.attributes.get("langfuse.observation.type") == "generation")
    assert len(attempts) == 2
    assert generation.attributes["sage.retry_count"] == 1
    assert generation.attributes["sage.usage.total_tokens"] == 35
    assert generation.attributes["sage.usage.source"] == "provider"
    assert generation.attributes["sage.llm.provider_seconds"] >= 0
    assert [e.attributes["http.response.status_code"] for e in generation.events] == [429, 200]
    assert all(x not in str(generation.attributes) + str(generation.events) for x in ["PRIVATE_", "SECRET_", "test.invalid"])


def test_local_classifier_usage_and_stream_estimation(telemetry):
    from sage_scholar.routing.classifier import LocalClassifier, IntentPrediction
    from sage_scholar.runtime.streaming import openai_text
    value = IntentPrediction(workflow="direct", source_intent="conversation", confidence=.9)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "message": {"content": value.model_dump_json()}, "prompt_eval_count": 12, "eval_count": 3}))
    class Stream:
        def __iter__(self):
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content="private answer"))])
        def close(self):
            pass
    with active(telemetry[0]):
        LocalClassifier(transport=transport).classify({"query": "private query"})
        with span("stream", "generation") as observation:
            token = GENERATION.set(observation)
            def track(prompt, completion, elapsed, tag, msg, **kwargs):
                usage("test-model", prompt, completion, source=kwargs["usage_source"])
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: Stream())))
            control = SimpleNamespace(check=lambda: None, emit=lambda *args: None)
            assert openai_text(client, {"messages": [{"role": "user", "content": "private"}]}, control, track, "direct").content
            GENERATION.reset(token)
    spans = exported(telemetry)
    router = next(s for s in spans if s.name == "LLM.router:classify")
    assert router.attributes["sage.usage.total_tokens"] == 15
    stream = next(s for s in spans if s.name == "stream")
    assert stream.attributes["sage.usage.source"] == "estimated"
    assert stream.attributes["sage.llm.ttft_seconds"] >= 0


def test_failed_run_and_waiting_cancel_have_authoritative_status(tmp_path, telemetry):
    async def scenario():
        c = setup(tmp_path, Classifier(TimeoutError("SECRET_REQUEST"), intent("clarify", clarification="PRIVATE_QUESTION")))
        c.manager.observability = telemetry[0]
        first = await c.service.execute(QueryCommand(query="private"))
        await c.manager._tasks[first.run_id]
        assert c.manager.get(first.run_id).status == RunStatus.FAILED
        second = await c.service.execute(QueryCommand(query="private"))
        await c.manager._tasks[second.run_id]
        assert c.manager.get(second.run_id).status == RunStatus.WAITING_FOR_USER
        await c.manager.cancel(second.run_id)
        assert c.manager.get(second.run_id).status == RunStatus.CANCELLED
        spans = exported(telemetry)
        failed = next(s for s in spans if s.name == "Agent.Run" and s.attributes["sage.run_id"] == first.run_id)
        assert failed.attributes["sage.status"] == "failed"
        cancelled = next(s for s in spans if s.name == "Lifecycle.run_cancelled")
        assert cancelled.attributes["sage.status"] == "cancelled"
        assert any(s.attributes.get("sage.status") == "timeout" for s in spans)
        assert all("SECRET_" not in str(s.attributes) + str(s.events) for s in spans)
        await c.close()
    asyncio.run(scenario())


def test_queue_is_bounded_and_slow_exporter_does_not_block_agent():
    entered, release = threading.Event(), threading.Event()
    class SlowExporter(SpanExporter):
        def export(self, spans):
            entered.set()
            release.wait(3)
            return SpanExportResult.SUCCESS
    observer = Observer(ObservabilityConfig(enabled=True, queue_size=4, batch_size=1, delay_ms=100), exporter=SlowExporter())
    try:
        with active(observer):
            with span("first"):
                pass
            assert entered.wait(2)
            # Export is blocked. All business spans still complete and memory stays bounded.
            for _ in range(30):
                with span("work"):
                    pass
        processor = observer.provider._active_span_processor._span_processors[0]
        assert len(processor._batch_processor._queue) <= 4
    finally:
        release.set()
        observer.provider.shutdown()


def test_broken_span_backend_and_corrupt_association_cannot_stop_run(tmp_path, telemetry):
    class BrokenTracer:
        def start_as_current_span(self, *args, **kwargs):
            raise RuntimeError("collector credentials or internal failure")
    async def scenario():
        c = setup(tmp_path, Classifier(intent()))
        c.manager.observability = telemetry[0]
        telemetry[0].tracer = BrokenTracer()
        run = await c.service.execute(QueryCommand(query="private", metadata={"_observability": ["malformed"]}))
        await c.manager._tasks[run.run_id]
        assert c.manager.get(run.run_id).status == RunStatus.COMPLETED
        assert c.calls
        await c.close()
    asyncio.run(scenario())


def test_shutdown_does_not_join_an_unbounded_exporter():
    import time
    release = threading.Event()
    class SlowProvider:
        def shutdown(self):
            release.wait(5)
    observer = Observer(ObservabilityConfig(timeout_seconds=.01))
    observer.provider = SlowProvider()
    started = time.monotonic()
    try:
        asyncio.run(observer.close())
        assert time.monotonic() - started < 2.5
    finally:
        release.set()


def test_langfuse_base_url_precedence_and_legacy_host(monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("OBSERVABILITY_MODEL_PRICES_JSON", "{}")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    monkeypatch.setenv("LANGFUSE_BASE_URL", " https://jp.cloud.langfuse.com/ ")
    assert ObservabilityConfig.from_env().host == "https://jp.cloud.langfuse.com"
    monkeypatch.setenv("LANGFUSE_BASE_URL", " ")
    assert ObservabilityConfig.from_env().host == "https://cloud.langfuse.com"
    monkeypatch.delenv("LANGFUSE_BASE_URL")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3001/")
    assert ObservabilityConfig.from_env().host == "http://localhost:3001"
