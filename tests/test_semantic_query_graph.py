"""Local classifier boundary + real SQLite/LangGraph lifecycle; no paid APIs."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from sage_scholar.application.query_service import QueryService, QueryCommand
from sage_scholar.application.qa_service import QAResult
from sage_scholar.config import RouterConfig
from sage_scholar.domain.execution import RunOutcome
from sage_scholar.routing import QueryRouter, Workflow, SourceScope
from sage_scholar.routing.classifier import LocalClassifier, SemanticIntent, IntentPrediction, RouterUnavailable
from sage_scholar.runtime import RunManager, RunStatus
from sage_scholar.runtime.checkpoints import Checkpoints
from sage_scholar.runtime.store import RunStore
from sage_scholar.runtime.human import HumanResponse


def intent(workflow="direct", **kwargs):
    return SemanticIntent(workflow=workflow, source_scope="conversation", confidence=.98, **kwargs)


class Classifier:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def classify(self, payload):
        self.calls.append(payload)
        value = self.answers.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def setup(path, classifier, research=None, docs=None):
    store, checkpoints = RunStore(path), Checkpoints(path)
    manager = RunManager(store=store, checkpoints=checkpoints)
    calls = []
    class Direct:
        async def answer(self, request):
            calls.append(request)
            return "direct answer"
    class QA:
        async def answer(self, request):
            calls.append(request)
            return QAResult("qa answer", [])
    service = QueryService(QueryRouter(classifier), Direct(), QA(), manager,
        {Workflow.DEEP_RESEARCH: research} if research else {}, document_resolver=docs)
    async def close():
        await manager.shutdown()
        checkpoints.close(); store.close()
    return SimpleNamespace(store=store, checkpoints=checkpoints, manager=manager, service=service, calls=calls, close=close)


def test_local_ollama_schema_no_cloud_or_proxy():
    seen = []
    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": IntentPrediction(workflow="direct",source_intent="conversation",confidence=.98).model_dump_json()}})
    client = LocalClassifier(RouterConfig(), transport=httpx.MockTransport(respond))
    assert client.classify({"query": "不要调研，只解释概念"}).workflow == "direct"
    assert seen[0]["think"] is False and seen[0]["format"]["type"] == "object"
    assert seen[0]["model"] == "qwen3.5:2b"
    with pytest.raises(ValueError, match="本机"):
        LocalClassifier(RouterConfig(base_url="https://cloud.example"))
    with pytest.raises(RouterUnavailable):
        LocalClassifier(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"message": {"content": "not JSON"}}))).classify({})


def test_semantics_and_source_boundaries():
    classifier = Classifier(intent(), SemanticIntent(workflow="deep_research", source_scope="mixed", allow_web=True, confidence=.99))
    router = QueryRouter(classifier)
    assert router.route("不要调研，只解释", document_ids=["paper"]).workflow == Workflow.DIRECT
    route = router.route("结合联网", requested_scope=SourceScope.SELECTED_DOCUMENTS,
                         document_ids=["paper"], allow_web=False)
    assert not route.allow_web and route.document_ids == ["paper"]
    assert route.source_scope == SourceScope.SELECTED_DOCUMENTS


def test_direct_runs_inside_graph_and_persists_route(tmp_path):
    async def scenario():
        classifier = Classifier(intent())
        c = setup(tmp_path, classifier)
        run = await c.service.execute(QueryCommand(query="这句话包含调研但只需要解释"))
        await c.manager._tasks[run.run_id]
        snapshot = c.manager.get(run.run_id)
        assert snapshot.status == RunStatus.COMPLETED, snapshot.error
        assert snapshot.workflow == Workflow.DIRECT and snapshot.result["answer"] == "direct answer"
        values = c.checkpoints.saver.get_tuple(c.checkpoints.config(run.run_id)).checkpoint["channel_values"]
        assert values["route"].workflow == Workflow.DIRECT and values["outcome"].result["answer"]
        assert len(classifier.calls) == len(c.calls) == 1
        assert c.checkpoints.compatible(snapshot)
        assert not c.store.evidence_ids(run.run_id)
        await c.close()
    asyncio.run(scenario())


def test_qa_branch_uses_existing_service_and_policy(tmp_path):
    async def scenario():
        classifier = Classifier(SemanticIntent(workflow="qa", source_scope="selected_documents",
                                               document_ids=["a"], confidence=.99))
        c = setup(tmp_path, classifier)
        result = await c.service.execute(QueryCommand(query="实验呢", document_ids=["a"], allow_web=False))
        await c.manager._tasks[result.run_id]
        run = c.manager.get(result.run_id)
        assert run.status == RunStatus.COMPLETED, run.error
        assert run.result["answer"] == "qa answer" and c.calls[0].document_ids == ("a",)
        assert run.execution.budget["max_total_tokens"] == 300000
        assert not c.calls[0].source_policy.permits_web
        await c.close()
    asyncio.run(scenario())


def test_clarify_restart_resume_same_run_and_no_double_answer(tmp_path):
    async def scenario():
        classifier = Classifier(intent("clarify", clarification="你指的是哪个对象？"))
        c = setup(tmp_path, classifier)
        conversation = c.store.conversations.create().conversation_id
        result = await c.service.execute(QueryCommand(query="它怎么样", conversation_id=conversation))
        await c.manager._tasks[result.run_id]
        run = c.manager.get(result.run_id)
        assert run.status == RunStatus.WAITING_FOR_USER, run.error
        question = run.interrupts[0]
        await c.close()
        restored = Classifier(intent())
        c = setup(tmp_path, restored)
        await c.manager.recover()
        assert c.manager.get(run.run_id).status == RunStatus.WAITING_FOR_USER
        response = HumanResponse(run_id=run.run_id, conversation_id=conversation,
                                 interrupt_id=question["interrupt_id"], answer="只解释 KL 散度")
        await c.manager.answer(run.run_id, response)
        await c.manager._tasks[run.run_id]
        finished = c.manager.get(run.run_id)
        assert finished.status == RunStatus.COMPLETED, finished.error
        assert "只解释 KL" in c.calls[0].query and len(restored.calls) == 1
        with pytest.raises(ValueError):
            await c.manager.answer(run.run_id, response)
        await c.close()
    asyncio.run(scenario())


def test_router_failure_is_visible_and_resume_retries_only_routing(tmp_path):
    async def scenario():
        classifier = Classifier(RouterUnavailable("本地模型未启动"), intent())
        c = setup(tmp_path, classifier)
        value = await c.service.execute(QueryCommand(query="hello"))
        await c.manager._tasks[value.run_id]
        run = c.manager.get(value.run_id)
        assert run.status == RunStatus.FAILED and not c.calls
        await c.manager.resume(run.run_id)
        await c.manager._tasks[run.run_id]
        assert c.manager.get(run.run_id).status == RunStatus.COMPLETED
        assert len(classifier.calls) == 2
        await c.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("human", [False, True])
def test_research_graph_nested_with_private_checkpoints(monkeypatch, tmp_path, library, human):
    from test_research_runtime import build_runtime
    from sage_scholar.application.research_service import ResearchService
    _, adapter, a, _, _, _ = library
    runtime, _ = build_runtime(monkeypatch, tmp_path, adapter, SourceScope.SELECTED_DOCUMENTS, [a.document_id])
    monkeypatch.setattr("sage_scholar.research.runtime.assess", lambda *args: {"question": "先比较理论还是实验？"} if human else None)
    async def scenario():
        classifier = Classifier(SemanticIntent(workflow="deep_research", source_scope="selected_documents", confidence=.99))
        provider = SimpleNamespace(get=lambda require_external: runtime, checkpoints=True)
        c = setup(tmp_path, classifier, ResearchService(provider))
        runtime.store, runtime.checkpoints = c.store, c.checkpoints
        conversation = c.store.conversations.create().conversation_id
        result = await c.service.execute(QueryCommand(query="比较方法", document_ids=[a.document_id], conversation_id=conversation))
        await c.manager._tasks[result.run_id]
        run = c.manager.get(result.run_id)
        if human:
            assert run.status == RunStatus.WAITING_FOR_USER, run.error
            question = run.interrupts[0]
            await c.close()
            c = setup(tmp_path, classifier, ResearchService(provider))
            runtime.store, runtime.checkpoints = c.store, c.checkpoints
            await c.manager.recover()
            await c.manager.answer(run.run_id, HumanResponse(run_id=run.run_id, conversation_id=conversation,
                interrupt_id=question["interrupt_id"], answer="先理论"))
            await c.manager._tasks[run.run_id]
            run = c.manager.get(run.run_id)
        assert run.status == RunStatus.COMPLETED, run.error
        assert run.workflow == Workflow.DEEP_RESEARCH and run.result["report"]
        child = c.checkpoints.saver.get_tuple(c.checkpoints.research_config(run.execution))
        assert child.checkpoint["channel_values"]["research_notes"]
        root = c.checkpoints.saver.get_tuple(c.checkpoints.config(run.run_id))
        assert root.checkpoint["channel_values"]["outcome"]
        assert len(classifier.calls) == 1
        await c.close()
    asyncio.run(scenario())


# Shared lightweight document fixture; no embedding downloads.
from test_document_evidence import library


@pytest.mark.parametrize("workflow", ["direct", "qa"])
def test_cancellation_reaches_async_service_and_waits_for_cleanup(tmp_path, workflow):
    async def scenario():
        from router_fakes import FixedClassifier
        c = setup(tmp_path, FixedClassifier(workflow))
        entered, cleaned = asyncio.Event(), asyncio.Event()
        async def slow(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(.01)
                cleaned.set()
        service = c.service.graph_runner.direct if workflow == "direct" else c.service.graph_runner.qa
        service.answer = slow
        result = await c.service.execute(QueryCommand(query="test", document_ids=["paper"] if workflow == "qa" else []))
        await asyncio.wait_for(entered.wait(), 3)
        await c.manager.cancel(result.run_id)
        await asyncio.wait_for(asyncio.gather(*list(c.manager._tasks.values())), 3)
        assert cleaned.is_set()
        assert c.manager.get(result.run_id).status == RunStatus.CANCELLED
        await c.close()
    asyncio.run(scenario())


def test_continue_research_does_not_reroute_or_repeat_completed_notes(monkeypatch, tmp_path, library):
    from test_research_runtime import build_runtime
    from sage_scholar.application.research_service import ResearchService
    _, adapter, a, _, _, _ = library
    runtime, _ = build_runtime(monkeypatch, tmp_path, adapter, SourceScope.SELECTED_DOCUMENTS, [a.document_id], approved=None)
    monkeypatch.setattr("sage_scholar.research.runtime.assess", lambda *args: None)
    async def scenario():
        from router_fakes import FixedClassifier
        classifier = FixedClassifier("deep_research")
        c = setup(tmp_path, classifier, ResearchService(SimpleNamespace(get=lambda require_external: runtime, checkpoints=True)))
        runtime.store, runtime.checkpoints = c.store, c.checkpoints
        result = await c.service.execute(QueryCommand(query="比较实验", document_ids=[a.document_id], budget={"max_rounds": 0}))
        await c.manager._tasks[result.run_id]
        run = c.manager.get(result.run_id)
        assert run.status == RunStatus.PARTIAL, run.error
        config = c.checkpoints.research_config(run.execution)
        first = c.checkpoints.saver.get_tuple(config).checkpoint["channel_values"]["research_notes"]["T1"]
        c.store.record_usage(run.run_id, 123, 12)
        await c.close()
        c = setup(tmp_path, classifier, ResearchService(SimpleNamespace(get=lambda require_external: runtime, checkpoints=True)))
        runtime.store, runtime.checkpoints = c.store, c.checkpoints
        await c.manager.resume(run.run_id)
        await c.manager._tasks[run.run_id]
        finished = c.manager.get(run.run_id)
        assert finished.status == RunStatus.COMPLETED, finished.error
        notes = c.checkpoints.saver.get_tuple(config).checkpoint["channel_values"]["research_notes"]
        assert notes["T1"] == first and set(notes) == {"T1", "T2"}
        assert len(classifier.calls) == 1
        assert finished.consumed_budget["prompt_tokens"] >= 123
        await c.close()
    asyncio.run(scenario())


def test_mentioned_document_is_not_automatically_selected(tmp_path):
    from sage_scholar.application.document_resolver import DocumentResolution
    class Documents:
        async def catalog(self):
            return [{"document_id": "A", "title": "Paper A", "ready": True}]
        async def resolve(self, query, selected):
            assert selected, "Free-text title matching must not create a hard source constraint"
            return DocumentResolution(document_ids=selected)
    async def scenario():
        class Research:
            async def run(self, request, emit, cancelled):
                assert request.document_ids == () and request.source_policy.permits_web
                return RunOutcome(result={"report": "report"})
        classifier = Classifier(SemanticIntent(workflow="deep_research", source_scope="public", allow_web=True, confidence=.99))
        c = setup(tmp_path, classifier, Research(), docs=Documents())
        result = await c.service.execute(QueryCommand(query="不要用 Paper A，重新调研外部资料"))
        await c.manager._tasks[result.run_id]
        run = c.manager.get(result.run_id)
        assert run.status == RunStatus.COMPLETED, run.error
        assert classifier.calls[0]["selected_document_ids"] == []
        assert classifier.calls[0]["catalog"][0]["document_id"] == "A"
        await c.close()
    asyncio.run(scenario())


def test_ignored_stale_ui_selection_does_not_block_public_research(tmp_path):
    from sage_scholar.application.document_resolver import DocumentResolution
    class Documents:
        async def catalog(self):return []
        async def resolve(self,query,selected):
            return DocumentResolution(message='所选论文已不存在')
    async def scenario():
        class Research:
            async def run(self,request,emit,cancelled):
                assert request.document_ids==() and request.source_policy.permits_web
                return RunOutcome(result={'report':'public report'})
        classifier=Classifier(SemanticIntent(workflow='deep_research',source_scope='public',allow_web=True,confidence=.9))
        c=setup(tmp_path,classifier,Research(),docs=Documents())
        response=await c.service.execute(QueryCommand(query='忽略选中项，独立公网研究',document_ids=['deleted']))
        await c.manager._tasks[response.run_id]
        run=c.manager.get(response.run_id)
        assert run.status==RunStatus.COMPLETED,run.error
        assert run.result['report']=='public report'
        await c.close()
    asyncio.run(scenario())
