import asyncio
import json
from types import SimpleNamespace

from sage_scholar.application.research_service import ResearchService
from sage_scholar.config import Config
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.research.runtime import ResearchRuntime
from sage_scholar.routing import ExecutionMode, SourceScope, Workflow
from test_document_evidence import library
from research_fakes import FakeSupervisor, SupportedVerifier
from sage_scholar.domain.research import Claim, Finding, GapReview, FollowUp, ReportDraft, ResearchNote
from test_source_policy import ExternalTool


def build_runtime(monkeypatch, tmp_path, adapter, scope, ids, approved=True):
    import sage_scholar.research.runtime as module
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    (configs / "agents.json").write_text(json.dumps({
        "researcher": {"allowed_tools": ["rag_search", "web_fetch"]},
    }))
    class Supervisor(FakeSupervisor):
        def __init__(self, **kwargs):
            pass
    class Review:
        def __init__(self, *args):
            pass
        def review(self, contract, tasks, notes):
            if approved or (approved is None and notes[0].revision > 0):
                return GapReview()
            return GapReview(missing=["实验结果未核验"], next_tasks=[FollowUp(
                task_id="T1", objective="补充实验结果")])
    class Verifier(SupportedVerifier):
        def __init__(self, *args):
            pass
    class Researcher:
        def __init__(self, tool_list, **kwargs):
            self.tools = {tool.name: tool for tool in tool_list}
            self.ids = kwargs["observed_ids"]
        def run(self, task, contract, dependency_notes):
            assert contract.source_policy.document_ids == tuple(ids)
            assert contract.source_policy.permits_web == (scope == SourceScope.PUBLIC)
            if scope == SourceScope.PUBLIC:
                assert set(self.tools) == {"web_fetch"}
                content = self.tools["web_fetch"].run_tool({"url": "https://example.org/paper"})
            else:
                assert set(self.tools) == {"rag_search"}
                content = self.tools["rag_search"].run_tool({"query": "method", "document_ids": ["other"]})
                assert "Method A" in content and "SECRET B" not in content
            return ResearchNote(task_id=task.task_id, question=task.objective, revision=task.revision,
                evidence_ids=self.ids(), findings=[Finding(text="source finding", evidence_ids=self.ids())])
    class Writer:
        def __init__(self, **kwargs):
            pass
        def run(self, contract, notes, **kwargs):
            assert not contract.source_policy.permits_web or scope == SourceScope.PUBLIC
            return ReportDraft(title="report", claims=[Claim(claim_id="C1", text="source finding",
                evidence_ids=notes[0].evidence_ids)])
    monkeypatch.setattr(module, "Supervisor", Supervisor)
    monkeypatch.setattr(module, "Researcher", Researcher)
    monkeypatch.setattr(module, "Writer", Writer)
    monkeypatch.setattr(module, "GapReviewer", Review)
    monkeypatch.setattr(module, "CitationVerifier", Verifier)
    def forbidden_connect(*args):
        raise AssertionError("Offline Research must not connect any MCP server")
    monkeypatch.setattr(module, "create_mcp_clients", forbidden_connect)
    runtime = ResearchRuntime(
        Config(data_dir=adapter._config.data_dir, config_dir=str(configs), max_rounds=0),
        llm_client=SimpleNamespace(total_calls=0, total_prompt_tokens=0, total_completion_tokens=0),
        retrieval=adapter, connect_external=False,
    )
    tool = ExternalTool()
    tool.evidence_kind = "source_content"
    runtime.registry.register_tool(tool)
    request = ExecutionRequest(
        query="调研所选论文", workflow=Workflow.DEEP_RESEARCH, source_scope=scope,
        document_ids=tuple(ids), allow_web=scope == SourceScope.PUBLIC, execution_mode=ExecutionMode.RUN,
    )
    return runtime, request


def test_real_langgraph_reuses_local_evidence_without_external_connection(monkeypatch, tmp_path, library):
    _, adapter, a, _, _, _ = library
    runtime, request = build_runtime(monkeypatch, tmp_path, adapter, SourceScope.SELECTED_DOCUMENTS, [a.document_id])
    events = list(runtime.run_research(request))
    assert any("write_node" in event for event in events)
    evidence_ids = next(event["evidence_node"]["evidence_ids"] for event in events if "evidence_node" in event)
    assert evidence_ids
    for identity in evidence_ids:
        assert adapter.evidence_store.get(identity).document_id == a.document_id
    assert events[-1]["outcome_node"]["remaining_gaps"] == []


def test_public_research_preserves_raw_tool_evidence(monkeypatch, tmp_path, library):
    _, adapter, _, _, _, _ = library
    runtime, request = build_runtime(monkeypatch, tmp_path, adapter, SourceScope.PUBLIC, [])
    events = list(runtime.run_research(request))
    evidence_ids = next(event["evidence_node"]["evidence_ids"] for event in events if "evidence_node" in event)
    assert len(evidence_ids) == 1
    evidence = adapter.evidence_store.get(evidence_ids[0])
    assert evidence.original_text == "external content"
    assert evidence.metadata["kind"] == "source_content"
    assert evidence.page is None and evidence.document_id is None


def test_review_limit_produces_partial_via_research_service(monkeypatch, tmp_path, library):
    _, adapter, a, _, _, _ = library
    runtime, request = build_runtime(monkeypatch, tmp_path, adapter, SourceScope.SELECTED_DOCUMENTS,
                                     [a.document_id], approved=False)
    class Provider:
        def get(self, require_external):
            assert require_external is False
            return runtime
    async def scenario():
        emitted = []
        async def emit(kind, data):
            emitted.append(kind)
        outcome = await ResearchService(Provider()).run(request, emit, lambda: False)
        assert outcome.status == "partial"
        assert "实验结果未核验" in outcome.remaining_gaps
        assert outcome.result["report"].startswith("# report")
        assert outcome.result["citations"]
        assert outcome.result["evidence_ids"]
        assert "write" in emitted and "stats" in emitted
    asyncio.run(scenario())


def test_no_available_tool_produces_partial_without_fabricating_report(monkeypatch, tmp_path, library):
    _, adapter, _, _, _, _ = library
    runtime, request = build_runtime(monkeypatch, tmp_path, adapter, SourceScope.PUBLIC, [])
    request = request.model_copy(update={"allow_web": False})
    events = list(runtime.run_research(request))
    assert not any("write_node" in event for event in events)
    assert events[-1]["outcome_node"]["remaining_gaps"]


def test_targeted_retry_keeps_selected_documents_and_no_web(monkeypatch, tmp_path, library):
    _, adapter, a, _, _, _ = library
    runtime, request = build_runtime(monkeypatch, tmp_path, adapter, SourceScope.SELECTED_DOCUMENTS,
                                     [a.document_id], approved=None)
    request = request.model_copy(update={"budget": {"max_rounds": 1}})
    events = list(runtime.run_research(request))
    notes = [event["research_node"]["research_notes"]["T1"] for event in events if "research_node" in event]
    assert [note.revision for note in notes] == [0, 1]
    assert notes[0].task_id == notes[1].task_id
    assert events[-1]["outcome_node"]["remaining_gaps"] == []
    assert all(adapter.evidence_store.get(identity).document_id == a.document_id
               for note in notes for identity in note.evidence_ids)
    public = [kind for event in events for kind, _ in runtime.to_public_events(event)]
    assert {"research_planned", "research_task_started", "research_task_completed", "gap_reviewed",
            "report_drafting", "citation_verifying", "report_completed", "write"} <= set(public)
