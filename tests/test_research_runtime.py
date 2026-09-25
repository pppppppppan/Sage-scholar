import asyncio
import json
from types import SimpleNamespace

from sage_scholar.application.research_service import ResearchService
from sage_scholar.config import Config
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.research.runtime import ResearchRuntime
from sage_scholar.routing import ExecutionMode, SourceScope, Workflow
from test_document_evidence import library
from test_research_graph import FakeSupervisor
from test_source_policy import ExternalTool


def build_runtime(monkeypatch, tmp_path, adapter, scope, ids, approved=True):
    import sage_scholar.research.runtime as module
    from sage_scholar.agents.supervisor import NoteReview, ReviewResult
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    (configs / "agents.json").write_text(json.dumps({
        "researcher": {"allowed_tools": ["rag_search", "web_fetch"]},
    }))
    class Supervisor(FakeSupervisor):
        def __init__(self, **kwargs):
            pass
        def review(self, *args, **kwargs):
            if approved:
                return super().review(*args, **kwargs)
            return ReviewResult(note_reviews=[NoteReview(
                relevance="PASS", depth="FAIL", citations="PASS", sources="PASS",
                completeness="FAIL", verdict="retry",
            )], missing_dimensions="实验结果未核验")
    class Researcher:
        def __init__(self, tool_list, **kwargs):
            self.tools = {tool.name: tool for tool in tool_list}
        def run(self, **kwargs):
            if scope == SourceScope.PUBLIC:
                assert set(self.tools) == {"web_fetch"}
                return self.tools["web_fetch"].run_tool({}), {"web_fetch": 1}
            assert set(self.tools) == {"rag_search"}
            content = self.tools["rag_search"].run_tool({"query": "method"})
            assert "Method A" in content and "SECRET B" not in content
            return content, {"rag_search": 1}
    class Writer:
        def __init__(self, **kwargs):
            pass
        def run(self, research_brief, notes):
            assert '"allow_web":false' in research_brief or scope == SourceScope.PUBLIC
            return "# report"
    monkeypatch.setattr(module, "Supervisor", Supervisor)
    monkeypatch.setattr(module, "Researcher", Researcher)
    monkeypatch.setattr(module, "Writer", Writer)
    def forbidden_connect(*args):
        raise AssertionError("Offline Research must not connect any MCP server")
    monkeypatch.setattr(module, "create_mcp_clients", forbidden_connect)
    runtime = ResearchRuntime(
        Config(data_dir=adapter._config.data_dir, config_dir=str(configs), max_rounds=0),
        llm_client=SimpleNamespace(total_calls=0, total_prompt_tokens=0, total_completion_tokens=0),
        retrieval=adapter, connect_external=False,
    )
    runtime.registry.register_tool(ExternalTool())
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
    assert evidence.metadata["kind"] == "tool_observation"
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
        assert outcome.result["report"] == "# report"
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
