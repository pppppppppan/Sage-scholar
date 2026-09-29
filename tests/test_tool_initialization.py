import asyncio
import json
from types import SimpleNamespace

import pytest
from mcp.types import Tool

from sage_scholar.application.document_service import DocumentService
from sage_scholar.application.query_service import QueryCommand
from sage_scholar.application.research_service import ResearchService
from sage_scholar.config import Config
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.research.runtime import ResearchRuntime, ResearchRuntimeProvider
from sage_scholar.routing import SourceScope, Workflow, ExecutionMode
from test_document_evidence import library
from test_query_service import FakeModel
from test_research_runtime import build_runtime


def forbidden(*args, **kwargs):
    raise AssertionError("MCP must remain lazy for this workflow")


def test_direct_and_local_qa_do_not_initialize_mcp(monkeypatch, library):
    _, adapter, a, _, _, _ = library
    import sage_scholar.infrastructure.container as module
    monkeypatch.setattr(module, "LegacyRetrievalAdapter", lambda _: adapter)
    monkeypatch.setattr(module, "LegacyModelGateway", lambda _: FakeModel("Method A [1]"))
    monkeypatch.setattr("sage_scholar.research.runtime.iter_mcp_clients", forbidden)
    monkeypatch.setattr("sage_scholar.research.runtime.create_mcp_clients", forbidden)
    async def scenario():
        container = module.ApplicationContainer(adapter._config)
        direct = await container.query_service.execute(QueryCommand(query="解释 KL 散度", workflow=Workflow.DIRECT))
        qa = await container.query_service.execute(QueryCommand(query="论文的方法是什么", workflow=Workflow.QA,
            source_scope=SourceScope.SELECTED_DOCUMENTS, document_ids=[a.document_id], allow_web=False))
        assert direct.answer and qa.answer and qa.evidence
        assert container.runtime_provider._runtimes == {} and container.runtime_provider._library_clients == []
        await container.close()
    asyncio.run(scenario())


def test_offline_research_never_loads_web_even_in_external_capable_runtime(monkeypatch, tmp_path, library):
    _, adapter, a, _, _, _ = library
    runtime, request = build_runtime(monkeypatch, tmp_path, adapter, SourceScope.SELECTED_DOCUMENTS, [a.document_id])
    runtime._connect_external, runtime._tools_initialized = True, False
    monkeypatch.setattr("sage_scholar.research.runtime.iter_mcp_clients", forbidden)
    events = list(runtime.run_research(request))
    assert any("render_node" in event for event in events)
    assert runtime.mcp_clients == []


@pytest.mark.parametrize("gap", [False, True])
def test_public_initialization_progress_failure_fallback_and_cache(monkeypatch, tmp_path, library, gap):
    _, adapter, _, _, _, _ = library
    runtime, request = build_runtime(monkeypatch, tmp_path, adapter, SourceScope.PUBLIC, [], approved=not gap)
    runtime._connect_external, runtime._tools_initialized = True, False
    calls = []
    def initialize(path, *, server_names, connect_timeout):
        calls.append(set(server_names))
        yield {"server": "fetch", "status": "unavailable", "reason": "timeout",
               "message": "网页读取不可用，将继续使用其他来源"}, None
        yield {"server": "paper-search", "status": "available", "message": "论文搜索与读取已就绪"}, None
    monkeypatch.setattr("sage_scholar.research.runtime.iter_mcp_clients", initialize)
    class Provider:
        def get(self, require_external):
            assert require_external
            return runtime
    async def scenario():
        emitted = []
        async def emit(kind, data):
            emitted.append((kind, data))
        result = await ResearchService(Provider()).run(request, emit, lambda: False)
        kinds = [kind for kind, _ in emitted]
        assert kinds.index("initializing_tools") < kinds.index("tool_unavailable") < kinds.index("research_started") < kinds.index("plan")
        assert "tool_available" in kinds and "write" in kinds
        assert result.result["report"] and result.status == ("partial" if gap else "completed")
        assert "Traceback" not in json.dumps(emitted, ensure_ascii=False)
        return result
    asyncio.run(scenario())
    asyncio.run(scenario())
    assert calls == [{"tavily", "fetch", "paper-search"}]
    assert "pdfmux" not in calls[0] and "github" not in calls[0]


def test_all_optional_tools_unavailable_returns_partial_without_planning(monkeypatch, tmp_path, library):
    _, adapter, _, _, _, _ = library
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agents.json").write_text('{"researcher":{"allowed_tools":["search","mcp__fetch__fetch"]}}')
    touched = []
    def initialize(path, **kwargs):
        touched.append(True)
        yield {"server": "fetch", "status": "unavailable", "reason": "timeout", "message": "网页读取不可用"}, None
    monkeypatch.setattr("sage_scholar.research.runtime.iter_mcp_clients", initialize)
    runtime = ResearchRuntime(Config(config_dir=str(config_dir)), retrieval=adapter,
        llm_client=SimpleNamespace(total_calls=0, total_prompt_tokens=0, total_completion_tokens=0))
    assert touched == []  # Constructor does not initialize even public MCP.
    request = ExecutionRequest(query="research", workflow=Workflow.DEEP_RESEARCH,
        source_scope=SourceScope.PUBLIC, allow_web=True, execution_mode=ExecutionMode.RUN)
    class Provider:
        def get(self, require_external):
            return runtime
    async def scenario():
        kinds = []
        async def emit(kind, data): kinds.append(kind)
        result = await ResearchService(Provider()).run(request, emit, lambda: False)
        assert result.status == "partial" and result.remaining_gaps
        assert "tool_unavailable" in kinds and "plan" not in kinds
    asyncio.run(scenario())


def test_pdf_ingestion_only_loads_pdf_tool_and_no_research_model(monkeypatch, tmp_path, library):
    _, adapter, _, _, _, _ = library
    selections = []
    class PDFClient:
        name = "pdfmux"
        tools = [Tool(name="convert_pdf", description="PDF", inputSchema={"type": "object", "properties": {}})]
        def call_tool(self, name, args):
            assert name == "convert_pdf" and args["file_path"].endswith(".pdf")
            return "# PDF\nExtracted source content"
        def disconnect(self): pass
    def connect(path, *, server_names, connect_timeout):
        selections.append(set(server_names))
        return [PDFClient()]
    monkeypatch.setattr("sage_scholar.research.runtime.create_mcp_clients", connect)
    provider = ResearchRuntimeProvider(adapter._config, forbidden, adapter)
    service = DocumentService(provider, adapter._config.data_dir, lambda: adapter.pipeline)
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"test fixture PDF parsed by fake adapter")
    result = asyncio.run(service.ingest(str(source)))
    assert result.ingestion_status == "ready"
    assert selections == [{"pdfmux"}] and provider._runtimes == {}
    provider.get_library_manager("pdf")
    assert len(selections) == 1
    provider.close()
