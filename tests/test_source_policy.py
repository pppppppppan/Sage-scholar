import asyncio
from types import SimpleNamespace

import pytest

from sage_scholar.application.direct_service import DirectService
from sage_scholar.application.qa_service import QAService
from sage_scholar.application.query_service import QueryCommand, QueryService
from sage_scholar.application.research_service import ResearchService
from sage_scholar.domain.execution import SourcePolicy, RunOutcome
from sage_scholar.routing import QueryRouter, SourceScope, Workflow
from sage_scholar.runtime import RunManager, RunStatus
from sage_scholar.tools import BaseTool, ToolRegistry, ToolCallError
from sage_scholar.tools.policy import PolicyTool
from test_query_service import FakeModel, FakeRetrieval


class ExternalTool(BaseTool):
    source_access = "public"

    def __init__(self, name="web_fetch"):
        super().__init__(name, "external", [])
        self.calls = 0

    def run_tool(self, parameters):
        self.calls += 1
        return "external content"


@pytest.mark.parametrize("name", ["search", "web_fetch", "search_arxiv", "search_google_scholar"])
def test_no_web_excludes_and_rejects_external_tools(name):
    tool = ExternalTool(name)
    registry = ToolRegistry()
    registry.register_tool(tool)
    policy = SourcePolicy(source_scope=SourceScope.PUBLIC, allow_web=False)
    assert registry.for_policy([name], policy) == []
    with pytest.raises(ToolCallError):
        PolicyTool(tool, policy).run_tool({})
    assert tool.calls == 0


def test_unknown_tool_is_denied_and_policy_is_per_run():
    tool = ExternalTool()
    registry = ToolRegistry()
    registry.register_tool(tool)
    public = SourcePolicy(source_scope=SourceScope.PUBLIC, allow_web=True)
    local = SourcePolicy(source_scope=SourceScope.WORKSPACE, allow_web=True)
    assert registry.for_policy([tool.name], local) == []
    assert registry.for_policy([tool.name], public)[0].run_tool({}) == "external content"
    tool.source_access = "unknown"
    assert registry.for_policy([tool.name], public) == []


@pytest.mark.parametrize("scope,ids", [
    (SourceScope.SELECTED_DOCUMENTS, ["doc-a", "doc-b"]),
    (SourceScope.WORKSPACE, []),
])
def test_router_contract_reaches_actual_research_service_runtime(scope, ids):
    async def scenario():
        seen = {}
        class Runtime:
            def usage_snapshot(self):
                return {"total_calls": 0}
            def run_research(self, request):
                seen["request"] = request
                yield {"write_node": {"final_report": "report"}}
            def to_public_event(self, event):
                return "write", {"report": event["write_node"]["final_report"]}
        class Provider:
            def get(self, require_external=True):
                seen["external"] = require_external
                return Runtime()
        manager = RunManager()
        model = FakeModel()
        service = QueryService(QueryRouter(), DirectService(model), QAService(FakeRetrieval(), model),
                               manager, {Workflow.DEEP_RESEARCH: ResearchService(Provider())})
        result = await service.execute(QueryCommand(
            query="调研我的资料", source_scope=scope, document_ids=ids,
            allow_web=False, conversation_id="conversation-a", budget={"steps": 3},
        ))
        events = [event async for event in manager.events(result.run_id)]
        request = seen["request"]
        assert request.source_scope == result.route.source_scope == scope
        assert request.document_ids == tuple(ids)
        assert request.allow_web is False
        assert request.conversation_id == "conversation-a"
        assert request.budget == {"steps": 3}
        assert seen["external"] is False
        snapshot = manager.get(result.run_id)
        assert snapshot.execution == request
        assert snapshot.status == RunStatus.COMPLETED
        assert any(event.type == "write" for event in events)
        await manager.shutdown()
    asyncio.run(scenario())


def test_partial_outcome_is_not_completed():
    async def scenario():
        class Runner:
            async def run(self, request, emit, cancelled):
                return RunOutcome(status="partial", result={"report": "some findings"},
                                  reason="step budget exhausted", remaining_gaps=["experiment settings"])
        from sage_scholar.domain.execution import ExecutionRequest
        route = QueryRouter().route("调研 AI")
        request = ExecutionRequest(query="调研 AI", **route.model_dump(exclude={
            "needs_clarification", "clarification_message", "reason_code"}))
        manager = RunManager()
        run = await manager.start(request, Runner())
        events = [event async for event in manager.events(run.run_id)]
        snapshot = manager.get(run.run_id)
        assert snapshot.status == RunStatus.PARTIAL
        assert snapshot.result["report"] == "some findings"
        assert snapshot.remaining_gaps == ["experiment settings"]
        assert events[-1].data["status"] == "partial"
        assert events[-1].data["reason"] == "step budget exhausted"
    asyncio.run(scenario())


def test_empty_selected_documents_never_falls_back_to_workspace():
    route = QueryRouter().route("调研论文", requested_scope=SourceScope.SELECTED_DOCUMENTS)
    assert route.needs_clarification
    assert not SourcePolicy(source_scope=SourceScope.SELECTED_DOCUMENTS).permits_document("any")
