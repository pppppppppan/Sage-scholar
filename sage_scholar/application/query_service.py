from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import uuid

from sage_scholar.domain import EvidenceRef
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.routing import QueryRouter, RouteDecision, SourceScope, Workflow, ExecutionMode
from sage_scholar.runtime import RunManager

from .direct_service import DirectService
from .document_resolver import DocumentResolver
from .ports import LongTaskRunner
from .qa_service import QAService
from sage_scholar.graph.query_graph import QueryGraphRunner, ENTRY_VERSION
from sage_scholar.runtime.limits import execution_budget


class QueryStatus(StrEnum):
    INLINE = "inline"
    ACCEPTED = "accepted"
    NEEDS_CLARIFICATION = "needs_clarification"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass(slots=True)
class QueryCommand:
    query: str
    workflow: Workflow | None = None
    source_scope: SourceScope | None = None
    document_ids: list[str] = field(default_factory=list)
    allow_web: bool | None = None
    timezone: str = "Asia/Shanghai"
    history: list[dict[str, str]] = field(default_factory=list)
    workspace_id: str = "default"
    conversation_id: str | None = None
    session_id: str | None = None
    budget: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    request_id: str | None = None


@dataclass(slots=True)
class QueryResult:
    request_id: str
    route: RouteDecision
    status: QueryStatus
    answer: str | None = None
    evidence: list[EvidenceRef] = field(default_factory=list)
    run_id: str | None = None
    message: str | None = None
    citations: dict[int, str] = field(default_factory=dict)
    citation_issues: list[str] = field(default_factory=list)
    reading_strategy: str | None = None


class QueryService:
    def __init__(
        self,
        router: QueryRouter,
        direct: DirectService,
        qa: QAService,
        run_manager: RunManager,
        long_runners: dict[Workflow, LongTaskRunner],
        document_resolver: DocumentResolver | None = None,
        memory_store=None,
    ):
        self._run_manager = run_manager
        self.graph_runner = QueryGraphRunner(router, direct, qa, long_runners.get(Workflow.DEEP_RESEARCH),
            documents=document_resolver, memory=memory_store, checkpoints=run_manager.checkpoints)
        run_manager.register_entry_runner(self.graph_runner)

    async def execute(self, command: QueryCommand) -> QueryResult:
        if command.workflow in {Workflow.CITATION_AUDIT, Workflow.PAPER2CODE, Workflow.SEARCH, Workflow.ROUTER, Workflow.CLARIFY}:
            raise ValueError("请使用 Direct、QA 或 Deep Research；Router 和 Clarify 由工作流自动管理")
        request_id = command.request_id or uuid.uuid4().hex
        # Validate overrides before acknowledging; semantic inference happens inside the durable Run.
        execution_budget(Workflow.ROUTER, command.budget)
        options = {"workflow": command.workflow, "source_scope": command.source_scope,
                   "document_ids": command.document_ids, "allow_web": command.allow_web, "budget": command.budget}
        route = RouteDecision(workflow=Workflow.ROUTER, source_scope=SourceScope.CONVERSATION,
            allow_web=False, execution_mode=ExecutionMode.RUN, reason_code="routing_pending")
        execution = ExecutionRequest(
            **route.model_dump(include={"workflow", "source_scope", "document_ids", "allow_web", "execution_mode"}),
            query=command.query, request_id=request_id, timezone=command.timezone, workspace_id=command.workspace_id,
            history=command.history, conversation_id=command.conversation_id, session_id=command.session_id,
            budget=execution_budget(Workflow.ROUTER, command.budget),
            metadata={**command.metadata, "_entry_graph": ENTRY_VERSION, "_routing": options},
        )
        run = await self._run_manager.start(execution, self.graph_runner)
        return QueryResult(request_id=request_id, route=route, status=QueryStatus.ACCEPTED, run_id=run.run_id)
