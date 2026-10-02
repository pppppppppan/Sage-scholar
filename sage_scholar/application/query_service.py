from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import uuid
import re
import asyncio
from sage_scholar.context.manager import request_context
from sage_scholar.memory import MemoryInput

from sage_scholar.domain import EvidenceRef
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.routing import QueryRouter, RouteDecision, SourceScope, Workflow
from sage_scholar.runtime import RunManager

from .direct_service import DirectService
from .document_resolver import DocumentResolver
from .ports import LongTaskRunner
from .qa_service import QAService


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
        self._memory_store = memory_store
        self._document_resolver = document_resolver
        self._router = router
        self._direct = direct
        self._qa = qa
        self._run_manager = run_manager
        self._long_runners = long_runners

    async def execute(self, command: QueryCommand) -> QueryResult:
        request_id = uuid.uuid4().hex
        explicit_memory = re.fullmatch(r"(?:请)?记住[：:]\s*(.+)", command.query.strip(), re.S)
        if self._memory_store and explicit_memory:
            if hasattr(self._memory_store, "remember"):
                memory = await asyncio.to_thread(self._memory_store.remember, explicit_memory[1],
                    workspace_id=command.workspace_id, session_id=command.session_id or command.conversation_id)
            else:
                memory = await asyncio.to_thread(self._memory_store.add, MemoryInput(content=explicit_memory[1], confirmed=True))
            return QueryResult(request_id=request_id, route=self._router.route(command.query, requested_workflow=Workflow.DIRECT),
                               status=QueryStatus.INLINE, answer=f"已记住：{memory.content}。可在记忆管理中修改或删除。")
        document_ids = command.document_ids
        resolution_message = None
        # Explicit remote/conversation scope and Direct/Search choices retain their semantics.
        if self._document_resolver and command.workflow not in {Workflow.DIRECT, Workflow.SEARCH} and command.source_scope in {None, SourceScope.SELECTED_DOCUMENTS}:
            resolution = await self._document_resolver.resolve(command.query, document_ids)
            document_ids, resolution_message = resolution.document_ids, resolution.message
        route = self._router.route(
            command.query,
            requested_workflow=command.workflow,
            requested_scope=command.source_scope,
            document_ids=document_ids,
            allow_web=command.allow_web,
        )
        if resolution_message:
            route = route.model_copy(update={"needs_clarification": True, "clarification_message": resolution_message})
        if route.needs_clarification:
            return QueryResult(
                request_id=request_id,
                route=route,
                status=QueryStatus.NEEDS_CLARIFICATION,
                message=route.clarification_message,
            )

        execution = ExecutionRequest(
            **route.model_dump(include={
                "workflow", "source_scope", "document_ids", "allow_web", "execution_mode",
            }),
            query=command.query, request_id=request_id, timezone=command.timezone, workspace_id=command.workspace_id,
            history=command.history, conversation_id=command.conversation_id,
            session_id=command.session_id, budget=command.budget, metadata=command.metadata,
        )

        if route.workflow is Workflow.DIRECT:
            with request_context(execution):
                answer = await self._direct.answer(execution)
            return QueryResult(request_id=request_id, route=route, status=QueryStatus.INLINE, answer=answer)

        if route.workflow is Workflow.QA:
            with request_context(execution):
                result = await self._qa.answer(execution)
            return QueryResult(
                request_id=request_id,
                route=route,
                status=QueryStatus.INLINE,
                answer=result.answer,
                evidence=result.evidence,
                citations=result.citations, citation_issues=result.citation_issues,
                reading_strategy=result.reading_strategy,
            )

        runner = self._long_runners.get(route.workflow)
        if runner is not None:
            run = await self._run_manager.start(execution, runner)
            return QueryResult(
                request_id=request_id,
                route=route,
                status=QueryStatus.ACCEPTED,
                run_id=run.run_id,
            )

        return QueryResult(
            request_id=request_id,
            route=route,
            status=QueryStatus.NOT_IMPLEMENTED,
            message=f"{route.workflow.value} 路径已定义，但尚未实现。",
        )
