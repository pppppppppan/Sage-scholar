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
from sage_scholar.domain.documents import utc_now
from sage_scholar.routing import QueryRouter, RouteDecision, SourceScope, Workflow
from sage_scholar.runtime import RunManager

from .direct_service import DirectService
from .document_resolver import DocumentResolver
from .ports import LongTaskRunner
from .qa_service import QAService
from .inline_runner import InlineRunner
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
        memory_background=None,
    ):
        self._memory_store = memory_store
        self._memory_background = memory_background
        self._document_resolver = document_resolver
        self._router = router
        self._direct = direct
        self._qa = qa
        self._run_manager = run_manager
        self._long_runners = long_runners
        self.inline_runner = InlineRunner(direct, qa, memory_store)

    async def execute(self, command: QueryCommand) -> QueryResult:
        started_at = utc_now()
        request_id = command.request_id or uuid.uuid4().hex
        explicit_memory = re.fullmatch(r"(?:请)?记住[：:]\s*(.+)", command.query.strip(), re.S)
        document_ids = command.document_ids
        resolution_message = None
        # Explicit remote/conversation scope and Direct/Search choices retain their semantics.
        if self._document_resolver and command.workflow not in {Workflow.DIRECT, Workflow.SEARCH} and command.source_scope in {None, SourceScope.SELECTED_DOCUMENTS}:
            resolution = await self._document_resolver.resolve(command.query, document_ids)
            document_ids, resolution_message = resolution.document_ids, resolution.message
        route = self._router.route(
            command.query,
            requested_workflow=Workflow.DIRECT if explicit_memory else command.workflow,
            requested_scope=command.source_scope,
            document_ids=document_ids,
            allow_web=command.allow_web,
        )
        if resolution_message:
            route = route.model_copy(update={"needs_clarification": True, "clarification_message": resolution_message})
        if route.needs_clarification and not command.conversation_id:
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
            session_id=command.session_id, budget=execution_budget(route.workflow, command.budget),
            metadata={**command.metadata, **({"clarification_message": route.clarification_message} if route.needs_clarification else {})},
        )

        if route.workflow in {Workflow.DIRECT, Workflow.QA} or route.needs_clarification:
            if command.conversation_id:
                run = await self._run_manager.start(execution, self.inline_runner)
                return QueryResult(request_id=request_id, route=route, status=QueryStatus.ACCEPTED, run_id=run.run_id)
            async def discard(*args):
                pass
            with request_context(execution):
                outcome = await self.inline_runner.run(execution, discard, lambda: False)
            result = outcome.result
            if self._memory_background:
                self._memory_background.conversation(execution, result["answer"], started_at=started_at)
            return QueryResult(request_id=request_id, route=route, status=QueryStatus.INLINE,
                answer=result["answer"], evidence=[EvidenceRef.model_validate(e) for e in result.get("evidence", [])],
                citations=result.get("citations", {}), citation_issues=result.get("citation_issues", []),
                reading_strategy=result.get("reading_strategy"))

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
