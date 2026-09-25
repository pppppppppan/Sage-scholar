from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import uuid

from sage_scholar.domain import EvidenceRef
from sage_scholar.routing import QueryRouter, RouteDecision, SourceScope, Workflow
from sage_scholar.runtime import RunManager

from .direct_service import DirectService
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


@dataclass(slots=True)
class QueryResult:
    request_id: str
    route: RouteDecision
    status: QueryStatus
    answer: str | None = None
    evidence: list[EvidenceRef] = field(default_factory=list)
    run_id: str | None = None
    message: str | None = None


class QueryService:
    def __init__(
        self,
        router: QueryRouter,
        direct: DirectService,
        qa: QAService,
        run_manager: RunManager,
        long_runners: dict[Workflow, LongTaskRunner],
    ):
        self._router = router
        self._direct = direct
        self._qa = qa
        self._run_manager = run_manager
        self._long_runners = long_runners

    async def execute(self, command: QueryCommand) -> QueryResult:
        request_id = uuid.uuid4().hex
        route = self._router.route(
            command.query,
            requested_workflow=command.workflow,
            requested_scope=command.source_scope,
            document_ids=command.document_ids,
            allow_web=command.allow_web,
        )
        if route.needs_clarification:
            return QueryResult(
                request_id=request_id,
                route=route,
                status=QueryStatus.NEEDS_CLARIFICATION,
                message=route.clarification_message,
            )

        if route.workflow is Workflow.DIRECT:
            answer = await self._direct.answer(command.query, command.timezone, command.history)
            return QueryResult(request_id=request_id, route=route, status=QueryStatus.INLINE, answer=answer)

        if route.workflow is Workflow.QA:
            result = await self._qa.answer(command.query, route.document_ids)
            return QueryResult(
                request_id=request_id,
                route=route,
                status=QueryStatus.INLINE,
                answer=result.answer,
                evidence=result.evidence,
            )

        runner = self._long_runners.get(route.workflow)
        if runner is not None:
            run = await self._run_manager.start(route.workflow, command.query, runner)
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
