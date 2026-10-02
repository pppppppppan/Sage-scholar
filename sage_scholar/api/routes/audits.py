from fastapi import APIRouter, HTTPException
from pydantic import Field
from sage_scholar.audit.models import AuditInput
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.routing import Workflow, SourceScope, ExecutionMode
from ..dependencies import Container
from ..schemas import RunResponse

router = APIRouter()


class AuditRequest(AuditInput):
    source_scope: SourceScope | None = None
    document_ids: list[str] = Field(default_factory=list)
    allow_web: bool = False
    session_id: str | None = None
    workspace_id: str = "default"


@router.post("/api/audits", response_model=RunResponse, status_code=202)
async def audit(body: AuditRequest, container: Container):
    scope, ids, web = body.source_scope, body.document_ids, body.allow_web
    if body.report_run_id:
        previous = container.run_manager.get(body.report_run_id)
        if previous is None or not previous.result:
            raise HTTPException(404, "report run not found")
        if scope is None and not ids:
            scope, ids = previous.execution.source_scope, list(previous.execution.document_ids)
            if "allow_web" not in body.model_fields_set:
                web = previous.execution.allow_web
    if not body.text and not body.claims and not body.report_run_id and not ids:
        raise HTTPException(400, "请提供待审报告、主张与引用或报告 Run ID")
    workspace_id = previous.execution.workspace_id if body.report_run_id else body.workspace_id
    request = ExecutionRequest(query=body.text or "核验报告中的引用和主张", workflow=Workflow.CITATION_AUDIT,
        source_scope=scope or (SourceScope.SELECTED_DOCUMENTS if ids else SourceScope.MIXED), document_ids=tuple(ids),
        allow_web=web, execution_mode=ExecutionMode.RUN, session_id=body.session_id, workspace_id=workspace_id,
        metadata={"audit": body.model_dump(include=set(AuditInput.model_fields))})
    return RunResponse(run=await container.run_manager.start(request, container.audit_service))
