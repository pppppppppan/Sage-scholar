from fastapi import APIRouter, HTTPException, Response, status
from sage_scholar.application import QueryCommand
from ..dependencies import Container
from ..schemas import QueryRequest, QueryResponse

router = APIRouter()

@router.post("/api/queries", response_model=QueryResponse)
async def submit_query(body: QueryRequest, response: Response, container: Container):
    try:
        result = await (container.conversations.submit if hasattr(container, "conversations") else container.query_service.execute)(
            QueryCommand(
                query=body.query, request_id=body.request_id,
                workflow=body.workflow,
                source_scope=body.source_scope,
                document_ids=body.document_ids,
                allow_web=body.allow_web,
                timezone=body.timezone,
                history=[message.model_dump() for message in body.history],
                conversation_id=body.conversation_id, session_id=body.session_id, workspace_id=body.workspace_id,
                budget=body.budget, metadata=body.metadata,
            )
        )
    except KeyError as exc:
        raise HTTPException(404, detail="conversation not found") from exc
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    if result.run_id:
        response.status_code = status.HTTP_202_ACCEPTED
    return QueryResponse(
        conversation_id=container.run_manager.get(result.run_id).execution.conversation_id if result.run_id else body.conversation_id,
        request_id=result.request_id,
        route=result.route,
        status=result.status.value,
        answer=result.answer,
        evidence=result.evidence,
        run_id=result.run_id,
        message=result.message,
        citations=result.citations, citation_issues=result.citation_issues,
        reading_strategy=result.reading_strategy,
    )
