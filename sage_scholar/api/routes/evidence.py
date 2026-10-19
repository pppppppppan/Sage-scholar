from fastapi import APIRouter, HTTPException
from sage_scholar.domain import EvidenceRef
from ..dependencies import Container

router = APIRouter()


@router.get("/api/evidence/{evidence_id}/image")
async def get_evidence_image(evidence_id: str, container: Container):
    from fastapi.responses import FileResponse
    from sage_scholar.runtime.control import cooperative_thread
    try:
        evidence = container.evidence_store.get(evidence_id)
        if evidence is None:
            raise ValueError("图片来源不存在")
        path = await cooperative_thread(container.retrieval.visual.resolve_image, evidence)
    except (ValueError, OSError) as exc:
        raise HTTPException(404, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})


@router.get("/api/evidence/{evidence_id}", response_model=EvidenceRef)
async def get_evidence(evidence_id: str, container: Container):
    try:
        evidence = container.evidence_store.get(evidence_id)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    if evidence is None:
        raise HTTPException(404, detail="evidence not found")
    return evidence
