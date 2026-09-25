from fastapi import APIRouter, HTTPException
from sage_scholar.domain import EvidenceRef
from ..dependencies import Container

router = APIRouter()


@router.get("/api/evidence/{evidence_id}", response_model=EvidenceRef)
async def get_evidence(evidence_id: str, container: Container):
    try:
        evidence = container.evidence_store.get(evidence_id)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    if evidence is None:
        raise HTTPException(404, detail="evidence not found")
    return evidence
