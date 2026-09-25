from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceRef(BaseModel):
    """A stable reference to source text returned by retrieval."""

    evidence_id: str
    source_id: str
    source_version: str = "legacy"
    text: str
    page: int | None = None
    section: str | None = None
    table: str | None = None
    parser_version: str = "legacy-markdown"
    score: float | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
