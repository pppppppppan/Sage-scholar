from typing import Literal
from pydantic import BaseModel, Field
from sage_scholar.domain.research import Claim


class AuditClaim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)


class AuditInput(BaseModel):
    text: str = ""
    claims: list[AuditClaim] = Field(default_factory=list, max_length=100)
    citations: dict[str, str] = Field(default_factory=dict)
    report_run_id: str | None = None


class ParsedClaims(BaseModel):
    claims: list[AuditClaim] = Field(default_factory=list, max_length=100)


class CitationIdentity(BaseModel):
    citation: str
    source: str
    status: Literal["source_found", "source_not_found"]
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class AuditItem(BaseModel):
    claim: Claim
    sources: list[CitationIdentity]
    source_identity_status: Literal["source_found", "source_not_found"]
    support_status: Literal["supported", "partially_supported", "unsupported", "uncertain"]
    evidence_ids: list[str]
    locations: list[dict]
    reason: str
    revision_suggestion: str
