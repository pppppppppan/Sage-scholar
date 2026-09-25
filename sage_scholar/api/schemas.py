from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from sage_scholar.domain import EvidenceRef
from sage_scholar.routing import RouteDecision, SourceScope, Workflow
from sage_scholar.runtime import RunSnapshot


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    workflow: Workflow | None = None
    source_scope: SourceScope | None = None
    document_ids: list[str] = Field(default_factory=list)
    allow_web: bool | None = None
    timezone: str = "Asia/Shanghai"
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class QueryResponse(BaseModel):
    request_id: str
    route: RouteDecision
    status: Literal["inline", "accepted", "needs_clarification", "not_implemented"]
    answer: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    run_id: str | None = None
    message: str | None = None


class IngestRequest(BaseModel):
    src: str
    custom_title: str | None = None
    overwrite: bool = True


class SaveReportRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1)


class IngestResult(BaseModel):
    title: str
    status: Literal["skipped", "overwritten", "created"]


class RunResponse(BaseModel):
    run: RunSnapshot
