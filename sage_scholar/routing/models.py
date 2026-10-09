from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Workflow(StrEnum):
    DIRECT = "direct"
    QA = "qa"
    SEARCH = "search"
    DEEP_RESEARCH = "deep_research"
    CITATION_AUDIT = "citation_audit"  # Read-only decoding of historical runs; Router rejects new requests.
    PAPER2CODE = "paper2code"


class SourceScope(StrEnum):
    CONVERSATION = "conversation"
    SELECTED_DOCUMENTS = "selected_documents"
    WORKSPACE = "workspace"
    PUBLIC = "public"
    MIXED = "mixed"


class ExecutionMode(StrEnum):
    INLINE = "inline"
    RUN = "run"


class RouteDecision(BaseModel):
    workflow: Workflow
    source_scope: SourceScope
    document_ids: list[str] = Field(default_factory=list)
    allow_web: bool
    execution_mode: ExecutionMode
    needs_clarification: bool = False
    clarification_message: str | None = None
    reason_code: str
