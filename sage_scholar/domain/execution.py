from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sage_scholar.routing.models import ExecutionMode, SourceScope, Workflow


class SourcePolicy(BaseModel):
    """External-source access is distinct from remote model inference."""

    model_config = ConfigDict(frozen=True)
    source_scope: SourceScope
    document_ids: tuple[str, ...] = ()
    allow_web: bool = False

    @property
    def permits_web(self) -> bool:
        return self.allow_web and self.source_scope in {SourceScope.PUBLIC, SourceScope.MIXED}

    @property
    def permits_local(self) -> bool:
        return self.source_scope in {
            SourceScope.SELECTED_DOCUMENTS, SourceScope.WORKSPACE, SourceScope.MIXED,
        }

    def permits_document(self, document_id: str) -> bool:
        if not self.permits_local:
            return False
        if self.source_scope == SourceScope.SELECTED_DOCUMENTS:
            return document_id in self.document_ids
        if self.source_scope == SourceScope.MIXED and self.document_ids:
            return document_id in self.document_ids
        return True

    def permits_tool(self, source_access: str) -> bool:
        if source_access == "public":
            return self.permits_web
        if source_access == "local":
            return self.permits_local
        return source_access == "none"


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    query: str
    workflow: Workflow
    source_scope: SourceScope
    document_ids: tuple[str, ...] = ()
    allow_web: bool
    execution_mode: ExecutionMode
    run_id: str | None = None
    request_id: str | None = None
    conversation_id: str | None = None
    session_id: str | None = None
    budget: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timezone: str = "Asia/Shanghai"
    history: list[dict[str, str]] = Field(default_factory=list)

    @property
    def source_policy(self) -> SourcePolicy:
        return SourcePolicy(
            source_scope=self.source_scope, document_ids=self.document_ids, allow_web=self.allow_web,
        )


class RunOutcome(BaseModel):
    status: Literal["completed", "partial"] = "completed"
    result: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    remaining_gaps: list[str] = Field(default_factory=list)
