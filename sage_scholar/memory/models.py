"""Durable memory is user background, never a citation source."""
import hashlib
import json
from datetime import datetime
from typing import Literal
from uuid import uuid4
from pydantic import AliasChoices, BaseModel, Field, computed_field, model_validator
from sage_scholar.domain.documents import utc_now

MemoryType = Literal["preference", "note", "decision", "background", "research_note"]
Scope = Literal["user", "workspace", "session"]


class MemorySource(BaseModel):
    kind: Literal["manual", "explicit", "conversation", "research", "legacy"] = "manual"
    reference: str | None = None
    quote: str = ""
    message_index: int | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class MemoryInput(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    memory_type: MemoryType = Field(default="preference", validation_alias=AliasChoices("memory_type", "type"))
    scope: Scope = "workspace"
    workspace_id: str | None = "default"
    session_id: str | None = None
    confirmed: bool = False
    importance: float = Field(default=0.6, ge=0, le=1)

    @computed_field
    @property
    def type(self) -> str:  # V4 API compatibility, not a second field of truth.
        return self.memory_type

    @model_validator(mode="after")
    def validate_scope(self):
        self.content = self.content.strip()
        if not self.content:
            raise ValueError("记忆内容不能为空")
        if self.scope == "user":
            self.workspace_id = self.session_id = None
        elif not self.workspace_id or not self.workspace_id.strip():
            raise ValueError("workspace/session memory requires workspace_id")
        elif self.scope == "session" and not self.session_id:
            raise ValueError("session scope requires session_id")
        elif self.scope == "workspace":
            self.session_id = None
        return self


class MemoryRecord(MemoryInput):
    memory_id: str = Field(default_factory=lambda: "mem_" + uuid4().hex)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    source: MemorySource = Field(default_factory=MemorySource)
    sources: list[MemorySource] = Field(default_factory=list)
    confidence: float = Field(default=1, ge=0, le=1)
    status: Literal["active", "superseded", "deleted"] = "active"
    version: int = 1
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
    manual_revision: bool = False


class Consolidation(BaseModel):
    action: Literal["new", "merge", "update", "ignore"] = "new"
    targets: list[str] = Field(default_factory=list, max_length=8)
    reason: str = ""
    # No model-written replacement text: the new version always uses user-reviewed content.


class MemoryCandidate(MemoryInput):
    candidate_id: str = Field(default_factory=lambda: "mc_" + uuid4().hex)
    source: MemorySource
    confidence: float = Field(default=0.7, ge=0, le=1)
    rationale: str
    status: Literal["pending", "accepted", "rejected"] = "pending"
    proposal: Consolidation = Field(default_factory=Consolidation)
    target_versions: dict[str, int] = Field(default_factory=dict)
    scope_fingerprint: str = ""
    revision: int = 1
    created_at: datetime = Field(default_factory=utc_now)
    result_id: str | None = None


def visible(item, workspace_id="default", session_id=None):
    return item.scope == "user" or (item.workspace_id == workspace_id and (
        item.scope == "workspace" or (session_id is not None and item.session_id == session_id)))


def same_scope(a, b):
    return (a.scope, a.workspace_id, a.session_id) == (b.scope, b.workspace_id, b.session_id)


def scope_fingerprint(items, candidate):
    state = sorted((r.memory_id, r.version, r.status) for r in items if same_scope(r, candidate))
    return hashlib.sha256(json.dumps(state).encode()).hexdigest()
