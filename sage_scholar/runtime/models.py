from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field

from sage_scholar.routing import Workflow
from sage_scholar.domain.execution import ExecutionRequest


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


TERMINAL_STATUSES = {
    RunStatus.CANCELLED,
    RunStatus.COMPLETED,
    RunStatus.PARTIAL,
    RunStatus.FAILED,
    RunStatus.INTERRUPTED,
}


class RunEvent(BaseModel):
    sequence: int
    type: str
    data: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RunSnapshot(BaseModel):
    run_id: str
    workflow: Workflow
    query: str
    status: RunStatus
    result: dict | None = None
    error: str | None = None
    execution: ExecutionRequest
    reason: str | None = None
    remaining_gaps: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
