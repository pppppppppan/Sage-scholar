"""Allowlisted Memory projection for prompts; provenance stays in the Memory Store."""
from pydantic import BaseModel, ConfigDict
from sage_scholar.memory.models import MemoryRecord, MemoryType, Scope


class MemoryContextView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    content: str
    memory_type: MemoryType
    scope: Scope

    @classmethod
    def from_record(cls, record: MemoryRecord) -> "MemoryContextView":
        # Read only prompt-safe fields; never serialize the full record/source first.
        return cls(memory_id=record.memory_id, content=record.content,
                   memory_type=record.memory_type, scope=record.scope)
