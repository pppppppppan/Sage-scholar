"""Explicit user memories; never stored in RAG or EvidenceStore."""
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator
from sage_scholar.domain.documents import utc_now
from sage_scholar.infrastructure.json_store import file_lock, write_json
from datetime import datetime


class MemoryInput(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    type: Literal["preference", "note"] = "preference"
    scope: Literal["workspace", "session"] = "workspace"
    session_id: str | None = None
    confirmed: bool = False

    @model_validator(mode="after")
    def validate_scope(self):
        if not self.content.strip():
            raise ValueError("记忆内容不能为空")
        if self.scope == "session" and not self.session_id:
            raise ValueError("session scope requires session_id")
        if self.scope == "workspace":
            self.session_id = None
        return self


class MemoryRecord(MemoryInput):
    memory_id: str = Field(default_factory=lambda: "mem_" + uuid4().hex)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MemoryStore:
    def __init__(self, data_dir):
        self.path = Path(data_dir) / "memory.json"
        self.lock = file_lock(self.path)

    def list(self, session_id=None, *, all_scopes=False):
        with self.lock:
            records = [MemoryRecord.model_validate(r) for r in json.loads(self.path.read_text())] if self.path.exists() else []
        return [r for r in records if all_scopes or r.scope == "workspace" or r.session_id == session_id]

    def add(self, value: MemoryInput):
        if not value.confirmed:
            raise ValueError("只保存用户明确要求或确认的记忆")
        item = MemoryRecord(**value.model_dump())
        with self.lock:
            items = self.list(all_scopes=True)
            items.append(item)
            write_json(self.path, [r.model_dump(mode="json") for r in items])
        return item

    def update(self, identity, value: MemoryInput):
        if not value.confirmed:
            raise ValueError("修改记忆需要用户明确确认")
        with self.lock:
            items = self.list(all_scopes=True)
            old = next((r for r in items if r.memory_id == identity), None)
            if not old:
                raise KeyError(identity)
            item = MemoryRecord(**value.model_dump(), memory_id=identity, created_at=old.created_at)
            write_json(self.path, [item.model_dump(mode="json") if r.memory_id == identity else r.model_dump(mode="json") for r in items])
            return item

    def delete(self, identity):
        with self.lock:
            items = self.list(all_scopes=True)
            if not any(r.memory_id == identity for r in items):
                raise KeyError(identity)
            write_json(self.path, [r.model_dump(mode="json") for r in items if r.memory_id != identity])
