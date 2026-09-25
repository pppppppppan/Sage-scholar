from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from pydantic import AliasChoices, BaseModel, Field, computed_field


class EvidenceRef(BaseModel):
    """A stable reference to source text returned by retrieval."""

    evidence_id: str
    source_id: str
    document_id: str | None = None
    source_version: str = "legacy"
    original_text: str = Field(validation_alias=AliasChoices("original_text", "text"))
    page: int | None = None
    section: str | None = None
    table: str | None = None
    parser_version: str = "legacy-markdown"
    score: float | None = None
    chunk_index: int | None = None
    location: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def text(self) -> str:
        """Compatibility field for the existing frontend."""
        return self.original_text

    @classmethod
    def create(cls, **values) -> "EvidenceRef":
        identity = {key: values.get(key) for key in (
            "source_id", "document_id", "source_version", "parser_version",
            "page", "section", "table", "chunk_index", "location", "original_text",
        )}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return cls(evidence_id="ev_" + digest, **values)


class EvidenceStore(Protocol):
    def save(self, evidence: EvidenceRef) -> None: ...
    def get(self, evidence_id: str) -> EvidenceRef | None: ...
