from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IngestionStatus(StrEnum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class DocumentVersion(BaseModel):
    document_version: str
    version_number: int
    content_hash: str | None = None
    parser_version: str
    original_path: str | None = None
    converted_path: str
    created_at: datetime = Field(default_factory=utc_now)


class DocumentRecord(BaseModel):
    document_id: str = Field(default_factory=lambda: "doc_" + uuid4().hex)
    title: str
    source_identifier: str
    original_path: str | None = None
    converted_path: str | None = None
    document_version: str = "pending"
    content_hash: str | None = None
    active_version: str | None = None
    versions: list[DocumentVersion] = Field(default_factory=list)
    parser_version: str = "unknown"
    ingestion_status: IngestionStatus = IngestionStatus.UPLOADED
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    title: str
    status: Literal["skipped", "overwritten", "created"]
    document_id: str
    document_version: str
    content_hash: str | None = None
    active_version: str | None = None
    ingestion_status: IngestionStatus
