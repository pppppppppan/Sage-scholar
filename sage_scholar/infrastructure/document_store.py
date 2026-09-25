from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sage_scholar.domain.documents import DocumentRecord, IngestionStatus

from .json_store import file_lock, write_json


class JsonDocumentStore:
    """Adapts legacy index.json in place; generated IDs persist across title changes."""

    def __init__(self, data_dir: str):
        self.path = Path(data_dir) / "index.json"
        self.lock = file_lock(self.path)

    def list(self) -> list[DocumentRecord]:
        with self.lock:
            entries = json.loads(self.path.read_text()) if self.path.exists() else []
            migrated = False
            result = []
            for entry in entries:
                if "document_id" not in entry:
                    converted = Path(entry["converted_path"])
                    exists = converted.is_file()
                    content_hash = hashlib.sha256(converted.read_bytes()).hexdigest() if exists else "unavailable"
                    doc = DocumentRecord(
                        title=entry["title"],
                        source_identifier=entry.get("arxiv_id") or entry.get("original_path") or str(converted),
                        original_path=entry.get("original_path"), converted_path=str(converted),
                        document_version=content_hash, parser_version="legacy-markdown",
                        ingestion_status=IngestionStatus.READY if exists else IngestionStatus.FAILED,
                        metadata={key: value for key, value in entry.items() if key not in {
                            "title", "original_path", "converted_path",
                        }},
                    )
                    migrated = True
                else:
                    doc = DocumentRecord.model_validate(entry)
                result.append(doc)
            if migrated:
                self._write(result)
            return result

    def get(self, document_id: str) -> DocumentRecord | None:
        return next((doc for doc in self.list() if doc.document_id == document_id), None)

    def put(self, doc: DocumentRecord) -> None:
        with self.lock:
            docs = self.list()
            docs = [item for item in docs if item.document_id != doc.document_id]
            self._write([*docs, doc])

    def delete(self, document_id: str) -> None:
        with self.lock:
            self._write([doc for doc in self.list() if doc.document_id != document_id])

    def _write(self, docs: list[DocumentRecord]) -> None:
        write_json(self.path, [self.to_legacy_dict(doc) for doc in docs])

    @staticmethod
    def to_legacy_dict(doc: DocumentRecord) -> dict:
        return {
            **doc.model_dump(mode="json"),
            "added_at": doc.created_at.isoformat(),
            "source_type": doc.metadata.get("source_type", "unknown"),
            "arxiv_id": doc.metadata.get("arxiv_id"),
        }
