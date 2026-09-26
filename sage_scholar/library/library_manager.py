from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
from uuid import uuid4

from sage_scholar.domain.documents import DocumentRecord, DocumentVersion, IngestionStatus, IngestResult, utc_now
from sage_scholar.infrastructure.document_store import JsonDocumentStore
from sage_scholar.infrastructure.json_store import file_lock
from .converter import convert_to_markdown


class LibraryManager:
    """Serialize ingestion independently of short catalog reads/writes."""

    def __init__(self, data_dir, paper_tool, pdfmux_tool, pipeline):
        self.data_dir = data_dir
        self.paper_tool = paper_tool
        self.pdfmux_tool = pdfmux_tool
        self.pipeline = pipeline
        self.store = JsonDocumentStore(data_dir)
        self.ingestion_lock = file_lock(Path(data_dir) / ".ingestion")

    def ingest(self, src: str, custom_title: str | None = None, overwrite: bool = True,
               save_original: bool = True, *, document_id: str | None = None,
               source_identifier: str | None = None) -> IngestResult:
        with self.ingestion_lock:
            return self._ingest(src, custom_title, overwrite, save_original, document_id, source_identifier)

    def _ingest(self, src, custom_title, overwrite, save_original, document_id, source_identifier):
        arxiv = re.fullmatch(r"(\d{4}\.\d{4,5})(?:v\d+)?", src)
        source_key = source_identifier or (
            "arxiv:" + arxiv.group(1) if arxiv else "path:" + str(Path(src).resolve())
        )
        if document_id:
            existing = self.store.get(document_id)
            if existing is None:
                raise ValueError("指定的 document_id 不存在")
        else:
            existing = next((doc for doc in self.store.list() if doc.source_identifier == source_key), None)
        if existing and not overwrite:
            return self._result(existing, "skipped")
        # Current paths/version remain active throughout the pending ingestion.
        doc = existing.model_copy(deep=True) if existing else DocumentRecord(
            title=custom_title or (f"arXiv:{src}" if arxiv else Path(src).stem),
            source_identifier=source_key,
        )
        if custom_title:
            doc.title = custom_title
        doc.metadata.pop("ingestion_error", None)
        self._status(doc, IngestionStatus.UPLOADED)
        staging = Path(self.data_dir) / "converted" / doc.document_id / uuid4().hex
        new_path = None
        committed = False
        try:
            staging.mkdir(parents=True, exist_ok=True)
            # Parse a stable snapshot, not a caller-owned file which can change mid-ingestion.
            original = None
            if not arxiv:
                original = staging / ("original" + Path(src).suffix)
                shutil.copyfile(src, original)
            self._status(doc, IngestionStatus.PARSING)
            converted = convert_to_markdown(
                src if arxiv else str(original), str(staging),
                self.paper_tool, self.pdfmux_tool, custom_title="document",
            )
            new_path = converted.output_path
            if arxiv:
                downloaded = Path(self.paper_tool.download_dir) / f"{src}.pdf"
                if downloaded.is_file():
                    original = staging / "original.pdf"
                    shutil.copyfile(downloaded, original)
            content_hash = hashlib.sha256((original or Path(new_path)).read_bytes()).hexdigest()
            parser_version = "pdfmux-adapter-v1" if converted.source_type == "pdf" else "markdown-adapter-v1"
            if (doc.active_version and content_hash == doc.content_hash
                    and parser_version == doc.parser_version):
                self._status(doc, IngestionStatus.READY)
                shutil.rmtree(staging)
                return self._result(doc, "skipped")

            self._status(doc, IngestionStatus.INDEXING)
            self.pipeline.add_document(new_path, save=False)
            # Keep previous chunks: active retrieval filters by the catalog's current path.
            # A crash before catalog commit leaves only unreachable new chunks, not a lost old index.
            self.pipeline.save()
            number = max((version.version_number for version in doc.versions), default=0) + 1
            version = DocumentVersion(
                document_version=f"v{number}", version_number=number,
                content_hash=content_hash, parser_version=parser_version,
                original_path=str(original) if original and save_original else None,
                converted_path=new_path,
            )
            current = doc.model_copy(deep=True)
            current.versions.append(version)
            current.document_version = current.active_version = version.document_version
            current.content_hash = content_hash
            current.parser_version = parser_version
            current.original_path = version.original_path
            current.converted_path = new_path
            current.metadata.update(source_type=converted.source_type, arxiv_id=converted.arxiv_id)
            self._status(current, IngestionStatus.READY)
            committed = True
            return self._result(current, "overwritten" if existing else "created")
        except Exception as exc:
            if new_path and not committed:
                self.pipeline.vector_store.remove_by_filepath(new_path)
            # FAILED describes the latest attempt; active fields still describe the last good version.
            doc.metadata["ingestion_error"] = str(exc)
            self._status(doc, IngestionStatus.FAILED)
            raise

    def _status(self, doc, status):
        doc.ingestion_status = status
        doc.updated_at = utc_now()
        self.store.put(doc)  # Short atomic catalog write; no parser/index work under this lock.

    @staticmethod
    def _result(doc, status):
        return IngestResult(
            title=doc.title, status=status, document_id=doc.document_id,
            document_version=doc.document_version, content_hash=doc.content_hash,
            active_version=doc.active_version, ingestion_status=doc.ingestion_status,
        )

    def list_docs(self) -> list[dict]:
        return [self.store.to_legacy_dict(doc) for doc in self.store.list()]

    def delete_doc(self, identifier: str):
        with self.ingestion_lock:
            docs = self.store.list()
            found = [doc for doc in docs if doc.document_id == identifier]
            if not found:
                found = [doc for doc in docs if doc.title == identifier]
            if len(found) > 1:
                raise ValueError("存在同名文档，请使用 document_id 删除")
            for doc in found:
                # Hide from active retrieval first; historical evidence/files remain readable.
                self.store.delete(doc.document_id)
                for version in doc.versions:
                    self.pipeline.vector_store.remove_by_filepath(version.converted_path)
                self.pipeline.save()
