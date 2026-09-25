from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
from uuid import uuid4

from sage_scholar.domain.documents import DocumentRecord, IngestionStatus, IngestResult, utc_now
from sage_scholar.infrastructure.document_store import JsonDocumentStore
from .converter import convert_to_markdown


class LibraryManager:
    """Retain the existing converter/indexer, isolate each ingestion version on disk."""

    def __init__(self, data_dir, paper_tool, pdfmux_tool, pipeline):
        self.data_dir = data_dir
        self.paper_tool = paper_tool
        self.pdfmux_tool = pdfmux_tool
        self.pipeline = pipeline
        self.store = JsonDocumentStore(data_dir)

    def ingest(self, src: str, custom_title: str | None = None, overwrite: bool = True,
               save_original: bool = True) -> IngestResult:
        # One process, serial library mutations; reads see explicit ingestion status.
        with self.store.lock:
            return self._ingest(src, custom_title, overwrite, save_original)

    def _ingest(self, src, custom_title, overwrite, save_original):
        is_arxiv = bool(re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", src))
        title = custom_title or (f"arXiv:{src}" if is_arxiv else Path(src).stem)
        # Exact bytes / arXiv ID identify reuploads. Title never determines identity.
        source_key = "arxiv:" + src if is_arxiv else "sha256:" + hashlib.sha256(Path(src).read_bytes()).hexdigest()
        existing = next((doc for doc in self.store.list() if doc.source_identifier == source_key), None)
        if existing and not overwrite:
            return self._result(existing, "skipped")
        previous = existing.model_copy(deep=True) if existing else None
        doc = existing or DocumentRecord(title=title, source_identifier=source_key)
        doc.title = title
        doc.ingestion_status = IngestionStatus.UPLOADED
        doc.updated_at = utc_now()
        self.store.put(doc)
        staging = Path(self.data_dir) / "converted" / doc.document_id / uuid4().hex
        staging.mkdir(parents=True, exist_ok=True)
        new_path = None
        old_chunks = None
        try:
            doc.ingestion_status = IngestionStatus.PARSING
            self.store.put(doc)
            converted = convert_to_markdown(
                src, str(staging), self.paper_tool, self.pdfmux_tool, custom_title="document",
            )
            new_path = converted.output_path
            doc.converted_path = new_path
            doc.document_version = hashlib.sha256(Path(new_path).read_bytes()).hexdigest()
            doc.parser_version = "pdfmux-adapter-v1" if converted.source_type == "pdf" else "markdown-adapter-v1"
            doc.metadata.update(source_type=converted.source_type, arxiv_id=converted.arxiv_id)
            doc.metadata.pop("ingestion_error", None)
            if save_original:
                original = Path(self.paper_tool.download_dir) / f"{src}.pdf" if is_arxiv else Path(src)
                if original.is_file():
                    destination = staging / ("original" + original.suffix)
                    shutil.copyfile(original, destination)
                    doc.original_path = str(destination)
            doc.ingestion_status = IngestionStatus.INDEXING
            self.store.put(doc)
            self.pipeline.add_document(new_path, save=False)
            # Switch only after parsing/embedding succeeded. Keep old version files for evidence.
            if previous and previous.converted_path:
                old_chunks = [c for c in self.pipeline.vector_store.chunks if c.file_path == previous.converted_path]
                self.pipeline.vector_store.remove_by_filepath(previous.converted_path)
            self.pipeline.save()
            doc.ingestion_status = IngestionStatus.READY
            doc.updated_at = utc_now()
            self.store.put(doc)
            return self._result(doc, "overwritten" if previous else "created")
        except Exception as exc:
            if new_path:
                self.pipeline.vector_store.remove_by_filepath(new_path)
            if old_chunks:
                self.pipeline.vector_store.add_chunks(old_chunks)
            # Never expose a failed replacement as the current ready version.
            if previous and previous.ingestion_status == IngestionStatus.READY:
                doc = previous
            else:
                doc.ingestion_status = IngestionStatus.FAILED
            doc.metadata["ingestion_error"] = str(exc)
            doc.updated_at = utc_now()
            self.store.put(doc)
            raise

    @staticmethod
    def _result(doc, status):
        return IngestResult(title=doc.title, status=status, document_id=doc.document_id,
                            document_version=doc.document_version, ingestion_status=doc.ingestion_status)

    def list_docs(self) -> list[dict]:
        return [self.store.to_legacy_dict(doc) for doc in self.store.list()]

    def delete_doc(self, identifier: str):
        with self.store.lock:
            docs = self.store.list()
            found = [doc for doc in docs if doc.document_id == identifier]
            if not found:
                found = [doc for doc in docs if doc.title == identifier]
            if len(found) > 1:
                raise ValueError("存在同名文档，请使用 document_id 删除")
            for doc in found:
                if doc.converted_path:
                    self.pipeline.vector_store.remove_by_filepath(doc.converted_path)
                self.pipeline.save()
                self.store.delete(doc.document_id)
                # Immutable versions/evidence remain on disk for prior answers.
