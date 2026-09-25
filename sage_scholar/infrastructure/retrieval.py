from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path

from sage_scholar.config import Config
from sage_scholar.domain import EvidenceRef
from sage_scholar.domain.documents import IngestionStatus
from sage_scholar.domain.execution import SourcePolicy
from sage_scholar.rag import Pipeline

from .document_store import JsonDocumentStore
from .evidence_store import JsonEvidenceStore


class LegacyRetrievalAdapter:
    """Shared QA/Research adapter over the original hybrid retrieval pipeline."""

    def __init__(self, config: Config, pipeline: Pipeline | None = None, documents=None, evidence_store=None):
        self._config = config
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self.documents = documents or JsonDocumentStore(config.data_dir)
        self.evidence_store = evidence_store or JsonEvidenceStore(config.data_dir)

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self._pipeline = Pipeline(data_dir=self._config.data_dir)
        return self._pipeline

    def permitted_documents(self, policy: SourcePolicy):
        return [
            doc for doc in self.documents.list()
            if doc.ingestion_status == IngestionStatus.READY
            and doc.converted_path and policy.permits_document(doc.document_id)
        ]

    async def retrieve(self, query: str, *, policy: SourcePolicy, top_k: int = 5) -> list[EvidenceRef]:
        return await asyncio.to_thread(self.retrieve_sync, query, policy, top_k)

    def retrieve_sync(self, query: str, policy: SourcePolicy, top_k: int = 5) -> list[EvidenceRef]:
        docs = {doc.converted_path: doc for doc in self.permitted_documents(policy)}
        if not docs:
            return []
        # Restrict candidates before BM25/vector scoring, then defensively validate returned paths.
        results = self.pipeline.retrieve_chunks(query, top_k=top_k, allowed_file_paths=set(docs))
        evidence = []
        for chunk, score in results:
            doc = docs.get(chunk.file_path)
            if doc is None:
                continue
            item = self._evidence(doc, chunk.content, score=score, chunk_index=chunk.chunk_idx,
                                  location={"chunk_index": chunk.chunk_idx})
            evidence.append(item)
            # Read the containing section when the legacy chunk can be located unambiguously.
            matching = [
                section for section in self._sections(doc)
                if re.sub(r"\s+", "", chunk.content) in re.sub(r"\s+", "", section.original_text)
            ]
            if len(matching) == 1 and matching[0].original_text != chunk.content:
                evidence.append(matching[0])
        return list({item.evidence_id: item for item in evidence}.values())

    async def read_documents(self, *, policy: SourcePolicy) -> list[EvidenceRef]:
        return await asyncio.to_thread(self.read_documents_sync, policy)

    def read_documents_sync(self, policy: SourcePolicy) -> list[EvidenceRef]:
        return [section for doc in self.permitted_documents(policy) for section in self._sections(doc)]

    def _sections(self, doc) -> list[EvidenceRef]:
        text = Path(doc.converted_path).read_text(encoding="utf-8")
        headings = list(re.finditer(r"(?m)^#{1,6}\s+(.+)$", text))
        starts = sorted(set([0, *[match.start() for match in headings]]))
        titles = {match.start(): match.group(1) for match in headings}
        result = []
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(text)
            for offset in range(start, end, 6000):
                part_end = min(end, offset + 6000)
                original = text[offset:part_end]
                if original.strip():
                    result.append(self._evidence(
                        doc, original, section=titles.get(start),
                        location={"char_start": offset, "char_end": part_end},
                    ))
        return result

    def _evidence(self, doc, original_text, **values):
        item = EvidenceRef.create(
            source_id=doc.source_identifier, document_id=doc.document_id,
            source_version=doc.document_version, parser_version=doc.parser_version,
            original_text=original_text, page=None,
            metadata={"title": doc.title, "converted_path": doc.converted_path},
            **values,
        )
        self.evidence_store.save(item)
        return item
