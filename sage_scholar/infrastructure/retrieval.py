from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading

from sage_scholar.config import Config
from sage_scholar.domain import EvidenceRef
from sage_scholar.rag import Pipeline


class LegacyRetrievalAdapter:
    """Expose the migrated hybrid retriever through a structured Evidence API."""

    def __init__(self, config: Config, pipeline: Pipeline | None = None):
        self._config = config
        self._pipeline = pipeline
        self._lock = threading.Lock()

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self._pipeline = Pipeline(data_dir=self._config.data_dir)
        return self._pipeline

    async def retrieve(
        self,
        query: str,
        *,
        document_ids: list[str],
        top_k: int = 5,
    ) -> list[EvidenceRef]:
        return await asyncio.to_thread(self._retrieve_sync, query, document_ids, top_k)

    def _retrieve_sync(self, query: str, document_ids: list[str], top_k: int) -> list[EvidenceRef]:
        allowed_paths = self._resolve_document_paths(document_ids)
        results = self.pipeline.retrieve_chunks(query, top_k=top_k, allowed_file_paths=allowed_paths)
        evidence = []
        for chunk, score in results:
            identity = f"{chunk.file_path}:{chunk.chunk_idx}:{chunk.content}"
            evidence.append(
                EvidenceRef(
                    evidence_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                    source_id=chunk.file_path,
                    text=chunk.content,
                    score=score,
                    metadata={"chunk_index": str(chunk.chunk_idx)},
                )
            )
        return evidence

    def _resolve_document_paths(self, document_ids: list[str]) -> set[str] | None:
        if not document_ids:
            return None
        index_path = os.path.join(self._config.data_dir, "index.json")
        if not os.path.exists(index_path):
            return set()
        with open(index_path, "r", encoding="utf-8") as file:
            entries = json.load(file)
        requested = set(document_ids)
        return {
            entry["converted_path"]
            for entry in entries
            if entry.get("title") in requested or entry.get("converted_path") in requested
        }
