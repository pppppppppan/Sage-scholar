from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from sage_scholar.library import LibraryManager
from sage_scholar.infrastructure.document_store import JsonDocumentStore


class DocumentService:
    """Application facade over the migrated library manager."""

    def __init__(self, runtime_provider, data_dir: str, pipeline_factory):
        self._runtime_provider = runtime_provider
        self._data_dir = data_dir
        self._pipeline_factory = pipeline_factory
        self._store = JsonDocumentStore(data_dir)

    async def list_documents(self) -> list[dict]:
        return await asyncio.to_thread(
            lambda: [self._store.to_legacy_dict(doc) for doc in self._store.list()]
        )

    async def ingest(self, source: str, custom_title: str | None = None, overwrite: bool = True, *,
                     document_id: str | None = None, source_identifier: str | None = None):
        cancelled = threading.Event()

        def ingest():
            suffix = Path(source).suffix.lower()
            manager = self._local_manager() if suffix in {".md", ".markdown", ".txt"} else (
                self._runtime_provider.get_library_manager("pdf" if suffix == ".pdf" else "paper"))
            return manager.ingest(source, custom_title, overwrite, document_id=document_id,
                                  source_identifier=source_identifier, cancelled=cancelled)

        task = asyncio.create_task(asyncio.to_thread(ingest))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            # Wait until the worker reaches a safe boundary and finishes rollback.
            # Repeated cancellation must not abandon a thread still writing indexes.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except BaseException:
                    if task.done():
                        break
            if not task.cancelled():
                task.exception()
            raise

    async def delete(self, title: str) -> None:
        manager = await asyncio.to_thread(self._local_manager)
        await asyncio.to_thread(manager.delete_doc, title)

    def _local_manager(self) -> LibraryManager:
        return LibraryManager(
            data_dir=self._data_dir,
            paper_tool=None,
            pdfmux_tool=None,
            pipeline=self._pipeline_factory(),
        )
