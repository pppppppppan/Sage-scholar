from __future__ import annotations

import asyncio
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

    async def ingest(self, source: str, custom_title: str | None = None, overwrite: bool = True):
        suffix = Path(source).suffix.lower()
        if suffix in {".md", ".markdown", ".txt"}:
            manager = await asyncio.to_thread(self._local_manager)
            return await asyncio.to_thread(
                manager.ingest,
                source,
                custom_title,
                overwrite,
            )
        runtime = await asyncio.to_thread(self._runtime_provider.get)
        manager = await asyncio.to_thread(runtime.create_library_manager)
        if suffix == ".pdf" and runtime.pdfmux_tool is None:
            raise RuntimeError("PDF 解析工具未配置；请启用 pdfmux MCP 服务。")
        return await asyncio.to_thread(
            manager.ingest,
            source,
            custom_title,
            overwrite,
        )

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
