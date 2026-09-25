from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from sage_scholar.library import LibraryManager


class DocumentService:
    """Application facade over the migrated library manager."""

    def __init__(self, runtime_provider, data_dir: str, pipeline_factory):
        self._runtime_provider = runtime_provider
        self._data_dir = data_dir
        self._pipeline_factory = pipeline_factory

    async def list_documents(self) -> list[dict]:
        index_path = os.path.join(self._data_dir, "index.json")
        if not os.path.exists(index_path):
            return []
        return await asyncio.to_thread(self._read_index, index_path)

    @staticmethod
    def _read_index(index_path: str) -> list[dict]:
        with open(index_path, "r", encoding="utf-8") as file:
            return json.load(file)

    async def ingest(self, source: str, custom_title: str | None = None, overwrite: bool = True):
        suffix = Path(source).suffix.lower()
        if suffix in {".md", ".markdown", ".txt"}:
            manager = self._local_manager()
            return await asyncio.to_thread(
                manager.ingest,
                source,
                custom_title,
                overwrite,
            )
        runtime = await asyncio.to_thread(self._runtime_provider.get)
        manager = runtime.create_library_manager()
        if suffix == ".pdf" and runtime.pdfmux_tool is None:
            raise RuntimeError("PDF 解析工具未配置；请启用 pdfmux MCP 服务。")
        return await asyncio.to_thread(
            manager.ingest,
            source,
            custom_title,
            overwrite,
        )

    async def delete(self, title: str) -> None:
        await asyncio.to_thread(self._local_manager().delete_doc, title)

    def _local_manager(self) -> LibraryManager:
        return LibraryManager(
            data_dir=self._data_dir,
            paper_tool=None,
            pdfmux_tool=None,
            pipeline=self._pipeline_factory(),
        )
