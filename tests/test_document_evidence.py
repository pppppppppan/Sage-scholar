import asyncio
import json
from pathlib import Path

import pytest

from sage_scholar.config import Config
from sage_scholar.domain.documents import IngestionStatus
from sage_scholar.domain.execution import SourcePolicy
from sage_scholar.infrastructure.document_store import JsonDocumentStore
from sage_scholar.infrastructure.evidence_store import JsonEvidenceStore
from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
from sage_scholar.library.library_manager import LibraryManager
from sage_scholar.rag.document import Chunk
from sage_scholar.rag.vector_store import VectorStore
from sage_scholar.routing import SourceScope
from sage_scholar.tools.tool_rag import RAGTool


class LocalPipeline:
    """No embedding API, real store/filter semantics; inspect supplied scope."""
    def __init__(self):
        self.vector_store = VectorStore()
        self.allowed = None
        self.fail = False
        self.statuses = []
        self.document_store = None
    def add_document(self, file_path, save=True):
        if self.document_store:
            self.statuses.append(self.document_store.list()[0].ingestion_status)
        if self.fail:
            raise RuntimeError("embedding failed")
        self.vector_store.add_chunks([
            Chunk(file_path=file_path, chunk_idx=0, content=Path(file_path).read_text(),
                  tokens=["method"], embedding=[1.0, 0.0])
        ])
    def save(self):
        pass
    def retrieve_chunks(self, query, top_k, allowed_file_paths):
        self.allowed = allowed_file_paths
        return [(chunk, 0.9) for chunk in self.vector_store.chunks if chunk.file_path in allowed_file_paths][:top_k]


@pytest.fixture
def library(tmp_path):
    pipeline = LocalPipeline()
    data = tmp_path / "data"
    manager = LibraryManager(str(data), None, None, pipeline)
    pipeline.document_store = manager.store
    source_a = tmp_path / "a.md"
    source_b = tmp_path / "b.md"
    source_a.write_text("# Method\nMethod A\n# Experiments\nDataset Alpha\n# Limitations\nNeeds more testing")
    source_b.write_text("# Method\nSECRET B")
    a = manager.ingest(str(source_a), custom_title="same title")
    b = manager.ingest(str(source_b), custom_title="same title")
    adapter = LegacyRetrievalAdapter(Config(data_dir=str(data)), pipeline)
    return manager, adapter, a, b, source_a, pipeline


def test_document_ids_are_stable_across_title_change_and_process_reload(library):
    manager, adapter, a, b, source, pipeline = library
    assert a.document_id != b.document_id
    again = manager.ingest(str(source), custom_title="renamed")
    assert again.document_id == a.document_id
    record = JsonDocumentStore(manager.data_dir).get(a.document_id)
    assert record.title == "renamed"
    assert record.document_id not in {record.title, record.converted_path}
    assert Path(record.converted_path).is_file()
    assert record.ingestion_status == IngestionStatus.READY
    assert record.document_version != "pending"
    assert IngestionStatus.INDEXING in pipeline.statuses


def test_selected_documents_filter_before_retrieval_and_tool_arguments_cannot_override(library):
    manager, adapter, a, b, source, pipeline = library
    policy = SourcePolicy(source_scope=SourceScope.SELECTED_DOCUMENTS, document_ids=(a.document_id,))
    result = asyncio.run(adapter.retrieve("method", policy=policy))
    assert {item.document_id for item in result} == {a.document_id}
    assert pipeline.allowed == {manager.store.get(a.document_id).converted_path}
    tool_result = RAGTool(adapter, policy).run_tool({
        "query": "method", "document_ids": [b.document_id], "allowed_file_paths": None,
    })
    assert "SECRET B" not in tool_result
    assert "Method A" in tool_result


def test_workspace_reads_entire_catalog_and_public_excludes_local(library):
    manager, adapter, a, b, source, pipeline = library
    result = asyncio.run(adapter.retrieve("method", policy=SourcePolicy(source_scope=SourceScope.WORKSPACE)))
    assert {item.document_id for item in result} == {a.document_id, b.document_id}
    assert asyncio.run(adapter.retrieve("method", policy=SourcePolicy(source_scope=SourceScope.PUBLIC, allow_web=True))) == []


def test_invalid_or_empty_selected_ids_do_not_leak_workspace(library):
    _, adapter, a, b, source, pipeline = library
    for ids in [(), ("missing",), ("same title",), (str(source),)]:
        policy = SourcePolicy(source_scope=SourceScope.SELECTED_DOCUMENTS, document_ids=ids)
        assert asyncio.run(adapter.retrieve("method", policy=policy)) == []


def test_evidence_versions_locations_and_ids_survive_reload(library):
    manager, adapter, a, b, source, pipeline = library
    policy = SourcePolicy(source_scope=SourceScope.SELECTED_DOCUMENTS, document_ids=(a.document_id,))
    first = asyncio.run(adapter.read_documents(policy=policy))
    again = asyncio.run(adapter.read_documents(policy=policy))
    assert [item.evidence_id for item in first] == [item.evidence_id for item in again]
    item = first[-1]
    saved = JsonEvidenceStore(manager.data_dir).get(item.evidence_id)
    assert saved == item
    assert saved.document_id == a.document_id
    assert saved.source_version == a.document_version
    assert saved.parser_version == "markdown-adapter-v1"
    assert saved.section == "Limitations"
    assert saved.page is None
    text = Path(saved.metadata["converted_path"]).read_text()
    assert text[saved.location["char_start"]:saved.location["char_end"]] == saved.original_text


def test_failed_replacement_preserves_ready_old_version(library):
    manager, adapter, a, b, source, pipeline = library
    old = manager.store.get(a.document_id)
    pipeline.fail = True
    with pytest.raises(RuntimeError, match="embedding failed"):
        manager.ingest(str(source))
    current = manager.store.get(a.document_id)
    assert current.converted_path == old.converted_path
    assert current.ingestion_status == IngestionStatus.READY
    assert "embedding failed" in current.metadata["ingestion_error"]
    assert any(chunk.file_path == old.converted_path for chunk in pipeline.vector_store.chunks)


def test_new_failed_ingestion_is_recorded(tmp_path):
    pipeline = LocalPipeline()
    pipeline.fail = True
    source = tmp_path / "fail.md"
    source.write_text("test")
    manager = LibraryManager(str(tmp_path / "data"), None, None, pipeline)
    with pytest.raises(RuntimeError):
        manager.ingest(str(source))
    record = manager.store.list()[0]
    assert record.ingestion_status == IngestionStatus.FAILED
    assert record.created_at and record.updated_at
    assert record.metadata["ingestion_error"] == "embedding failed"


def test_legacy_index_migrates_once_without_changing_path(tmp_path):
    converted = tmp_path / "old.md"
    converted.write_text("legacy paper")
    (tmp_path / "index.json").write_text(json.dumps([
        {"title": "old", "converted_path": str(converted), "source_type": "markdown"},
    ]))
    first = JsonDocumentStore(str(tmp_path)).list()[0]
    reloaded = JsonDocumentStore(str(tmp_path)).list()[0]
    assert first.document_id == reloaded.document_id
    assert first.converted_path == str(converted)
    assert first.parser_version == "legacy-markdown"


def test_vector_store_restricts_both_recall_branches_before_scoring():
    store = VectorStore()
    store.add_chunks([
        Chunk(file_path="allowed", chunk_idx=0, content="less similar", tokens=["local"], embedding=[0.1, 0.9]),
        Chunk(file_path="forbidden", chunk_idx=0, content="perfect", tokens=["perfect"], embedding=[1.0, 0.0]),
    ])
    hits = store.hybrid_search("perfect", [1.0, 0.0], top_k=1, allowed_file_paths={"allowed"})
    assert len(hits) == 1
    assert hits[0][0].file_path == "allowed"


def test_evidence_identity_does_not_depend_on_retrieval_order(library):
    manager, adapter, a, b, source, pipeline = library
    policy = SourcePolicy(source_scope=SourceScope.WORKSPACE)
    before = asyncio.run(adapter.retrieve("method", policy=policy))
    pipeline.vector_store.chunks.reverse()
    after = asyncio.run(adapter.retrieve("method", policy=policy))
    assert {item.evidence_id for item in before} == {item.evidence_id for item in after}
    assert before[0].document_id != after[0].document_id
