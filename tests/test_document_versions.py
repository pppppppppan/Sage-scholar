import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading

import pytest
from fastapi.testclient import TestClient

from sage_scholar.api.app import create_app
from sage_scholar.application.document_service import DocumentService
from sage_scholar.domain.documents import IngestionStatus
from sage_scholar.domain.execution import SourcePolicy
from sage_scholar.infrastructure.document_store import JsonDocumentStore
from sage_scholar.infrastructure.evidence_store import JsonEvidenceStore
from sage_scholar.library.library_manager import LibraryManager
from sage_scholar.routing import SourceScope
from test_api import FakeContainer
from test_document_evidence import LocalPipeline, library


def test_unchanged_source_keeps_version_and_does_not_reindex(library):
    manager, _, a, _, source, pipeline = library
    old = manager.store.get(a.document_id)
    before = len(pipeline.vector_store.chunks)
    result = manager.ingest(str(source), custom_title="renamed")
    current = manager.store.get(a.document_id)
    assert result.document_id == a.document_id
    assert result.document_version == old.document_version
    assert result.status == "skipped"
    assert current.versions == old.versions
    assert current.title == "renamed"
    assert len(pipeline.vector_store.chunks) == before


def test_new_content_keeps_document_identity_and_increments_version(library):
    manager, _, a, _, source, _ = library
    old = manager.store.get(a.document_id)
    source.write_text("# Revised\nnew experiment")
    result = manager.ingest(str(source))
    current = manager.store.get(a.document_id)
    assert result.document_id == a.document_id
    assert current.document_version == current.active_version == "v2"
    assert current.content_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    assert current.content_hash != old.content_hash
    assert [item.version_number for item in current.versions] == [1, 2]
    assert current.versions[0].converted_path == old.converted_path
    assert Path(old.converted_path).is_file()


def test_identical_content_and_titles_do_not_merge_different_sources(tmp_path):
    manager = LibraryManager(str(tmp_path / "data"), None, None, LocalPipeline())
    sources = [tmp_path / "paper-final.md", tmp_path / "paper-final-v2.md"]
    for source in sources:
        source.write_text("same bytes")
    a, b = [manager.ingest(str(source), custom_title="same title") for source in sources]
    assert a.document_id != b.document_id
    assert a.content_hash == b.content_hash


def test_historical_evidence_and_active_retrieval_are_separate(library):
    manager, adapter, a, _, source, _ = library
    policy = SourcePolicy(source_scope=SourceScope.SELECTED_DOCUMENTS, document_ids=(a.document_id,))
    old = asyncio.run(adapter.read_documents(policy=policy))[0]
    source.write_text("# NEW\nnew findings")
    manager.ingest(str(source))
    current = manager.store.get(a.document_id)
    hits = asyncio.run(adapter.retrieve("findings", policy=SourcePolicy(source_scope=SourceScope.WORKSPACE)))
    selected = [hit for hit in hits if hit.document_id == a.document_id]
    assert selected and all(hit.source_version == current.active_version for hit in selected)
    assert all("Method A" not in hit.original_text for hit in selected)
    historic = JsonEvidenceStore(manager.data_dir).get(old.evidence_id)
    assert historic == old
    assert historic.source_version == "v1"
    assert Path(historic.metadata["converted_path"]).read_text().startswith("# Method")
    assert historic.parser_version == old.parser_version
    assert historic.location == old.location


@pytest.mark.parametrize("phase", ["parsing", "indexing"])
def test_library_api_observes_in_progress_status_without_waiting(tmp_path, monkeypatch, phase):
    import sage_scholar.library.library_manager as module
    source = tmp_path / "paper.md"
    source.write_text("paper")
    pipeline = LocalPipeline()
    reached, release = threading.Event(), threading.Event()
    if phase == "parsing":
        original = module.convert_to_markdown
        def blocked(*args, **kwargs):
            reached.set()
            assert release.wait(10), "test parser was not released"
            return original(*args, **kwargs)
        monkeypatch.setattr(module, "convert_to_markdown", blocked)
    else:
        original = pipeline.add_document
        def blocked(*args, **kwargs):
            reached.set()
            assert release.wait(10), "test indexer was not released"
            return original(*args, **kwargs)
        monkeypatch.setattr(pipeline, "add_document", blocked)
    container = FakeContainer()
    container.document_service = DocumentService(None, str(tmp_path / "data"), lambda: pipeline)
    with TestClient(create_app(container)) as client, ThreadPoolExecutor(max_workers=2) as executor:
        uploading = executor.submit(client.post, "/api/library/ingest", json={"src": str(source)})
        try:
            assert reached.wait(5)
            response = executor.submit(client.get, "/api/library").result(timeout=2)
            assert response.status_code == 200
            assert response.json()[0]["ingestion_status"] == phase
            assert response.json()[0]["active_version"] is None
            assert not uploading.done()
        finally:
            release.set()
        assert uploading.result(timeout=5).status_code == 200
        assert client.get("/api/library").json()[0]["ingestion_status"] == "ready"


@pytest.mark.parametrize("phase", ["parser", "indexer", "save"])
def test_failures_record_failed_and_keep_old_active_version(library, monkeypatch, phase):
    import sage_scholar.library.library_manager as module
    manager, adapter, a, _, source, pipeline = library
    old = manager.store.get(a.document_id)
    source.write_text("new content")
    def fail(*args, **kwargs):
        raise RuntimeError(f"{phase} failed")
    if phase == "parser":
        monkeypatch.setattr(module, "convert_to_markdown", fail)
    elif phase == "indexer":
        monkeypatch.setattr(pipeline, "add_document", fail)
    else:
        monkeypatch.setattr(pipeline, "save", fail)
    with pytest.raises(RuntimeError, match=f"{phase} failed"):
        manager.ingest(str(source))
    current = manager.store.get(a.document_id)
    assert current.ingestion_status == IngestionStatus.FAILED
    assert current.active_version == old.active_version
    assert current.versions == old.versions
    assert current.metadata["ingestion_error"] == f"{phase} failed"
    assert Path(old.converted_path).is_file()
    hits = asyncio.run(adapter.retrieve("method", policy=SourcePolicy(
        source_scope=SourceScope.SELECTED_DOCUMENTS, document_ids=(a.document_id,),
    )))
    assert hits and all(hit.source_version == old.document_version for hit in hits)


def test_old_active_version_is_available_during_indexing(library, monkeypatch):
    manager, adapter, a, _, source, pipeline = library
    old = manager.store.get(a.document_id)
    source.write_text("new contents")
    reached, release = threading.Event(), threading.Event()
    original = pipeline.add_document
    def blocked(*args, **kwargs):
        original(*args, **kwargs)  # New chunks exist, but must not yet be visible.
        reached.set()
        assert release.wait(10)
    monkeypatch.setattr(pipeline, "add_document", blocked)
    with ThreadPoolExecutor() as executor:
        task = executor.submit(manager.ingest, str(source))
        try:
            assert reached.wait(5)
            assert manager.store.get(a.document_id).ingestion_status == IngestionStatus.INDEXING
            hits = adapter.retrieve_sync("method", SourcePolicy(
                source_scope=SourceScope.SELECTED_DOCUMENTS, document_ids=(a.document_id,),
            ))
            assert hits and all(item.source_version == old.document_version for item in hits)
        finally:
            release.set()
        task.result(timeout=5)


def test_two_managers_serialize_ingestion_and_release_lock_after_failure(tmp_path, monkeypatch):
    pipeline = LocalPipeline()
    data = str(tmp_path / "data")
    first = LibraryManager(data, None, None, pipeline)
    second = LibraryManager(data, None, None, pipeline)
    assert first.ingestion_lock is second.ingestion_lock
    assert first.ingestion_lock is not first.store.lock
    reached, release = threading.Event(), threading.Event()
    entered = []
    original = pipeline.add_document
    def blocked(file_path, save=True):
        entered.append(file_path)
        if len(entered) == 1:
            reached.set()
            assert release.wait(10)
            raise RuntimeError("first failed")
        return original(file_path, save)
    monkeypatch.setattr(pipeline, "add_document", blocked)
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text("first")
    b.write_text("second")
    with ThreadPoolExecutor() as executor:
        failed = executor.submit(first.ingest, str(a))
        try:
            assert reached.wait(5)
            succeeding = executor.submit(second.ingest, str(b))
            assert not succeeding.done()
            assert len(first.store.list()) == 1
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="first failed"):
            failed.result(timeout=5)
        assert succeeding.result(timeout=5).ingestion_status == IngestionStatus.READY
    assert len(entered) == 2
    assert {doc.ingestion_status for doc in first.store.list()} == {
        IngestionStatus.FAILED, IngestionStatus.READY,
    }


def test_upload_update_uses_explicit_id_not_temporary_path_or_filename(tmp_path):
    container = FakeContainer()
    container.config.data_dir = str(tmp_path / "data")
    pipeline = LocalPipeline()
    container.document_service = DocumentService(None, container.config.data_dir, lambda: pipeline)
    with TestClient(create_app(container)) as client:
        def upload(content, document_id=None):
            return client.post("/api/library/upload", files={"file": ("paper.md", content, "text/markdown")},
                               data={"document_id": document_id} if document_id else {})
        first = upload(b"v1").json()
        updated_response = upload(b"v2", first["document_id"])
        assert updated_response.status_code == 200
        updated = updated_response.json()
        assert updated["document_id"] == first["document_id"]
        assert updated["document_version"] == "v2"
        duplicate = upload(b"v2", first["document_id"]).json()
        assert duplicate["document_version"] == "v2" and duplicate["status"] == "skipped"
        independent = upload(b"v2").json()
        assert independent["document_id"] != first["document_id"]
        assert len(client.get("/api/library").json()) == 2
        assert upload(b"v3", "unknown-id").status_code == 400


def test_arxiv_revisions_share_logical_identity(tmp_path):
    class PaperTool:
        download_dir = str(tmp_path)
        def run_tool(self, parameters):
            return parameters["paper_id"]
    manager = LibraryManager(str(tmp_path / "data"), PaperTool(), None, LocalPipeline())
    first = manager.ingest("2601.12345v1")
    second = manager.ingest("2601.12345v2")
    assert first.document_id == second.document_id
    assert second.document_version == "v2"
    assert manager.store.get(first.document_id).source_identifier == "arxiv:2601.12345"


def test_legacy_version_migration_preserves_old_version_identifier(tmp_path):
    original = tmp_path / "original.md"
    original.write_text("old bytes")
    converted = tmp_path / "converted.md"
    converted.write_text("old parsed text")
    old_version = hashlib.sha256(converted.read_bytes()).hexdigest()
    (tmp_path / "index.json").write_text(json.dumps([{
        "document_id": "doc_old", "title": "old", "source_identifier": "sha256:old",
        "document_version": old_version, "parser_version": "legacy-markdown",
        "original_path": str(original), "converted_path": str(converted), "ingestion_status": "ready",
    }]))
    doc = JsonDocumentStore(str(tmp_path)).list()[0]
    assert doc.document_id == "doc_old"
    assert doc.active_version == doc.versions[0].document_version == old_version
    assert doc.content_hash == hashlib.sha256(original.read_bytes()).hexdigest()
    new_source = tmp_path / "replacement.md"
    new_source.write_text("new bytes")
    updated = LibraryManager(str(tmp_path), None, None, LocalPipeline()).ingest(
        str(new_source), document_id="doc_old",
    )
    assert updated.document_id == "doc_old" and updated.document_version == "v2"
    assert converted.read_text() == "old parsed text"


def test_catalog_commit_failure_keeps_old_active_version(library, monkeypatch):
    manager, adapter, a, _, source, _ = library
    old = manager.store.get(a.document_id)
    source.write_text("new contents")
    original = manager.store.put
    def fail_new_ready(doc):
        if doc.ingestion_status == IngestionStatus.READY and doc.active_version != old.active_version:
            raise OSError("catalog commit failed")
        return original(doc)
    monkeypatch.setattr(manager.store, "put", fail_new_ready)
    with pytest.raises(OSError, match="catalog commit failed"):
        manager.ingest(str(source))
    current = manager.store.get(a.document_id)
    assert current.active_version == old.active_version
    assert current.ingestion_status == IngestionStatus.FAILED
    hits = adapter.retrieve_sync("method", SourcePolicy(
        source_scope=SourceScope.SELECTED_DOCUMENTS, document_ids=(a.document_id,),
    ))
    assert hits and all(hit.source_version == old.document_version for hit in hits)


def test_vector_index_replace_failure_leaves_previous_json_readable(tmp_path, monkeypatch):
    from sage_scholar.infrastructure import json_store
    from sage_scholar.rag.vector_store import VectorStore
    from sage_scholar.rag.document import Chunk
    store = VectorStore()
    store.add_chunks([Chunk(file_path="old", chunk_idx=0, content="old", tokens=["old"])])
    index = tmp_path / "vector_store.json"
    store.save_to_json(str(index))
    before = index.read_bytes()
    store.add_chunks([Chunk(file_path="new", chunk_idx=0, content="new", tokens=["new"])])
    def fail_replace(*args):
        raise OSError("replace failed")
    monkeypatch.setattr(json_store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.save_to_json(str(index))
    assert index.read_bytes() == before
    assert [chunk.file_path for chunk in VectorStore.load_json(str(index)).chunks] == ["old"]
