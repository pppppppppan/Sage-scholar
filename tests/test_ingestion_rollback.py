"""Real JSON indexes/catalogs and PDF parsing; no remote embeddings or MCP."""
import asyncio
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_seven_structured_rag import paper, pipeline
from sage_scholar.application.document_service import DocumentService
from sage_scholar.library.library_manager import LibraryManager
from sage_scholar.rag.pipeline import Pipeline
from sage_scholar.rag.vector_store import VectorStore
from sage_scholar.rag.document import Chunk
from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
from sage_scholar.config import Config
from sage_scholar.domain.execution import SourcePolicy


class PDFTool:
    def run_tool(self, parameters):
        return '# Parsed paper\nFallback text.'


def manager_for(pipeline, visual=None):
    return LibraryManager(pipeline.data_dir, None, PDFTool(), pipeline, visual)


def paths(store):
    return {c.file_path for c in store.chunks}


def disk(pipeline):
    return VectorStore.load_json(str(Path(pipeline.data_dir) / pipeline.VECTOR_STORE_FILE))


def seed(pipeline, tmp_path):
    source = tmp_path / 'old.md'
    source.write_text('# Findings\nOld experiment findings.')
    manager = manager_for(pipeline)
    result = manager.ingest(str(source))
    source.write_text('# Findings\nCurrent experiment findings.')
    manager.ingest(str(source))
    external = tmp_path / 'non-library.md'
    external.write_text('# External\nIndependent search results.')
    pipeline.add_document(str(external))
    return manager, result.document_id, paths(pipeline.vector_store)


@pytest.mark.parametrize('failure', ['visual', 'catalog', 'cancelled', 'system_exit'])
def test_failure_after_index_save_rolls_back_disk_and_preserves_all_versions(pipeline, paper, tmp_path, monkeypatch, failure):
    manager, doc_id, before = seed(pipeline, tmp_path)
    old = manager.store.get(doc_id)
    adapter = LegacyRetrievalAdapter(Config(data_dir=pipeline.data_dir), pipeline)
    policy = SourcePolicy(source_scope='selected_documents', document_ids=(doc_id,))
    evidence = adapter.retrieve_sync('experiment', policy)[0]
    revision = pipeline.vector_store.index_revision
    exception = {'cancelled': asyncio.CancelledError, 'system_exit': SystemExit}.get(failure, RuntimeError)
    def fail(*args):
        assert paths(disk(pipeline)) > before  # The problematic durability boundary is reached.
        raise exception('original import failure')
    if failure == 'catalog':
        original = manager.store.put
        def put(doc):
            if doc.active_version == 'v3':
                fail()
            return original(doc)
        monkeypatch.setattr(manager.store, 'put', put)
    else:
        manager.visual = SimpleNamespace(build=fail)
    with pytest.raises(exception, match='original import failure'):
        manager.ingest(str(paper), document_id=doc_id)
    assert paths(pipeline.vector_store) == paths(disk(pipeline)) == before
    assert pipeline.vector_store.index_revision == disk(pipeline).index_revision > revision
    assert manager.store.get(doc_id).versions == old.versions
    assert manager.store.get(doc_id).active_version == 'v2'
    assert adapter.evidence_store.get(evidence.evidence_id) == evidence
    assert all(e.source_version == 'v2' for e in adapter.retrieve_sync('experiment', policy))
    for version in old.versions:
        assert pipeline.retrieve_chunks('experiment', allowed_file_paths={version.converted_path})
    assert paths(Pipeline(pipeline.data_dir).vector_store) == before


@pytest.mark.parametrize("recovery", ["same_process", "restart"])
def test_failed_rollback_does_not_mask_error_and_retries_on_restart(pipeline, paper, tmp_path, monkeypatch, recovery):
    manager, doc_id, before = seed(pipeline, tmp_path)
    save = pipeline.save
    def build(*args):
        monkeypatch.setattr(pipeline, 'save', lambda: (_ for _ in ()).throw(OSError('disk unavailable')))
        raise RuntimeError('visual failed first')
    manager.visual = SimpleNamespace(build=build)
    with pytest.raises(RuntimeError, match='visual failed first') as error:
        manager.ingest(str(paper), document_id=doc_id)
    assert 'disk unavailable' in '\n'.join(error.value.__notes__)
    assert paths(pipeline.vector_store) == before and pipeline.vector_store.cleanup_pending
    assert paths(disk(pipeline)) > before
    if recovery == "restart":
        restored = Pipeline(pipeline.data_dir)
        assert paths(restored.vector_store) == paths(disk(restored)) == before
        return
    # Retry with the same instance must flush the pending revision even with zero orphans in memory.
    monkeypatch.setattr(pipeline, 'save', save)
    manager.ingest(str(tmp_path / 'old.md'), document_id=doc_id, overwrite=False)
    assert paths(disk(pipeline)) == before and not pipeline.vector_store.cleanup_pending


def test_actual_process_exit_leaves_orphan_then_pipeline_startup_recovers(pipeline, tmp_path):
    manager, doc_id, before = seed(pipeline, tmp_path)
    orphan = str(Path(pipeline.data_dir) / 'converted' / doc_id / ('a' * 32) / 'document.md')
    # No Python finally/exception handler runs in this child.
    code = '''import os,sys
from sage_scholar.rag.vector_store import VectorStore
from sage_scholar.rag.document import Chunk
s=VectorStore.load_json(sys.argv[1])
s.replace_file(sys.argv[2], [Chunk(file_path=sys.argv[2], chunk_idx=0, content='uncommitted', tokens=['uncommitted'], embedding=[1.,0.])], save_path=sys.argv[1])
os._exit(19)
'''
    result = subprocess.run([sys.executable, '-c', code, str(Path(pipeline.data_dir) / pipeline.VECTOR_STORE_FILE), orphan], capture_output=True)
    assert result.returncode == 19, result.stderr.decode()
    assert orphan in paths(disk(pipeline))
    restored = Pipeline(pipeline.data_dir)
    assert paths(restored.vector_store) == paths(disk(restored)) == before
    assert len(manager.store.get(doc_id).versions) == 2
    # Reconciliation is idempotent, with no spurious revision bumps.
    revision = restored.vector_store.index_revision
    assert Pipeline(pipeline.data_dir).vector_store.index_revision == revision


def test_catalog_committed_then_exception_does_not_erase_success(pipeline, paper, tmp_path, monkeypatch):
    manager, doc_id, before = seed(pipeline, tmp_path)
    put = manager.store.put
    def commit_then_error(doc):
        put(doc)
        if doc.active_version == 'v3':
            raise OSError('after replace')
    monkeypatch.setattr(manager.store, 'put', commit_then_error)
    with pytest.raises(OSError, match='after replace'):
        manager.ingest(str(paper), document_id=doc_id)
    current = manager.store.get(doc_id)
    assert current.active_version == 'v3' and current.ingestion_status == 'ready'
    assert paths(pipeline.vector_store) == paths(disk(pipeline)) > before
    assert paths(Pipeline(pipeline.data_dir).vector_store) == paths(pipeline.vector_store)


def test_async_cancel_joins_import_and_cleans_before_return(pipeline, paper, tmp_path):
    manager, doc_id, before = seed(pipeline, tmp_path)
    entered, release = threading.Event(), threading.Event()
    def build(*args):
        entered.set()
        assert release.wait(10)
        return {'status': 'ready'}
    manager.visual = SimpleNamespace(build=build)
    service = DocumentService(SimpleNamespace(get_library_manager=lambda _: manager), pipeline.data_dir, lambda: pipeline)
    async def scenario():
        task = asyncio.create_task(service.ingest(str(paper), document_id=doc_id))
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()  # Worker still owns the transaction.
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert paths(pipeline.vector_store) == paths(disk(pipeline)) == before
    assert manager.store.get(doc_id).active_version == 'v2'


def test_successful_pdf_and_delete_keep_external_index(pipeline, paper, tmp_path):
    manager, doc_id, before = seed(pipeline, tmp_path)
    manager.visual = SimpleNamespace(build=lambda *a: {'status': 'ready', 'images': 1})
    result = manager.ingest(str(paper), document_id=doc_id)
    assert result.document_version == 'v3'
    doc = manager.store.get(doc_id)
    assert doc.metadata['visual']['status'] == 'ready'
    assert doc.converted_path in paths(disk(pipeline))
    assert any(c.page_start for c in pipeline.vector_store.chunks if c.file_path == doc.converted_path)
    manager.delete_doc(doc_id)
    assert paths(pipeline.vector_store) == paths(disk(pipeline)) == {str(tmp_path / 'non-library.md')}
    assert paths(Pipeline(pipeline.data_dir).vector_store) == paths(disk(pipeline))
