import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sage_scholar.rag.document import Chunk
from sage_scholar.rag.persistent_store import PersistentStore, open_index
from sage_scholar.rag.vector_store import VectorStore


def chunk(path='paper',index=0,text='student distillation method',vector=None):
    return Chunk(file_path=path,chunk_idx=index,content=text,embedding=vector or [1.,0.,0.],
        schema_version=2,parser_version='pymupdf-structure-v2',page_start=2,section='Method',
        metadata={'label':'Figure 2','bbox':[1,2,3,4]})


@pytest.mark.parametrize('threshold', [1,2048])
def test_fts_faiss_rrf_scope_and_stable_ids(tmp_path,threshold):
    store=PersistentStore(tmp_path/'rag.sqlite3',hnsw_threshold=threshold)
    store.add_chunks([chunk('allowed'),chunk('secret',text='secret student',vector=[1.,.01,0])])
    store.save()
    original_id=store.db.execute("SELECT id FROM chunks WHERE file_path='allowed'").fetchone()[0]
    statements=[];store.db.set_trace_callback(statements.append)
    for _ in range(2):
        hits=store.hybrid_search('student',[1.,0,0],5,allowed_file_paths={'allowed'})
        assert [c.file_path for c,_ in hits]==['allowed']
    assert not any('CREATE' in s or 'INSERT' in s for s in statements)
    assert store.hybrid_search('student',[1.,0,0],5,allowed_file_paths=set())==[]
    store.replace_file('allowed',[chunk('allowed',text='Revised student')],save_path=True)
    assert store.db.execute("SELECT id FROM chunks WHERE file_path='allowed'").fetchone()[0]==original_id
    store.remove_by_filepath('secret');store.save();store.close()
    reopened=PersistentStore(tmp_path/'rag.sqlite3')
    assert len(reopened.chunks)==1 and reopened.chunks[0].metadata['bbox']==[1,2,3,4]
    assert reopened.chunks[0].page_start==2
    reopened.close()


def test_json_migration_preserves_chunks_identity_and_original_bytes(tmp_path):
    old=VectorStore();old.embedding_identity={'model':'fixture'}
    old.add_chunks([chunk('a'),chunk('b')]);path=tmp_path/'vector_store.json'
    old.save_to_json(str(path));before=path.read_bytes()
    migrated=open_index(tmp_path/'rag.sqlite3',path)
    assert migrated.embedding_identity==old.embedding_identity
    assert [c.model_dump() for c in migrated.chunks]==[c.model_dump() for c in old.chunks]
    assert path.read_bytes()==before
    migrated.remove_by_filepath('a');migrated.save();migrated.close()
    again=open_index(tmp_path/'rag.sqlite3',path)
    assert [c.file_path for c in again.chunks]==['b']  # Never re-import stale JSON.
    again.close()


def test_missing_and_corrupt_faiss_snapshot_rebuild_without_model(tmp_path):
    store=PersistentStore(tmp_path/'rag.sqlite3');store.add_chunks([chunk()]);store.save()
    snapshot=store._snapshot();store.close();snapshot.write_bytes(b'corrupt')
    restored=PersistentStore(tmp_path/'rag.sqlite3')
    assert restored.cosine_search([1,0,0],1)[0][0].content==chunk().content
    assert restored._snapshot().read_bytes()!=b'corrupt'
    restored.close()


def test_failed_snapshot_save_retains_previous_committed_index(tmp_path,monkeypatch):
    import sage_scholar.rag.persistent_store as module
    store=PersistentStore(tmp_path/'rag.sqlite3');store.add_chunks([chunk()]);store.save()
    revision=store.index_revision
    def fail(*args):
        raise OSError('snapshot write failed')
    monkeypatch.setattr(module.faiss,'write_index',fail)
    with pytest.raises(OSError,match='snapshot write failed'):
        store.replace_file('paper',[chunk(text='new result')],save_path=True)
    assert store.index_revision==revision and store.chunks[0].content==chunk().content
    fresh=PersistentStore(tmp_path/'rag.sqlite3')
    assert fresh.chunks[0].content==chunk().content
    fresh.close();store.close()


def test_staged_transaction_death_and_schema_validation(tmp_path):
    path=tmp_path/'rag.sqlite3';store=PersistentStore(path)
    store.add_chunks([chunk()]);store.save()
    store.add_chunks([chunk('uncommitted')]);store.close()  # Connection death rolls SQL back.
    restored=PersistentStore(path);assert len(restored.chunks)==1
    restored.set_meta('schema',999);restored.db.commit();restored.close()
    with pytest.raises(ValueError,match='schema'):
        PersistentStore(path)


def test_query_language_does_not_filter_out_dense_english_source(tmp_path):
    store=PersistentStore(tmp_path/'rag.sqlite3')
    store.add_chunks([chunk(text='On policy distillation student trajectories')]);store.save()
    assert store.bm25_search('学生轨迹',5)==[]
    hits=store.hybrid_search('学生轨迹',[1,0,0],5)
    assert hits[0][0].file_path=='paper'  # Same embedding space is preserved; no lexical gate.
    store.close()


def test_catalog_binding_and_history_orphan_recovery(tmp_path):
    store=PersistentStore(tmp_path/'rag.sqlite3')
    doc_id='doc_'+'a'*32
    paths=[str(tmp_path/'converted'/doc_id/(letter*32)/'document.md') for letter in 'bcd']
    store.add_chunks([chunk(p) for p in paths]+[chunk('external')]);store.save()
    versions=[SimpleNamespace(converted_path=p,document_version=f'v{i}') for i,p in enumerate(paths[:2])]
    doc=SimpleNamespace(document_id=doc_id,converted_path=paths[1],versions=versions)
    store.reconcile_library(tmp_path,[doc],save=store.save);store.bind_catalog([doc])
    assert store.indexed_paths()=={*paths[:2],'external'}
    assert store.db.execute('SELECT document_version FROM chunks WHERE file_path=?',(paths[0],)).fetchone()[0]=='v0'
    store.close()


def test_invalid_embeddings_and_empty_match_do_not_damage_index(tmp_path):
    store=PersistentStore(tmp_path/'rag.sqlite3');store.add_chunks([chunk()]);store.save()
    for vector in ([float('nan'),0,0],[1,0]):
        with pytest.raises(ValueError):
            store.replace_file('paper',[chunk(vector=vector)],save_path=True)
    assert store.bm25_search('" * ()',5)==[]
    assert store.chunks[0].content==chunk().content
    store.close()


def test_failed_first_write_keeps_model_identity_and_sql_changes_atomic(tmp_path,monkeypatch):
    import sage_scholar.rag.persistent_store as module
    store=PersistentStore(tmp_path/'rag.sqlite3')
    store.embedding_identity={'model':'test'}
    assert not store.db.in_transaction
    original=module.tokenize
    def fail(text):
        raise RuntimeError('tokenizer failure after metadata insert')
    monkeypatch.setattr(module,'tokenize',fail)
    with pytest.raises(RuntimeError,match='tokenizer failure'):
        store.replace_file('paper',[chunk()],save_path=True)
    assert len(store.chunks)==0 and store.embedding_identity=={'model':'test'}
    monkeypatch.setattr(module,'tokenize',original)
    store.replace_file('paper',[chunk()],save_path=True)
    store.close()
    again=PersistentStore(tmp_path/'rag.sqlite3')
    assert again.embedding_identity=={'model':'test'} and len(again.chunks)==1
    again.close()
