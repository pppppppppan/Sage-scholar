"""SQLite is authoritative; revision-named FAISS snapshots are recoverable derivatives.

One process/shared instance, as with the existing application. Staged writes are
visible on this connection and published by save(); catalog controls visibility.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
from collections.abc import Sequence

import faiss
import jieba
import numpy as np

from .document import Chunk
from .vector_store import VectorStore, locked

SCHEMA = 3


def tokenize(text):
    return [word.lower() for word in jieba.lcut(text) if any(c.isalnum() for c in word)]


class ChunkView(Sequence):
    def __init__(self, store):
        self.store = store
    def __len__(self):
        with self.store.lock:
            return self.store.db.execute('SELECT count(*) FROM chunks').fetchone()[0]
    def __iter__(self):
        with self.store.lock:
            rows = self.store.db.execute('SELECT payload,vector FROM chunks ORDER BY id')
            for row in rows:
                yield self.store.decode(row)
    def __eq__(self, other):
        return list(self) == list(other)
    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)[index]
        with self.store.lock:
            if index < 0:
                index += len(self)
            row = self.store.db.execute('SELECT payload,vector FROM chunks ORDER BY id LIMIT 1 OFFSET ?', (index,)).fetchone()
            if row is None:
                raise IndexError(index)
            return self.store.decode(row)


class PersistentStore(VectorStore):
    """Reuse the existing RRF/reconciliation contract, not the legacy search engine."""
    def __init__(self, path, *, hnsw_threshold=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cleanup_pending = False
        self.threshold = hnsw_threshold if hnsw_threshold is not None else int(os.getenv("RAG_HNSW_THRESHOLD", "2048"))
        faiss.omp_set_num_threads(max(1, int(os.getenv("FAISS_NUM_THREADS", "1"))))
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        existing_meta = self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
        if existing_meta and self.meta('schema') not in {None, SCHEMA}:
            self.db.close()
            raise ValueError('未知持久化 RAG schema；请使用兼容版本或明确重建索引')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL UNIQUE, file_path TEXT NOT NULL, chunk_idx INTEGER NOT NULL,
                document_id TEXT, document_version TEXT, payload TEXT NOT NULL, vector BLOB NOT NULL,
                UNIQUE(file_path,chunk_idx));
            CREATE INDEX IF NOT EXISTS chunk_path ON chunks(file_path);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='unicode61');
        ''')
        version = self.meta('schema')
        if version is not None and version != SCHEMA:
            self.db.close()
            raise ValueError('未知持久化 RAG schema；请使用兼容版本或明确重建索引')
        if version is None:
            self.set_meta('schema', SCHEMA)
            self.db.commit()
        self.index_revision = self.meta('revision') or 0
        self.chunks = ChunkView(self)
        self._index = None
        self._ids = np.empty(0, dtype='int64')
        self._positions = {}
        self._dirty = False
        self._dense_dirty = False
        self._load_dense()

    def meta(self, key):
        row = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, json.dumps(value)))

    @property
    def embedding_identity(self):
        return self.meta('embedding_identity')

    @embedding_identity.setter
    def embedding_identity(self, value):
        with self.lock:
            if self.embedding_identity != value:
                self.set_meta('embedding_identity', value)
                # Metadata-only setup must not hold a SQLite write transaction
                # while the service is idle, or be lost on the first failed import.
                if not self._dirty:
                    self.db.commit()

    @property
    def dimension(self):
        row = self.db.execute('SELECT length(vector)/4 FROM chunks LIMIT 1').fetchone()
        return row[0] if row else None

    @staticmethod
    def decode(row):
        return Chunk.model_validate({**json.loads(row[0]), 'embedding': np.frombuffer(row[1], dtype='float32').tolist()})

    def _snapshot(self, revision=None):
        return self.path.with_name(f'{self.path.stem}-faiss-{self.index_revision if revision is None else revision}.index')

    def _build_dense(self):
        rows = self.db.execute('SELECT id,vector FROM chunks ORDER BY id').fetchall()
        ids = np.asarray([row[0] for row in rows], dtype='int64')
        if rows:
            matrix = np.stack([np.frombuffer(row[1], dtype='float32') for row in rows])
            faiss.normalize_L2(matrix)
            index = (faiss.IndexHNSWFlat(matrix.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
                     if len(rows) >= self.threshold else faiss.IndexFlatIP(matrix.shape[1]))
            if hasattr(index, 'hnsw'):
                index.hnsw.efConstruction = 80
                index.hnsw.efSearch = 128
            index.add(matrix)
        else:
            index = None
        self._index, self._ids = index, ids
        self._positions = {int(identity): pos for pos, identity in enumerate(ids)}
        self._dense_dirty = False

    def _load_dense(self):
        ids = np.asarray([r[0] for r in self.db.execute('SELECT id FROM chunks ORDER BY id')], dtype='int64')
        snapshot = self._snapshot()
        expected = self.meta('faiss_sha256')
        try:
            if len(ids) and expected and hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected:
                self._index = faiss.read_index(str(snapshot))
                if self._index.ntotal != len(ids) or self._index.d != self.dimension:
                    raise ValueError('snapshot shape mismatch')
                self._ids = ids
                self._positions = {int(v): i for i, v in enumerate(ids)}
                return
        except (OSError, RuntimeError, ValueError):
            pass
        self._build_dense()  # Repair from stored embeddings, never reparse/re-embed PDFs.
        if len(ids):
            self._dirty = True
            self.save()

    @locked
    def save(self):
        if not self._dirty and not self.cleanup_pending:
            return
        if self._dense_dirty:
            self._build_dense()
        revision = self.index_revision + 1
        target = self._snapshot(revision)
        if self._index is not None:
            temporary = target.with_suffix('.tmp')
            try:
                faiss.write_index(self._index, str(temporary))
                with temporary.open('rb') as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
                self.set_meta('faiss_sha256', hashlib.sha256(target.read_bytes()).hexdigest())
            finally:
                temporary.unlink(missing_ok=True)
        else:
            self.set_meta('faiss_sha256', None)
        self.set_meta('revision', revision)
        self.db.commit()  # Only now is the new snapshot authoritative.
        self.index_revision = revision
        self._dirty = self.cleanup_pending = False
        for old in self.path.parent.glob(f'{self.path.stem}-faiss-*.index'):
            if old != target:
                try:
                    old.unlink()
                except OSError:
                    pass  # Harmless derivative; not referenced by any query.

    @locked
    def replace_file(self, file_path, chunks, *, save_path=None):
        chunks = list(chunks)
        dimension = self.dimension
        for c in chunks:
            if c.file_path != file_path or c.schema_version not in {1, 2}:
                raise ValueError('Chunk 路径/schema 不一致')
            with np.errstate(over='ignore', invalid='ignore'):
                finite = np.isfinite(np.asarray(c.embedding, dtype='float32')).all()
            if not c.embedding or (dimension and len(c.embedding) != dimension) or not finite:
                raise ValueError('Embedding 维度或数值无效；保留原索引')
            dimension = len(c.embedding)
        if len({c.chunk_idx for c in chunks}) != len(chunks):
            raise ValueError('重复 Chunk index')
        previous_dirty = self._dirty
        if not self.db.in_transaction:
            self.db.execute('BEGIN')
        self.db.execute('SAVEPOINT replace_file')
        try:
            old = {r[1]: r[0] for r in self.db.execute('SELECT id,chunk_idx FROM chunks WHERE file_path=?', (file_path,))}
            for c in chunks:
                stable = hashlib.sha256(f'{file_path}\0{c.chunk_idx}'.encode()).hexdigest()
                payload = c.model_dump(exclude={'embedding'})
                self.db.execute('''INSERT INTO chunks(chunk_id,file_path,chunk_idx,payload,vector) VALUES (?,?,?,?,?)
                    ON CONFLICT(file_path,chunk_idx) DO UPDATE SET payload=excluded.payload,vector=excluded.vector''',
                    (stable, file_path, c.chunk_idx, json.dumps(payload, ensure_ascii=False), np.asarray(c.embedding,dtype='float32').tobytes()))
                identity = self.db.execute('SELECT id FROM chunks WHERE file_path=? AND chunk_idx=?', (file_path,c.chunk_idx)).fetchone()[0]
                self.db.execute('DELETE FROM chunks_fts WHERE rowid=?', (identity,))
                self.db.execute('INSERT INTO chunks_fts(rowid,text) VALUES (?,?)', (identity, ' '.join(tokenize(c.content))))
                old.pop(c.chunk_idx, None)
            for identity in old.values():
                self.db.execute('DELETE FROM chunks_fts WHERE rowid=?', (identity,))
                self.db.execute('DELETE FROM chunks WHERE id=?', (identity,))
            self._dirty = self._dense_dirty = True
            if save_path is not None:
                self.save()  # Commits the outer transaction and releases savepoint.
            else:
                self.db.execute('RELEASE replace_file')
        except BaseException as original:
            try:
                self.db.execute('ROLLBACK TO replace_file')
                self.db.execute('RELEASE replace_file')
                if not previous_dirty:
                    self.db.rollback()
                self._dirty = previous_dirty
                self._build_dense()
            except Exception as recovery_error:
                self.cleanup_pending = True
                original.add_note(f'索引恢复失败，需重新打开: {recovery_error}')
            raise

    @locked
    def add_chunks(self, chunks):
        groups = {}
        for chunk in chunks:
            groups.setdefault(chunk.file_path, []).append(chunk)
        for path, group in groups.items():
            self.replace_file(path, group)

    @locked
    def remove_by_filepath(self, file_path):
        ids = [r[0] for r in self.db.execute('SELECT id FROM chunks WHERE file_path=?', (file_path,))]
        for identity in ids:
            self.db.execute('DELETE FROM chunks_fts WHERE rowid=?', (identity,))
        self.db.execute('DELETE FROM chunks WHERE file_path=?', (file_path,))
        if ids:
            self._dirty = self._dense_dirty = True
        return len(ids)

    @locked
    def indexed_paths(self):
        return {r[0] for r in self.db.execute("SELECT DISTINCT file_path FROM chunks")}

    @locked
    def bind_catalog(self, documents):
        for doc in documents:
            for version in doc.versions:
                self.db.execute('UPDATE chunks SET document_id=?, document_version=? WHERE file_path=? AND (document_id IS NOT ? OR document_version IS NOT ?)',
                    (doc.document_id, version.document_version, version.converted_path, doc.document_id, version.document_version))
        # Never commit pending index changes without their FAISS snapshot.
        if not self._dirty:
            self.db.commit()

    def _scope(self, paths):
        # JSON1 avoids SQLite variable limits for large workspaces.
        if paths is None:
            return '', []
        return ' AND c.file_path IN (SELECT value FROM json_each(?))', [json.dumps(sorted(paths))]

    @locked
    def bm25_search(self, query, top_k, allowed_file_paths=None):
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or top_k <= 0:
            return []
        expression = ' OR '.join('"' + word.replace('"','""') + '"' for word in terms)
        scope, args = self._scope(allowed_file_paths)
        rows = self.db.execute('SELECT c.payload,c.vector,bm25(chunks_fts) FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid WHERE chunks_fts MATCH ?' + scope + ' ORDER BY bm25(chunks_fts),c.id LIMIT ?', [expression,*args,top_k]).fetchall()
        return [(self.decode(r), -r[2]) for r in rows]

    @locked
    def cosine_search(self, query_embedding, top_k, allowed_file_paths=None):
        if self._dense_dirty:
            self._build_dense()
        if self._index is None or top_k <= 0:
            return []
        vector = np.asarray([query_embedding], dtype='float32')
        if vector.shape[1] != self._index.d or not np.isfinite(vector).all():
            raise ValueError('查询向量维度或数值无效')
        faiss.normalize_L2(vector)
        params = faiss.SearchParametersHNSW() if hasattr(self._index,'hnsw') else faiss.SearchParameters()
        if hasattr(params,'efSearch'):
            params.efSearch = max(128, top_k * 4)
            params.bounded_queue = True
        if allowed_file_paths is not None:
            scope, args = self._scope(allowed_file_paths)
            ids = [r[0] for r in self.db.execute('SELECT c.id FROM chunks c WHERE 1=1' + scope, args)]
            positions = np.asarray([self._positions[i] for i in ids], dtype='int64')
            if not len(positions):
                return []
            selector = faiss.IDSelectorBatch(positions)
            params.sel = selector
        scores, positions = self._index.search(vector, min(top_k, len(self._ids)), params=params)
        result = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0:
                continue
            row = self.db.execute('SELECT payload,vector,file_path FROM chunks WHERE id=?', (int(self._ids[pos]),)).fetchone()
            if row and (allowed_file_paths is None or row[2] in allowed_file_paths):
                result.append((self.decode(row), float(score)))
        return result

    def save_to_json(self, file_path):
        raise ValueError('新索引不写 JSON；使用 save()。JSON 仅作一次性迁移输入')

    def close(self):
        with self.lock:
            self.db.rollback()
            self.db.close()


def open_index(path, legacy_path=None):
    from sage_scholar.infrastructure.json_store import file_lock
    with file_lock(Path(path).parent / ".ingestion"):
        return _open_index(path, legacy_path)


def _open_index(path, legacy_path=None):
    """Atomic one-time migration; legacy source stays untouched as a backup."""
    path = Path(path)
    if path.exists() or not legacy_path or not Path(legacy_path).exists():
        return PersistentStore(path)
    legacy = VectorStore.load_json(str(legacy_path))
    temporary = path.with_name(path.stem + '-migration.sqlite3')
    # Retry an interrupted migration from the untouched source.
    for item in [temporary, Path(str(temporary)+'-wal'), Path(str(temporary)+'-shm'), *path.parent.glob(temporary.stem+'-faiss-*')]:
        item.unlink(missing_ok=True)
    migrated = PersistentStore(temporary)
    try:
        migrated.embedding_identity = legacy.embedding_identity
        migrated.add_chunks(legacy.chunks)
        migrated.save()
        migrated.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        migrated.close()
        # Rename the derivative before publishing the SQLite authority.
        for snapshot in path.parent.glob(temporary.stem+'-faiss-*.index'):
            os.replace(snapshot, path.with_name(snapshot.name.replace(temporary.stem, path.stem, 1)))
        os.replace(temporary, path)
    except BaseException:
        try:
            migrated.close()
        except sqlite3.ProgrammingError:
            pass
        raise
    return PersistentStore(path)
