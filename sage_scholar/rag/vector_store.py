import math
import logging
import threading
import re
from functools import wraps

import json
from pathlib import Path
from sage_scholar.infrastructure.json_store import write_json
import jieba
from rank_bm25 import BM25Okapi

from .document import Chunk


def locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return call


class VectorStore:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.index_revision = 0
        self.cleanup_pending = False
        self.embedding_identity: dict | None = None
        self.chunks: list[Chunk] = []
        self._file_index: dict[str, list[Chunk]] = {}

    @locked
    def add_chunks(self, chunks: list[Chunk]):
        for chunk in chunks:
            if chunk.schema_version not in {1, 2}:
                raise ValueError("未知 Chunk schema，请使用兼容版本或重建索引")
            if chunk.tokens is None:
                chunk.tokens = jieba.lcut(chunk.content)
            self.chunks.append(chunk)
            self._file_index.setdefault(chunk.file_path, []).append(chunk)

    def replace_file(self, file_path, chunks, *, save_path=None):
        """Prepare a complete candidate before publishing; failure retains the old version."""
        with self.lock:
            candidate = VectorStore()
            candidate.embedding_identity = self.embedding_identity
            candidate.index_revision = self.index_revision + 1
            candidate.add_chunks([c for c in self.chunks if c.file_path != file_path] + list(chunks))
            if save_path is not None:
                candidate.save_to_json(save_path)
            self.chunks, self._file_index = candidate.chunks, candidate._file_index
            self.index_revision = candidate.index_revision

    @locked
    def cosine_search(
        self, query_embedding: list[float], top_k: int,
        allowed_file_paths: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        score_pairs = []
        mod_query = math.sqrt(sum(x**2 for x in query_embedding))

        for chunk in self.chunks:
            if allowed_file_paths is not None and chunk.file_path not in allowed_file_paths:
                continue
            ebd = chunk.embedding
            if ebd is None or len(ebd) != len(query_embedding):
                raise RuntimeError("向量维度不一致，请使用同一向量模型重新建立索引。")
            dot = 0
            for qi, ei in zip(query_embedding, ebd):
                dot += qi * ei
            mod_ebd = math.sqrt(sum(x**2 for x in ebd))
            cosine_score = dot / (mod_query * mod_ebd) if mod_query and mod_ebd else 0.0
            score_pairs.append((chunk, cosine_score))

        score_pairs = sorted(score_pairs, key=lambda x: x[1], reverse=True)

        return score_pairs[:top_k]

    @locked
    def bm25_search(
        self, query: str, top_k: int,
        allowed_file_paths: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        candidates = [
            chunk for chunk in self.chunks
            if allowed_file_paths is None or chunk.file_path in allowed_file_paths
        ]
        if not candidates:
            return []
        bm25 = BM25Okapi([chunk.tokens for chunk in candidates])
        bm25_scores = bm25.get_scores(jieba.lcut(query))
        scores = []
        for chunk, bm25_score in zip(candidates, bm25_scores):
            scores.append((chunk, bm25_score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    @locked
    def hybrid_search(
        self, query: str, query_embedding: list[float], top_k: int, k: int = 60,
        allowed_file_paths: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []

        bm25_result = self.bm25_search(
            query=query, top_k=top_k * 2, allowed_file_paths=allowed_file_paths
        )
        cosine_result = self.cosine_search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            allowed_file_paths=allowed_file_paths,
        )
        result: dict[tuple[str, int], tuple[Chunk, float]] = {}

        for i, pair in enumerate(bm25_result):
            result[(pair[0].file_path, pair[0].chunk_idx)] = (pair[0], 1 / (k + i))
        for i, pair in enumerate(cosine_result):
            if (pair[0].file_path, pair[0].chunk_idx) in result:
                chunk, old_score = result[(pair[0].file_path, pair[0].chunk_idx)]
                score = old_score + 1 / (k + i)
                result[(pair[0].file_path, pair[0].chunk_idx)] = (chunk, score)
            else:
                result[(pair[0].file_path, pair[0].chunk_idx)] = (pair[0], 1 / (k + i))

        result = sorted(result.values(), key=lambda x: x[1], reverse=True)

        return result[:top_k]

    @locked
    def save_to_json(self, file_path: str):
        chunk_dicts = [chunk.model_dump() for chunk in self.chunks]
        write_json(Path(file_path), {"schema_version": 2, "index_revision": self.index_revision,
            "parser_versions": sorted({c.parser_version for c in self.chunks}),
            "embedding_identity": self.embedding_identity, "chunks": chunk_dicts})
        self.cleanup_pending = False

    def reconcile_library(self, data_dir, documents, *, save):
        """Remove only uncommitted library-owned paths, under the ingestion lock.

        A failed disk write hides orphans in memory and leaves cleanup_pending set.
        The caller must retry before ingestion; startup repeats the catalog-based
        scan, so even a process death before cleanup cannot resurrect the chunks.
        """
        root = (Path(data_dir) / "converted").resolve()
        owned_ids = {d.document_id for d in documents}
        valid = {str(Path(v.converted_path).resolve()) for d in documents for v in d.versions}
        valid.update(str(Path(d.converted_path).resolve()) for d in documents if d.converted_path)

        def orphan(path):
            resolved = Path(path).resolve()
            if str(resolved) in valid:
                return False
            try:
                parts = resolved.relative_to(root).parts
            except ValueError:
                return False
            # Only our reserved staging namespace, never search_cache or arbitrary
            # imported indexes. All committed history is protected above.
            return (len(parts) == 3 and (parts[0] in owned_ids or bool(re.fullmatch(r"doc_[0-9a-f]{32}", parts[0])))
                    and bool(re.fullmatch(r"[0-9a-f]{32}", parts[1])))

        with self.lock:
            paths = {c.file_path for c in self.chunks if orphan(c.file_path)}
            for path in paths:
                self.remove_by_filepath(path)
            if paths or self.cleanup_pending:
                self.cleanup_pending = True
                save()  # save serializes this very same locked revision atomically.
                self.cleanup_pending = False
            return paths

    @classmethod
    def load_json(cls, file_path: str) -> VectorStore:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("schema_version", 1) not in {1, 2}:
            raise ValueError("未知向量索引 schema；拒绝混用，请使用兼容版本或重建")
        revision = data.get("index_revision", 0) if isinstance(data, dict) else 0
        identity = data.get("embedding_identity") if isinstance(data, dict) else None
        data = data["chunks"] if isinstance(data, dict) else data

        chunks: list[Chunk] = []
        for chunk_dict in data:
            chunk = Chunk(**chunk_dict)
            chunks.append(chunk)

        vector_store = cls()
        vector_store.index_revision = revision
        if any(c.schema_version == 1 for c in chunks):
            logging.getLogger(__name__).warning("读取旧文本索引：未知页码保持为空；重新导入 PDF 可建立结构化新版本")
        vector_store.embedding_identity = identity
        vector_store.add_chunks(chunks=chunks)

        return vector_store

    @locked
    def remove_by_filepath(self, file_path: str) -> int:
        if file_path not in self._file_index:
            return 0

        self.index_revision += 1
        count = len(self._file_index[file_path])
        self.chunks = [chunk for chunk in self.chunks if chunk.file_path != file_path]
        del self._file_index[file_path]
        
        return count
