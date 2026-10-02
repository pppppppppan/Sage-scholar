"""Small rebuildable memory index using the existing embedding and cosine implementation."""
import hashlib
import json
import logging
import re
import threading
from pathlib import Path

import jieba
from sage_scholar.domain.documents import utc_now
from sage_scholar.infrastructure.json_store import write_json
from sage_scholar.rag.document import Chunk
from sage_scholar.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)
STOP = {"的", "了", "是", "我", "这", "这个", "项目", "请", "一下", "a", "the", "is", "of", "to"}


def normalized(text):
    text = re.sub(r"fine[ -]?tuning|finetune", "微调", text.lower())
    return re.sub(r"[\W_]+", "", text)


def terms(text):
    text = re.sub(r"fine[ -]?tuning|finetune", "微调", text.lower())
    return {t for t in jieba.lcut(text) if t.strip() and t not in STOP and re.search(r"\w", t)}


def lexical_rank(query, records):
    words = terms(query)
    scored = []
    for record in records:
        tokens = terms(record.content)
        overlap = words & tokens
        # Response-style preferences apply to answers, not to scientific truth.
        style = record.scope == "user" and record.memory_type == "preference" and bool(re.search(r"回答|回复|中文|简洁|concise|response", record.content, re.I))
        score = len(overlap) / max(1, min(len(words), len(tokens)))
        if score > 0 or style:
            scored.append((record, max(score, 0.25 if style else 0)))
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


class MemoryRecall:
    def __init__(self, data_dir, embedding_factory=None):
        self.path = Path(data_dir) / "memory_vectors.json"
        self.embedding_factory = embedding_factory
        self._embedding = None
        self.lock = threading.RLock()

    def rank(self, query, records, *, top_k=5, threshold=0.78):
        if not records:
            return []
        lexical = {r.memory_id: score for r, score in lexical_rank(query, records)}
        semantic = {}
        if self.embedding_factory:
            try:
                semantic = self._semantic(query, records)
            except Exception as exc:
                logger.warning("Memory embedding unavailable, using bounded lexical recall (%s)", type(exc).__name__)
        ranked = []
        for record in records:
            lex, sem = lexical.get(record.memory_id, 0), semantic.get(record.memory_id, -1)
            # Importance and recency reorder only relevant memories; they cannot admit unrelated ones.
            if lex <= 0 and sem < threshold:
                continue
            age_days = max(0, (utc_now() - record.updated_at).total_seconds() / 86400)
            relevance = max(lex, max(0, sem))
            score = .8 * relevance + .12 * record.importance + .08 / (1 + age_days / 90)
            ranked.append((record, score))
        return sorted(ranked, key=lambda p: (-p[1], p[0].memory_id))[:top_k]

    def _semantic(self, query, records):
        with self.lock:
            if self._embedding is None:
                self._embedding = self.embedding_factory()
            embedding = self._embedding
            identity = embedding.identity
            cache = json.loads(self.path.read_text()) if self.path.exists() else {}
            vectors = cache.get("vectors", {}) if cache.get("identity") == identity else {}
            keys = {r.memory_id: hashlib.sha256((r.memory_id + str(r.version) + r.content).encode()).hexdigest() for r in records}
            missing = [r for r in records if keys[r.memory_id] not in vectors]
            if missing:
                values, _ = embedding.invoke([r.content for r in missing], task="passage")
                if len(values) != len(missing):
                    raise ValueError("incomplete memory embeddings")
                vectors.update({keys[r.memory_id]: vector for r, vector in zip(missing, values)})
                write_json(self.path, {"identity": identity, "vectors": vectors})
            vector, _ = embedding.invoke(query, task="query")
            index = VectorStore()
            # Scope/status were filtered before materializing the index, including after restart.
            index.add_chunks([Chunk(file_path=r.memory_id, chunk_idx=0, content=r.content,
                tokens=[], embedding=vectors[keys[r.memory_id]]) for r in records])
            return {chunk.file_path: score for chunk, score in index.cosine_search(vector, top_k=len(records))}

    def invalidate(self):
        # Called after manual changes/deletion. Never allow a stale vector projection to resurrect text.
        with self.lock:
            self.path.unlink(missing_ok=True)
