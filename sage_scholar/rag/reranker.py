"""Lazy, shared cross-encoder with bounded score cache and observable RRF fallback."""
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
import hashlib
import logging
import os
import threading
import time
import numpy as np

from .document import Chunk

BASELINE = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
MULTILINGUAL = 'BAAI/bge-reranker-v2-m3'
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankerConfig:
    model: str = field(default_factory=lambda: os.getenv('RERANKER_MODEL', MULTILINGUAL))
    device: str = field(default_factory=lambda: os.getenv('RERANKER_DEVICE', 'cpu'))
    batch_size: int = field(default_factory=lambda: int(os.getenv('RERANKER_BATCH_SIZE', '8')))
    max_length: int = field(default_factory=lambda: int(os.getenv('RERANKER_MAX_LENGTH', '512')))
    revision: str | None = field(default_factory=lambda: os.getenv('RERANKER_REVISION') or None)
    local_only: bool = field(default_factory=lambda: os.getenv('RERANKER_LOCAL_ONLY', 'false').lower() == 'true')
    fallback: str = field(default_factory=lambda: os.getenv('RERANKER_FALLBACK', 'rrf'))
    cache_size: int = field(default_factory=lambda: int(os.getenv('RERANKER_CACHE_SIZE', '2048')))
    retry_seconds: float = 60

    def __post_init__(self):
        if self.batch_size < 1 or self.max_length < 1 or self.cache_size < 0 or self.retry_seconds < 0:
            raise ValueError('Invalid Reranker limits')
        if self.fallback not in {'rrf', 'raise'}:
            raise ValueError('RERANKER_FALLBACK must be rrf or raise')


class RerankerUnavailable(RuntimeError):
    pass


class _Shared:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = None
        self.cache = OrderedDict()
        self.failure = None
        self.retry_at = 0


class Reranker:
    _pool = OrderedDict()
    _pool_lock = threading.Lock()

    def __init__(self, config=None):
        self.config = config or RerankerConfig()
        with self._pool_lock:
            if self.config not in self._pool:
                self._pool[self.config] = _Shared()
            self._shared = self._pool[self.config]
            self._pool.move_to_end(self.config)
            while len(self._pool) > 2:
                self._pool.popitem(last=False)
        self.last_status = {'state': 'not_loaded', 'config': asdict(self.config)}

    def _load(self):
        shared = self._shared
        if shared.model is None:
            if shared.failure and time.monotonic() < shared.retry_at:
                raise RerankerUnavailable(shared.failure)
            from sentence_transformers import CrossEncoder
            try:
                shared.model = CrossEncoder(self.config.model,
                    device=None if self.config.device == 'auto' else self.config.device,
                    max_length=self.config.max_length, revision=self.config.revision,
                    local_files_only=self.config.local_only, trust_remote_code=False)
                shared.failure = None
            except Exception as exc:
                shared.failure = f'{self.config.model} on {self.config.device}: {type(exc).__name__}: {exc}'
                shared.retry_at = time.monotonic() + self.config.retry_seconds
                raise RerankerUnavailable(shared.failure) from exc
        return shared.model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not chunks or top_k <= 0:
            return []
        started = time.perf_counter()
        shared = self._shared
        try:
            with shared.lock:
                model = self._load()
                keys = [hashlib.sha256((query+'\0'+c.content).encode()).hexdigest() for c in chunks]
                missing = dict((key, c.content) for key, c in zip(keys, chunks) if key not in shared.cache)
                scores = {}
                if missing:
                    values = np.asarray(model.predict([(query, text) for text in missing.values()],
                        batch_size=self.config.batch_size, show_progress_bar=False)).reshape(-1)
                    if len(values) != len(missing) or not np.isfinite(values).all():
                        raise RerankerUnavailable('Invalid reranker scores')
                    scores.update(zip(missing, map(float, values)))
                resolved = [scores[key] if key in scores else shared.cache[key] for key in keys]
                for key, score in zip(keys, resolved):
                    shared.cache[key] = score
                    shared.cache.move_to_end(key)
                while len(shared.cache) > self.config.cache_size:
                    shared.cache.popitem(last=False)
                self.last_status = {'state':'ok','config':asdict(self.config),
                    'scored_pairs':len(missing),'cache_hits':len(keys)-len(missing),
                    'elapsed_ms':(time.perf_counter()-started)*1000,
                    'resolved_revision':getattr(getattr(getattr(model, 'model', None), 'config', None), '_commit_hash', None)}
                return sorted(zip(chunks, resolved), key=lambda item:item[1], reverse=True)[:top_k]
        except Exception as exc:
            self.last_status = {'state':'fallback','config':asdict(self.config),'error':str(exc),
                                'elapsed_ms':(time.perf_counter()-started)*1000}
            if self.config.fallback == 'raise':
                raise RerankerUnavailable(str(exc)) from exc
            logger.warning('Reranker unavailable; preserving fused RRF order: %s', exc)
            # Pipeline supplies RRF-ranked candidates. No unrelated/cross-scope fallback.
            return [(chunk, 1.0/(i+1)) for i, chunk in enumerate(chunks[:top_k])]
