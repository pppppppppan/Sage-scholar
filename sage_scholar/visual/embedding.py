"""Aligned query/image encoder. No text-embedding fallback for visual vectors."""
import threading

from sage_scholar.runtime.control import check_control


class VisualEmbedding:
    def __init__(self, config):
        self.config = config
        self._model = None
        self._lock = threading.Lock()

    @property
    def identity(self):
        return {"model": self.config.embedding_model, "revision": self.config.embedding_revision,
                "normalized": True, "input": "image-only-v1", "implementation": "sentence-transformers"}

    def encode(self, inputs):
        import numpy as np
        if not inputs:
            return []
        with self._lock:
            check_control()
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.embedding_model,
                    revision=self.config.embedding_revision, device=self.config.device,
                    local_files_only=self.config.local_only, trust_remote_code=False)
            result = []
            # One image at a time bounds activation memory on laptops.
            for value in inputs:
                check_control()
                vector = np.asarray(self._model.encode([value], batch_size=1,
                    normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0], dtype=float)
                if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                    raise ValueError("视觉 Embedding 返回无效向量")
                result.append((vector / np.linalg.norm(vector)).tolist())
            return result
