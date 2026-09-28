"""Local sentence-transformers embeddings; no remote inference or API credentials."""
import logging
import os
import threading

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class LocalEmbedding:
    def __init__(self, model: str):
        self.model = model
        self.device = os.getenv("EMBEDDING_DEVICE", "cpu")
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
        if self.batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE 必须大于 0")
        self.query_prefix = os.getenv("EMBEDDING_QUERY_PREFIX", "query: ")
        self.document_prefix = os.getenv("EMBEDDING_DOCUMENT_PREFIX", "passage: ")
        self._encoder = None
        self._lock = threading.Lock()

    @property
    def identity(self):
        return {"provider": "local", "model": self.model, "normalized": True,
                "query_prefix": self.query_prefix, "document_prefix": self.document_prefix}

    def invoke(self, texts: list[str], *, task: str) -> list[list[float]]:
        if not texts:
            return []
        prefix = self.query_prefix if task == "query" else self.document_prefix
        # Serialize first load and inference to avoid duplicate model allocation on Mac.
        with self._lock:
            if self._encoder is None:
                logger.info("加载本地向量模型 %s（首次使用会下载模型），device=%s", self.model, self.device)
                try:
                    from sentence_transformers import SentenceTransformer
                    self._encoder = SentenceTransformer(
                        self.model, device=self.device, trust_remote_code=False,
                        local_files_only=os.getenv("EMBEDDING_LOCAL_ONLY", "false").lower() == "true",
                    )
                except Exception as exc:
                    raise RuntimeError("本地向量模型加载失败。首次使用需要下载模型，请检查 Hugging Face 网络连接，或将 EMBEDDING_LOCAL_MODEL 设为已下载的模型目录。") from exc
            logger.info("本地向量化 %d 条文本，task=%s", len(texts), task)
            vectors = self._encoder.encode(
                [prefix + text for text in texts], batch_size=self.batch_size,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
            )
            return vectors.tolist()
