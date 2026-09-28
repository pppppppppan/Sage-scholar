from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import BadRequestError

from sage_scholar.rag.embedding import Embedding
from sage_scholar.rag.document import Chunk
from sage_scholar.rag.vector_store import VectorStore


def test_local_embedding_needs_no_key_and_uses_query_passage_prefix(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    calls = []
    class Encoder:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.array([[1., 0.] for _ in texts])
    embedding = Embedding()
    embedding.local._encoder = Encoder()
    assert embedding.client is None
    assert embedding.invoke("研究问题") == ([1., 0.], 0)
    assert embedding.invoke(["paper text"]) == ([[1., 0.]], 0)
    assert calls[0][0] == ["query: 研究问题"]
    assert calls[1][0] == ["passage: paper text"]
    assert calls[1][1]["normalize_embeddings"] is True
    assert embedding.invoke([]) == ([], 0)


def test_cloud_error_becomes_readable_upload_error():
    embedding = Embedding(provider="api", model="missing", api_key="test", base_url="https://example.test/v1")
    def fail(**kwargs):
        raise BadRequestError("provider detail", response=httpx.Response(400, request=httpx.Request("POST", "https://example.test")), body={"error": {"code": "1211"}})
    embedding.client = SimpleNamespace(embeddings=SimpleNamespace(create=fail))
    with pytest.raises(RuntimeError, match="向量化失败.*missing"):
        embedding.invoke(["paper"])


def test_index_records_model_and_rejects_incompatible_model(tmp_path, monkeypatch):
    import sage_scholar.rag.pipeline as module
    store = VectorStore()
    store.embedding_identity = {"provider": "api", "model": "old"}
    store.add_chunks([Chunk(file_path="paper.md", chunk_idx=0, content="paper", tokens=["paper"], embedding=[1., 0.])])
    path = tmp_path / "vector_store.json"
    store.save_to_json(str(path))
    assert VectorStore.load_json(str(path)).embedding_identity == store.embedding_identity
    monkeypatch.setattr(module, "Reranker", lambda: None)
    monkeypatch.setattr(module, "Embedding", lambda: SimpleNamespace(identity={"provider": "local", "model": "new"}))
    with pytest.raises(RuntimeError, match="不兼容"):
        module.Pipeline(str(tmp_path))
    with pytest.raises(RuntimeError, match="维度不一致"):
        store.cosine_search([1.], 1)
