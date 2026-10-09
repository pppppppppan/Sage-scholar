import json
import pytest
from research_fakes import ScriptedLLM, contract
from sage_scholar.domain import EvidenceRef
from sage_scholar.infrastructure.evidence_store import JsonEvidenceStore
from sage_scholar.research.evidence_context import ResearchEvidenceContext
from sage_scholar.context.manager import ContextManager
from sage_scholar.memory import MemoryStore, MemoryInput, MemoryService, MemoryRecall
from types import SimpleNamespace


def test_memory_explicit_crud_and_context_protection(tmp_path):
    memories = MemoryStore(str(tmp_path))
    saved = memories.add(MemoryInput(content="不考虑微调；FNO 更好"))
    memories.update(saved.memory_id, MemoryInput(content="不考虑微调"))
    manager = ContextManager(str(tmp_path), max_tokens=6000, memories=MemoryService(memories, MemoryRecall(tmp_path), lambda: ScriptedLLM()))
    request = contract().execution.model_copy(update={"query": "这次也比较微调方案", "allow_web": False,
        "history": [{"role": "user", "content": "不要使用私有数据"}], "request_id": "r1"})
    original = "source " * 10000
    prepared = manager.prepare([
        {"role": "system", "content": "research"},
        {"role": "user", "content": json.dumps({"conflicts": ["A 与 B 相矛盾"], "unknowns": ["test split unknown"]})},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "fetch", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": original},
        {"role": "user", "content": "继续"},
    ], tools=[{"type": "function", "function": {"name": "fetch"}}], request=request, output_reserve=1000)
    text = json.dumps(prepared.messages, ensure_ascii=False)
    assert "不要使用私有数据" in text and "这次也比较微调" in text and "不是 Evidence" in text
    assert "相矛盾" in text and "test split unknown" in text and 'allow_web' in text
    tool = next(m for m in prepared.messages if m["role"] == "tool")
    artifact = json.loads(tool["content"])
    assert manager.artifacts.read(artifact["artifact_id"], "r1")["text"] == original[:4000]
    assert prepared.accounting["total_tokens"] <= 6000 and prepared.accounting["tools_tokens"] > 0
    memories.delete(saved.memory_id)
    assert memories.list() == []


def test_long_history_preserves_constraints_and_evidence_reread(tmp_path):
    manager = ContextManager(str(tmp_path), max_tokens=4000)
    request = contract().execution.model_copy(update={"query": "继续研究", "allow_web": False,
        "history": [{"role": "user", "content": "不要使用微调"}]})
    prepared = manager.prepare([
        {"role": "user", "content": "不要使用微调"},
        {"role": "assistant", "content": "很长的旧解释 " * 5000},
        {"role": "user", "content": "继续研究"},
    ], request=request, output_reserve=1000)
    assert prepared.accounting["before_compression"] > prepared.accounting["input_tokens"]
    assert "不要使用微调" in json.dumps(prepared.messages, ensure_ascii=False)
    item = EvidenceRef.create(source_id="https://example.org/source", source_version="v1", parser_version="test",
        original_text="original " * 2000, metadata={"kind": "source_content"})
    store = JsonEvidenceStore(str(tmp_path)); store.save(item)
    context = ResearchEvidenceContext(store, contract().source_policy)
    context.register(item)
    assert manager.reread_evidence(context, item.evidence_id, offset=10)["original_text"] == item.original_text[10:6010]
    records = manager.read_evidence(context, [item.evidence_id, item.evidence_id], budget_tokens=100)
    assert len(records) == 1 and records[0]["truncated"]
    with pytest.raises(ValueError): context.get("mem_fake")
    memories = MemoryStore(str(tmp_path))
    memories.add(MemoryInput(content="FNO is better"))
    manager.memories = MemoryService(memories, MemoryRecall(tmp_path), lambda: ScriptedLLM())
    assert "FNO is better" not in json.dumps(manager.prepare([{"role": "user", "content": "verify"}],
        request=request, include_memory=False, output_reserve=1000).messages)


def test_removed_audit_endpoint_and_memory_details(tmp_path):
    from fastapi.testclient import TestClient
    from sage_scholar.api.app import create_app
    async def close(): pass
    container = SimpleNamespace(run_manager=SimpleNamespace(),
        memory_store=MemoryStore(str(tmp_path)), close=close)
    with TestClient(create_app(container)) as client:
        record = client.post("/api/memories", json={"content": "不考虑微调"}).json()
        assert client.get("/api/memories").json()[0]["memory_id"] == record["memory_id"]
        assert client.put("/api/memories/" + record["memory_id"], json={"content": "可以比较微调"}).status_code == 200
        assert client.delete("/api/memories/" + record["memory_id"]).status_code == 200
        assert client.get("/api/memories").json() == []
        assert client.post("/api/audits", json={}).status_code == 404
