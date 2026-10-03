import json
import pytest
from research_fakes import ScriptedLLM, contract
from sage_scholar.audit.service import CitationAuditService
from sage_scholar.audit.sources import AuditSources
from sage_scholar.audit.models import AuditClaim
from sage_scholar.domain import EvidenceRef
from sage_scholar.infrastructure.evidence_store import JsonEvidenceStore
from sage_scholar.research.verification import CitationVerifier
from sage_scholar.context.manager import ContextManager
from sage_scholar.memory import MemoryStore, MemoryInput
from types import SimpleNamespace


@pytest.mark.parametrize("status", ["unsupported", "partially_supported", "supported"])
def test_audit_reuses_verifier_with_joint_evidence(tmp_path, status):
    store = JsonEvidenceStore(str(tmp_path))
    items = [EvidenceRef.create(source_id=f"https://example.org/{i}", original_text=text,
             source_version="v1", parser_version="test", metadata={"kind": "source_content"})
             for i, text in enumerate(["A beats B on Dataset X", "A beats B on Dataset Y"])]
    for item in items: store.save(item)
    retrieval = SimpleNamespace(evidence_store=store, documents=None)
    sources = AuditSources(retrieval, contract().source_policy)
    llm = ScriptedLLM({"status": status, "reason": "Only X and Y are evaluated; not all tasks"})
    service = CitationAuditService(retrieval, lambda: llm)
    result = service.verify_claim(1, AuditClaim(text="A always beats B", citations=[i.evidence_id for i in items]),
                                  sources, {}, CitationVerifier(llm, sources.context))
    assert result.support_status == status and result.source_identity_status == "source_found"
    assert len(json.loads(llm.calls[0]["messages"][-1]["content"])["evidence"]) == 2
    missing = service.verify_claim(2, AuditClaim(text="claim", citations=["ev_missing"]), sources, {}, CitationVerifier(llm, sources.context))
    assert missing.source_identity_status == "source_not_found" and missing.support_status == "uncertain"
    assert "fabricated" not in missing.model_dump_json()


def test_memory_explicit_crud_and_context_protection(tmp_path):
    memories = MemoryStore(str(tmp_path))
    saved = memories.add(MemoryInput(content="不考虑微调；FNO 更好", confirmed=True))
    memories.update(saved.memory_id, MemoryInput(content="不考虑微调", confirmed=True))
    manager = ContextManager(str(tmp_path), max_tokens=6000, memories=memories)
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
    sources = AuditSources(SimpleNamespace(evidence_store=store, documents=None), contract().source_policy)
    assert sources.resolve(item.evidence_id, item.evidence_id).status == "source_found"
    assert manager.reread_evidence(sources.context, item.evidence_id, offset=10)["original_text"] == item.original_text[10:6010]
    records = manager.read_evidence(sources.context, [item.evidence_id, item.evidence_id], budget_tokens=100)
    assert len(records) == 1 and records[0]["truncated"]
    with pytest.raises(ValueError): sources.context.get("mem_fake")
    memories = MemoryStore(str(tmp_path))
    memories.add(MemoryInput(content="FNO is better", confirmed=True))
    manager.memories = memories
    assert "FNO is better" not in json.dumps(manager.prepare([{"role": "user", "content": "verify"}],
        request=request, include_memory=False, output_reserve=1000).messages)


def test_audit_and_memory_http_paths_share_existing_evidence_and_replay(tmp_path):
    from fastapi.testclient import TestClient
    from sage_scholar.api.app import create_app
    from sage_scholar.runtime import RunManager
    from sage_scholar.runtime.store import RunStore
    from sage_scholar.runtime.checkpoints import Checkpoints
    from sage_scholar.routing import Workflow
    store = JsonEvidenceStore(str(tmp_path))
    item = EvidenceRef.create(source_id="https://example.org/paper", original_text="A beats B only on X",
        source_version="v1", parser_version="test", metadata={"kind": "source_content"})
    store.save(item)
    retrieval = SimpleNamespace(evidence_store=store, documents=None)
    llm = ScriptedLLM({"status": "partially_supported", "reason": "Only X was tested"})
    run_store, checkpoints = RunStore(str(tmp_path)), Checkpoints(str(tmp_path))
    manager = RunManager(store=run_store, checkpoints=checkpoints)
    service = CitationAuditService(retrieval, lambda: llm, checkpoints=checkpoints)
    manager.register_runners({Workflow.CITATION_AUDIT: service})
    async def close():
        await manager.shutdown()
        checkpoints.close(); run_store.close()
    container = SimpleNamespace(run_manager=manager, audit_service=service,
        memory_store=MemoryStore(str(tmp_path)), close=close)
    with TestClient(create_app(container)) as client:
        record = client.post("/api/memories", json={"content": "不考虑微调", "confirmed": True}).json()
        assert client.get("/api/memories").json()[0]["memory_id"] == record["memory_id"]
        assert client.put("/api/memories/" + record["memory_id"], json={"content": "可以比较微调", "confirmed": True}).json()["content"] == "可以比较微调"
        assert client.delete("/api/memories/" + record["memory_id"]).status_code == 200
        assert client.get("/api/memories").json() == []
        response = client.post("/api/audits", json={"source_scope": "public", "allow_web": True,
            "claims": [{"text": "A beats B everywhere", "citations": ["[1]"]}], "citations": {"1": item.evidence_id}})
        assert response.status_code == 202
        identity = response.json()["run"]["run_id"]
        events = client.get(f"/api/runs/{identity}/events").text
        assert "audit_claim_checked" in events and "run_finished" in events
        result = client.get(f"/api/runs/{identity}").json()["run"]
        assert result["status"] == "partial" and result["result"]["audit_items"][0]["evidence_ids"] == [item.evidence_id]
        assert client.get(f"/api/runs/{identity}/events", headers={"Last-Event-ID": str(result["event_cursor"])}).text == ""
        assert len(llm.calls) == 1  # Refresh/replay never runs verification again.
