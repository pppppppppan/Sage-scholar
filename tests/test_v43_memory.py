"""Memory lifecycle and retrieval contracts without paid APIs or model downloads."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from research_fakes import ScriptedLLM, contract
from sage_scholar.memory import MemoryInput, MemoryStore, MemoryService, MemoryRecall
from sage_scholar.memory.store import MemoryConflict
from sage_scholar.context.manager import ContextManager


class Embedding:
    identity = {"model": "test-memory", "provider": "local"}
    calls = []
    def invoke(self, texts, task=None):
        self.calls.append((texts, task))
        def vector(text):
            if any(term in text.lower() for term in ("微调", "fine-tuning", "adapter", "低秩", "lora")):
                return [1.0, 0.0, 0.0]
            if any(term in text for term in ("架构", "数据库")):
                return [0.0, 1.0, 0.0]
            return [0.0, 0.0, 1.0]
        return (vector(texts) if isinstance(texts, str) else [vector(t) for t in texts]), 0


def value(content, **kwargs):
    return MemoryInput(content=content, confirmed=True, **kwargs)


def memory(tmp_path, *responses):
    store = MemoryStore(tmp_path)
    llm = ScriptedLLM(*responses)
    service = MemoryService(store, MemoryRecall(tmp_path, Embedding), lambda: llm)
    return store, service, llm


def extraction(content, **changes):
    return {"content": content, "quote": content, "message_index": 0, "scope": "workspace",
            "memory_type": "decision", "confidence": .95, "importance": .8, "durable": True,
            "rationale": "明确长期决策", **changes}


def test_large_memory_store_only_injects_relevant_active_subset(tmp_path):
    store, service, _ = memory(tmp_path)
    for i in range(80):
        store.add(value(f"课题其他主题 {i}"))
    wanted = store.add(value("这个课题不考虑 fine-tuning", memory_type="decision"))
    foreign = store.add(value("这个课题必须微调", workspace_id="another"))
    store.add(value("这轮使用 LoRA", scope="session", session_id="other-session"))
    manager = ContextManager(tmp_path, max_tokens=9000, memories=service)
    request = contract().execution.model_copy(update={"query": "这次比较低秩适配器", "allow_web": False})
    result = manager.prepare([{"role": "user", "content": request.query}], request=request, output_reserve=1000)
    text = json.dumps(result.messages, ensure_ascii=False)
    assert wanted.memory_id in text and foreign.memory_id not in text
    assert "课题其他主题" not in text and "这轮使用 LoRA" not in text
    assert "当前指令优先" in text and "不是 Evidence" in text and request.query in text
    assert result.accounting["memory_tokens"] < 1600
    # Scientific verifier does not get memory at all.
    verified = manager.prepare([{"role": "user", "content": "verify"}], request=request,
                               include_memory=False, output_reserve=1000)
    assert wanted.memory_id not in json.dumps(verified.messages)


def test_scope_filter_precedes_embedding_and_user_scope_is_shared(tmp_path):
    store, service, _ = memory(tmp_path)
    user = store.add(value("我喜欢简洁回答", scope="user"))
    a = store.add(value("工作区架构 A", workspace_id="a"))
    store.add(value("工作区架构 B", workspace_id="b"))
    session = store.add(value("架构本轮结论", scope="session", workspace_id="a", session_id="s1"))
    service.index.embedding_factory = lambda: Embedding()
    Embedding.calls = []
    recalled = service.recall("架构", workspace_id="a", session_id="s1")
    assert {r.memory_id for r in recalled} == {user.memory_id, a.memory_id, session.memory_id}
    assert "工作区架构 B" not in json.dumps(Embedding.calls, ensure_ascii=False)
    assert session.memory_id not in {r.memory_id for r in service.recall("架构", workspace_id="a", session_id="s2")}


def test_extraction_only_creates_grounded_reviewable_candidates(tmp_path):
    content = "SAGE Scholar 不考虑多用户架构"
    store, service, _ = memory(tmp_path, {"candidates": [
        extraction(content),
        extraction("用户喜欢 Transformer", quote="我猜他喜欢 Transformer", message_index=1, memory_type="preference", scope="user"),
        extraction("模型编造的背景", quote="不存在的原话"),
    ]})
    result = service.extract([{"role": "user", "content": content},
                              {"role": "assistant", "content": "我猜他喜欢 Transformer"}], workspace_id="sage", session_id="s1")
    assert len(result) == 1 and store.list(all_scopes=True) == []
    with pytest.raises(ValueError):
        service.approve(result[0].candidate_id)
    approved = service.approve(result[0].candidate_id, confirmed=True)
    assert approved.scope == "workspace" and approved.workspace_id == "sage"
    assert approved.source.quote == content
    assert service.approve(result[0].candidate_id, confirmed=True).memory_id == approved.memory_id


def test_temporary_user_content_never_promotes_to_global(tmp_path):
    text = "这轮只比较这三篇论文"
    _, service, _ = memory(tmp_path, {"candidates": [extraction(text, scope="user")]})
    candidates = service.extract([{"role": "user", "content": text}], workspace_id="a", session_id="s1")
    assert candidates[0].scope == "session"
    saved = service.approve(candidates[0].candidate_id, confirmed=True)
    assert saved.session_id == "s1" and not service.recall("论文", workspace_id="a", session_id="s2")
    with pytest.raises(ValueError):
        service.add(value(text, scope="user"))


@pytest.mark.parametrize("action", ["ignore", "merge", "update"])
def test_semantic_consolidation_and_temporal_links(tmp_path, action):
    store, service, llm = memory(tmp_path)
    old = store.add(value("这个课题不考虑微调", memory_type="decision"))
    new_text = "以后这个课题也考虑参数高效微调" if action == "update" else "当前项目暂时不考虑 fine-tuning"
    llm.responses = iter([{"action": action, "targets": [old.memory_id], "reason": "用户明确长期决定"}])
    new = service.add(value(new_text, memory_type="decision"))
    assert len(store.list()) == 1
    if action == "ignore":
        assert new.memory_id == old.memory_id
    else:
        history = store.list(include_inactive=True)
        previous = next(r for r in history if r.memory_id == old.memory_id)
        assert previous.status == "superseded" and previous.superseded_by == new.memory_id
        assert old.memory_id in new.supersedes and new.version == 2
        assert [r.memory_id for r in service.recall("微调")] == [new.memory_id]


def test_stale_candidate_cannot_override_manual_edit_or_deletion(tmp_path):
    store, service, llm = memory(tmp_path)
    old = store.add(value("不考虑微调"))
    text = "以后考虑微调"
    llm.responses = iter([{"candidates": [extraction(text)]},
                          {"action": "update", "targets": [old.memory_id], "reason": "later decision"}])
    candidate = service.extract([{"role": "user", "content": text}])[0]
    store.update(old.memory_id, value("手动修改：只比较 LoRA"))
    with pytest.raises(MemoryConflict):
        service.approve(candidate.candidate_id, confirmed=True)
    service.delete(old.memory_id)
    with pytest.raises(MemoryConflict):
        service.approve(candidate.candidate_id, confirmed=True)
    assert not service.recall("微调")


def test_manual_update_preserves_versions_and_resolves_other_active_conflicts(tmp_path):
    store, service, llm = memory(tmp_path)
    old = store.add(value("不考虑微调"))
    other = store.add(value("研究配置笔记", memory_type="note"))
    llm.responses = iter([{"action": "update", "targets": [old.memory_id], "reason": "新决定替代旧限制"}])
    edited = service.update(other.memory_id, value("以后也考虑微调"))
    assert edited.memory_id == other.memory_id and edited.version == 2
    assert len(store.list()) == 1 and len(store.list(include_inactive=True)) == 3
    assert set(edited.supersedes) == {old.memory_id, other.memory_id + "_v1"}


def test_deleted_memory_not_reintroduced_by_extraction_or_cached_vectors(tmp_path):
    store, service, llm = memory(tmp_path)
    old = store.add(value("不考虑微调"))
    assert service.recall("微调")
    service.delete(old.memory_id)
    llm.responses = iter([{"candidates": [extraction("不考虑微调")]}])
    assert service.extract([{"role": "user", "content": "不考虑微调"}]) == []
    assert not service.recall("微调")
    assert not MemoryService(MemoryStore(tmp_path), MemoryRecall(tmp_path, Embedding), lambda: llm).recall("微调")


def test_restart_and_legacy_migration_preserve_scope_source_history(tmp_path):
    (tmp_path / "memory.json").write_text(json.dumps([{"content": "不考虑微调", "type": "preference", "scope": "workspace",
        "confirmed": True, "memory_id": "mem_legacy", "created_at": "2026-09-01T00:00:00Z", "updated_at": "2026-09-01T00:00:00Z"}]))
    store = MemoryStore(tmp_path)
    assert store.list()[0].source.kind == "legacy"
    assert store.list()[0].workspace_id == "default" and not store.list(workspace_id="another")
    store.update("mem_legacy", value("以后也考虑微调", memory_type="decision"))
    reloaded = MemoryStore(tmp_path)
    assert len(reloaded.list(include_inactive=True)) == 2 and reloaded.list()[0].version == 2
    assert json.loads((tmp_path / "memory.json").read_text())["schema"] == 2


def test_memory_http_candidate_lifecycle_and_scope(tmp_path):
    from sage_scholar.api.app import create_app
    text = "以后本项目考虑微调"
    store, service, _ = memory(tmp_path, {"candidates": [extraction(text)]})
    async def close():
        pass
    container = SimpleNamespace(memory_store=store, memory_service=service, run_manager=SimpleNamespace(), close=close)
    with TestClient(create_app(container)) as client:
        response = client.post("/api/memories/extract", json={"workspace_id": "sage", "session_id": "s1",
            "messages": [{"role": "user", "content": text}]})
        assert response.status_code == 200
        identity = response.json()[0]["candidate_id"]
        assert client.get("/api/memories/candidates?workspace_id=other").json() == []
        assert client.post(f"/api/memories/candidates/{identity}/approve", json={"confirmed": False}).status_code == 400
        saved = client.post(f"/api/memories/candidates/{identity}/approve", json={"confirmed": True}).json()
        assert saved["status"] == "active" and saved["source"]["quote"] == text
        assert client.get("/api/memories").json() == []
        assert len(client.get("/api/memories?workspace_id=sage").json()) == 1
        assert client.delete(f"/api/memories/{saved['memory_id']}").status_code == 200
        assert client.get("/api/memories?workspace_id=sage").json() == []


def test_two_pending_conflicting_candidates_cannot_both_become_active(tmp_path):
    store, service, llm = memory(tmp_path, {"candidates": [extraction("不考虑微调"), extraction("以后考虑微调", message_index=1)]})
    candidates = service.extract([{"role": "user", "content": "不考虑微调"}, {"role": "user", "content": "以后考虑微调"}])
    saved = service.approve(candidates[0].candidate_id, confirmed=True)
    with pytest.raises(MemoryConflict):
        service.approve(candidates[1].candidate_id, confirmed=True)
    assert len(store.list()) == 1
    llm.responses = iter([{"candidates": [extraction("以后考虑微调", message_index=1)]},
                          {"action": "update", "targets": [saved.memory_id], "reason": "后续长期变更"}])
    replanned = service.extract([{"role": "user", "content": "不考虑微调"}, {"role": "user", "content": "以后考虑微调"}])[0]
    assert replanned.candidate_id == candidates[1].candidate_id and replanned.revision == 2
    with pytest.raises(MemoryConflict):
        service.approve(replanned.candidate_id, confirmed=True, revision=1)
    updated = service.approve(replanned.candidate_id, confirmed=True, revision=2)
    assert store.list()[0].memory_id == updated.memory_id and saved.memory_id in updated.supersedes


def test_embedding_model_change_rebuilds_projection_after_restart(tmp_path):
    store, service, _ = memory(tmp_path)
    saved = store.add(value("不考虑微调"))
    assert service.recall("微调")
    class NewEmbedding(Embedding):
        identity = {"model": "different", "provider": "local"}
        calls = []
    restored = MemoryService(MemoryStore(tmp_path), MemoryRecall(tmp_path, NewEmbedding), lambda: ScriptedLLM())
    assert restored.recall("低秩适配器")[0].memory_id == saved.memory_id
    assert any(task == "passage" for _, task in NewEmbedding.calls)
    assert json.loads((tmp_path / "memory_vectors.json").read_text())["identity"]["model"] == "different"


def test_explicit_memory_scope_does_not_spread_project_preferences_globally(tmp_path):
    _, service, _ = memory(tmp_path)
    personal = service.remember("我喜欢简洁回答", workspace_id="a")
    project = service.remember("这个项目我喜欢简洁回答", workspace_id="a")
    temporary = service.remember("这轮只比较三篇论文", workspace_id="a", session_id="s1")
    assert personal.scope == "user" and personal.workspace_id is None
    assert project.scope == "workspace" and project.workspace_id == "a"
    assert temporary.scope == "session" and temporary.session_id == "s1"


def test_research_lead_cannot_supersede_a_user_project_decision(tmp_path):
    store, service, llm = memory(tmp_path)
    old = store.add(value("项目不考虑微调", memory_type="decision"))
    text = "报告建议使用微调"
    llm.responses = iter([{"candidates": [extraction(text, message_index=1, memory_type="research_note")]}])
    candidate = service.extract([{"role": "user", "content": "比较方法"}, {"role": "assistant", "content": text}], research=True)[0]
    assert candidate.proposal.action == "new" and not candidate.proposal.targets
    service.approve(candidate.candidate_id, confirmed=True)
    assert next(r for r in store.list() if r.memory_id == old.memory_id).status == "active"
