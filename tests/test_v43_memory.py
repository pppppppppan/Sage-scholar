"""Memory lifecycle and retrieval contracts without paid APIs or model downloads."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from research_fakes import ScriptedLLM, contract
from sage_scholar.memory import MemoryInput, MemoryStore, MemoryService, MemoryRecall
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
    return MemoryInput(content=content, **kwargs)


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


def test_extraction_automatically_saves_only_grounded_durable_content(tmp_path):
    content = "SAGE Scholar 不考虑多用户架构"
    store, service, _ = memory(tmp_path, {"candidates": [
        extraction(content),
        extraction("用户喜欢 Transformer", quote="我猜他喜欢 Transformer", message_index=1, memory_type="preference", scope="user"),
        extraction("模型编造的背景", quote="不存在的原话"),
        extraction("这轮只比较三篇论文", message_index=2, scope="user"),
    ]})
    result = service.extract([{"role": "user", "content": content},
        {"role": "assistant", "content": "我猜他喜欢 Transformer"},
        {"role": "user", "content": "这轮只比较三篇论文"}], workspace_id="sage", session_id="s1")
    assert len(result) == len(store.list(all_scopes=True)) == 1
    assert result[0].source.quote == content and result[0].workspace_id == "sage"
    assert not hasattr(service, "approve")


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


def test_background_extraction_cannot_override_new_manual_edit(tmp_path):
    from sage_scholar.domain.documents import utc_now
    store, service, llm = memory(tmp_path)
    old = store.add(value("不考虑微调"))
    started = utc_now()
    store.update(old.memory_id, value("手动修改：只比较 LoRA"))
    text = "以后考虑微调"
    llm.responses = iter([{"candidates": [extraction(text)]},
                          {"action": "update", "targets": [old.memory_id], "reason": "later decision"}])
    assert service.extract([{"role": "user", "content": text}], started_at=started) == []
    assert store.list()[0].content == "手动修改：只比较 LoRA"


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
    assert json.loads((tmp_path / "memory.json").read_text())["schema"] == 3


def test_memory_http_automatic_writes_and_scope(tmp_path):
    from sage_scholar.api.app import create_app
    text = "以后本项目考虑微调"
    store, service, _ = memory(tmp_path, {"candidates": [extraction(text)]})
    from sage_scholar.runtime.store import RunStore
    from sage_scholar.memory.jobs import MemoryJobStore
    from sage_scholar.memory.background import MemoryBackground
    db = RunStore(tmp_path)
    jobs = MemoryJobStore(db)
    background = MemoryBackground(service, None, jobs)
    async def close():
        await background.close()
        db.close()
    container = SimpleNamespace(memory_store=store, memory_service=service, run_manager=SimpleNamespace(),
        memory_background=background, memory_jobs=jobs, close=close)
    with TestClient(create_app(container)) as client:
        response = client.post("/api/memories/extract", json={"workspace_id": "sage", "session_id": "s1",
            "messages": [{"role": "user", "content": text}]})
        assert response.status_code == 202
        client.portal.call(background.drain)
        assert client.get("/api/memories/jobs/" + response.json()["job_id"]).json()["status"] == "succeeded"
        saved = client.get("/api/memories?workspace_id=sage").json()[0]
        assert client.get("/api/memories/candidates").status_code in {404, 405}
        assert client.post("/api/memories/candidates/old/approve", json={}).status_code == 404
        assert saved["status"] == "active" and saved["source"]["quote"] == text
        assert client.get("/api/memories").json() == []
        assert len(client.get("/api/memories?workspace_id=sage").json()) == 1
        assert client.delete(f"/api/memories/{saved['memory_id']}").status_code == 200
        assert client.get("/api/memories?workspace_id=sage").json() == []
        # Explicit CRUD no longer depends on a confirmation flag either.
        assert client.post("/api/memories", json={"content": "简洁回答", "scope": "user"}).status_code == 200


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
    candidate = service.extract([{"role": "user", "content": "比较方法"}, {"role": "assistant", "content": text, "evidence_ids": ["ev_test"]}], research=True)[0]
    assert candidate.status == "active" and not candidate.supersedes
    assert next(r for r in store.list() if r.memory_id == old.memory_id).status == "active"
