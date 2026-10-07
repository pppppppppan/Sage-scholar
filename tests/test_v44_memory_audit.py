"""V4.4 boundaries: automatic writes, task recall, background isolation and local identities."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from research_fakes import contract
from test_v43_memory import memory, value, extraction, Embedding
from test_query_service import make_service
from sage_scholar.application.query_service import QueryCommand
from sage_scholar.audit.sources import AuditSources
from sage_scholar.domain import EvidenceRef
from sage_scholar.domain.documents import DocumentRecord, DocumentVersion
from sage_scholar.domain.execution import RunOutcome
from sage_scholar.infrastructure.evidence_store import JsonEvidenceStore
from sage_scholar.memory.background import MemoryBackground
from sage_scholar.memory.jobs import MemoryJobStore
from sage_scholar.runtime.store import RunStore
from sage_scholar.context.manager import ContextManager, CURRENT_REQUEST
from sage_scholar.routing import Workflow, SourceScope
from sage_scholar.runtime import RunManager


def test_automatic_consolidation_duplicate_and_later_conflict(tmp_path):
    old_text, new_text = "这个项目不考虑微调", "以后这个项目也考虑 LoRA 参数高效微调"
    store, service, llm = memory(tmp_path, {"candidates": [extraction(old_text)]})
    old = service.extract([{"role": "user", "content": old_text}])[0]
    llm.responses = iter([{"candidates": [extraction(old_text)]}])
    assert service.extract([{"role": "user", "content": old_text}])[0].memory_id == old.memory_id
    llm.responses = iter([{"candidates": [extraction(new_text)]},
        {"action": "update", "targets": [old.memory_id], "reason": "later explicit project decision"}])
    new = service.extract([{"role": "user", "content": new_text}])[0]
    assert len(store.list()) == 1 and new.supersedes == [old.memory_id]
    assert next(r for r in store.list(include_inactive=True) if r.memory_id == old.memory_id).status == "superseded"


def test_extracted_paraphrase_cannot_invent_more_than_source_quote(tmp_path):
    quote = "我更喜欢简洁的回答"
    store, service, _ = memory(tmp_path, {"candidates": [extraction("我讨厌英文，并喜欢简洁的回答", quote=quote, scope="user")]})
    service.extract([{"role": "user", "content": quote}])
    assert store.list()[0].content == quote


def test_task_aware_recall_cache_and_revision_invalidation(tmp_path):
    store, service, _ = memory(tmp_path)
    a = store.add(value("微调方向只做 LoRA"))
    b = store.add(value("工作区架构使用数据库"))
    manager = ContextManager(tmp_path, memories=service, max_tokens=12000)
    request = contract().execution.model_copy(update={"query": "比较主要技术路线"})
    Embedding.calls = []
    def prepare(objective):
        messages = [{"role": "user", "content": json.dumps({"objective": objective,
            "contract": {"objective": request.query}})}]
        result = manager.prepare(messages, request=request, purpose="researcher:react", output_reserve=1000)
        return json.dumps([m for m in result.messages if "以下是召回" in m.get("content", "")], ensure_ascii=False)
    assert a.memory_id in prepare("微调") and b.memory_id not in prepare("微调")
    assert b.memory_id in prepare("架构数据库") and a.memory_id not in prepare("架构数据库")
    assert len([x for x in Embedding.calls if x[1] == "query"]) == 2
    service.delete(a.memory_id)
    assert a.memory_id not in prepare("微调")
    assert len([x for x in Embedding.calls if x[1] == "query"]) == 3
    # A store write bypassing MemoryService must invalidate recall through the durable revision too.
    newer = store.add(value("使用低秩适配器微调"))
    assert newer.memory_id in prepare("微调")


@pytest.mark.parametrize("workflow", [Workflow.DIRECT, Workflow.QA])
def test_inline_completion_schedules_extraction_without_waiting(tmp_path, workflow):
    async def scenario():
        text = "这个项目以后都不考虑多用户架构"
        store, memories, _ = memory(tmp_path, {"candidates": [extraction(text)]})
        db = RunStore(tmp_path)
        background = MemoryBackground(memories, None, MemoryJobStore(db))
        service, manager, _ = make_service()
        service._memory_background = background
        result = await service.execute(QueryCommand(query=text, workflow=workflow,
            document_ids=["paper-a"] if workflow == Workflow.QA else [], workspace_id="sage", session_id="s1"))
        assert result.answer
        await asyncio.wait_for(background.drain(), timeout=3)
        assert store.list(workspace_id="sage")[0].source.reference == result.request_id
        await background.close()
        await manager.shutdown()
        db.close()
    asyncio.run(scenario())


def test_background_model_failure_cannot_fail_response(tmp_path):
    async def scenario():
        _, memories, _ = memory(tmp_path, RuntimeError("provider down"))
        db = RunStore(tmp_path)
        background = MemoryBackground(memories, None, MemoryJobStore(db))
        service, manager, _ = make_service()
        service._memory_background = background
        background.jobs.MAX_ATTEMPTS = 1
        result = await service.execute(QueryCommand(query="介绍一下方法", workflow=Workflow.DIRECT))
        assert result.answer
        await asyncio.wait_for(background.drain(), timeout=3)
        assert not memories.list()
        await background.close()
        await manager.shutdown()
        db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["completed", "partial"])
def test_research_automatic_memory_has_real_evidence_ids(tmp_path, status):
    async def scenario():
        text = "A 在 Dataset X 上优于 B，跨边界条件泛化仍待研究"
        store = JsonEvidenceStore(str(tmp_path))
        evidence = EvidenceRef.create(source_id="https://example.org/paper", original_text=text,
            source_version="v1", parser_version="test", metadata={"kind": "source_content"})
        store.save(evidence)
        retrieval = SimpleNamespace(evidence_store=store, documents=None)
        memories, service, _ = memory(tmp_path, {"candidates": [extraction(text, message_index=1, memory_type="research_note")]})
        db = RunStore(tmp_path)
        background = MemoryBackground(service, retrieval, MemoryJobStore(db))
        manager = RunManager(store=db, on_completed=background.completed)
        class Runner:
            async def run(self, request, emit, cancelled):
                await emit("research_task_completed", {"note": {"task_id": "T1", "findings": [
                    {"text": text, "evidence_ids": [evidence.evidence_id, "ev_missing"]}]}})
                return RunOutcome(status=status, result={"report": "report"})
        run = await manager.start(contract().execution, Runner())
        _ = [event async for event in manager.events(run.run_id)]
        await asyncio.wait_for(background.drain(), timeout=3)
        saved = memories.list()[0]
        assert saved.source.reference == run.run_id
        assert saved.source.evidence_ids == [evidence.evidence_id] and saved.memory_type == "research_note"
        assert manager.get(run.run_id).status == status
        assert CURRENT_REQUEST.get() is None
        await background.close()
        await manager.shutdown()
        db.close()
    asyncio.run(scenario())


def local_sources(tmp_path, metadata=None, duplicate=False):
    store = JsonEvidenceStore(str(tmp_path))
    path = tmp_path / "paper.md"
    path.write_text("A beats B on Dataset X only.")
    docs, sections = [], []
    for i in range(2 if duplicate else 1):
        doc = DocumentRecord(document_id=f"doc_{i}", title="Attention Is All You Need", source_identifier=str(path),
            converted_path=str(path), document_version="v1", active_version="v1", parser_version="test",
            metadata=metadata or {}, versions=[DocumentVersion(document_version="v1", version_number=1,
                parser_version="test", converted_path=str(path))])
        docs.append(doc)
        item = EvidenceRef.create(source_id=str(path), document_id=doc.document_id, original_text=path.read_text(),
            source_version="v1", parser_version="test", metadata={"converted_path": str(path)})
        store.save(item)
        sections.append(item)
    retrieval = SimpleNamespace(evidence_store=store, documents=SimpleNamespace(get=lambda id: next(d for d in docs if d.document_id == id)),
        permitted_documents=lambda policy: [d for d in docs if policy.permits_document(d.document_id)],
        read_documents_sync=lambda policy: [e for e in sections if policy.permits_document(e.document_id)])
    request = contract().execution.model_copy(update={"allow_web": False, "source_scope": SourceScope.WORKSPACE})
    class NoNetwork:
        def search(self, reference):
            pytest.fail("local identity must not start paper-search")
        def __call__(self, source):
            pytest.fail("local identity must not fetch")
    return AuditSources(retrieval, request.source_policy, NoNetwork()), request, retrieval


@pytest.mark.parametrize("metadata", [{}, {"authors": ["Ashish Vaswani"], "year": 2017}])
def test_local_bibliography_normalized_title_and_optional_metadata(tmp_path, metadata):
    sources, _, _ = local_sources(tmp_path, metadata)
    result = sources.resolve("1", "Vaswani et al. Attention Is All You Need. NeurIPS 2017.")
    assert result.status == "source_found" and result.evidence_ids
    assert sources.context.get(result.evidence_ids[0]).original_text.startswith("A beats B")


def test_ambiguous_local_title_is_not_arbitrarily_selected(tmp_path):
    sources, _, _ = local_sources(tmp_path, duplicate=True)
    result = sources.resolve("1", "Vaswani et al. Attention Is All You Need. NeurIPS 2017.")
    assert result.status == "source_not_found" and result.metadata["resolution"] == "ambiguous"
    assert result.evidence_ids == [] and "伪造" in result.reason


def test_local_identifier_priority_and_scope(tmp_path):
    sources, request, retrieval = local_sources(tmp_path, {"doi": "10.1000/test", "arxiv_id": "1706.03762"})
    for ref in ["DOI:10.1000/test", "arxiv:1706.03762", "doc_0"]:
        assert sources.resolve("1", ref).status == "source_found"
    assert sources.resolve("1", "Attention Is All You Need DOI:10.1000/other").status == "source_not_found"
    restricted = request.model_copy(update={"source_scope": SourceScope.SELECTED_DOCUMENTS, "document_ids": ["doc_other"]})
    assert AuditSources(retrieval, restricted.source_policy).resolve("1", "Attention Is All You Need").status == "source_not_found"


def test_older_run_cannot_override_newer_automatic_decision(tmp_path):
    from sage_scholar.domain.documents import utc_now
    store, service, llm = memory(tmp_path)
    started_at = utc_now()
    newer = store.add(value("以后考虑 LoRA 微调"))
    text = "不考虑微调"
    llm.responses = iter([{"candidates": [extraction(text)]},
        {"action": "update", "targets": [newer.memory_id], "reason": "conflict"}])
    assert service.extract([{"role": "user", "content": text}], started_at=started_at) == []
    assert store.list()[0].memory_id == newer.memory_id
