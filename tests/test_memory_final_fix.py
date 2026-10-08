"""Memory correctness boundaries: manual edits, confirmed context, durable replay."""
import asyncio
import hashlib
import json

import pytest
from research_fakes import contract
from test_v43_memory import memory, value, extraction
from test_memory_finalization import durable_research
from sage_scholar.domain.documents import utc_now
from sage_scholar.memory import MemoryStore
from sage_scholar.memory.background import MemoryBackground
from sage_scholar.memory.conversation_source import conversation_messages
from sage_scholar.memory.jobs import MemoryJobStore
from sage_scholar.memory.models import MemoryCandidate, MemorySource, scope_fingerprint
from sage_scholar.memory.research_source import ResearchMemorySource
from sage_scholar.memory.store import MemoryConflict
from sage_scholar.runtime.store import RunStore

PROPOSAL = '建议 SAGE Scholar 保持单体架构，不引入 Redis/Celery，优先完善 Evidence 和 Deep Research。'
CONFIRM = '对，就按这个方案，以后都这样。'
CANONICAL = 'SAGE Scholar 采用单体架构，暂不引入 Redis/Celery，优先完善 Evidence 和 Deep Research。'


def verdict(relation='accepted'):
    return dict(supported=True, user_committed=True, durable=True, standalone=True,
                relation=relation, reason='原始上下文和用户明确长期决定支持全部内容')


def proposal(content=CANONICAL, quote=CONFIRM, relation='accepted'):
    return extraction(content, quote=quote, message_index=1, relation=relation,
                      context_quotes=[{'message_index': 0, 'quote': PROPOSAL}])


def messages(quote=CONFIRM):
    return [{'role': 'assistant', 'content': PROPOSAL, 'message_id': 'a1'},
            {'role': 'user', 'content': quote, 'message_id': 'u2'}]


@pytest.mark.parametrize('operation', ['update', 'delete'])
def test_unrelated_manual_change_does_not_suppress_new_candidate(tmp_path, operation):
    store, service, llm = memory(tmp_path)
    old = store.add(value('SAGE Scholar 使用 FastAPI', memory_type='decision'))
    started = utc_now()
    if operation == 'update':
        store.update(old.memory_id, value('SAGE Scholar 使用 FastAPI 单体架构', memory_type='decision'))
    else:
        store.delete(old.memory_id)
    text = 'FNO 在边界条件变化下存在泛化问题，仍需进一步核验。'
    llm.responses = iter([{'candidates': [extraction(text, memory_type='research_note')]},
        {'action': 'new', 'targets': [], 'reason': '科研泛化线索与框架决策无关'}])
    saved = service.extract([{'role': 'assistant', 'content': text, 'evidence_ids': ['ev_original']}],
                            research=True, started_at=started)
    assert len(saved) == 1 and saved[0].content == text
    assert saved[0].source.evidence_ids == ['ev_original']


def test_deleted_equivalent_content_cannot_be_restored(tmp_path):
    store, service, llm = memory(tmp_path)
    old = store.add(value('这个项目不考虑微调'))
    service.delete(old.memory_id)
    text = '当前项目暂时不考虑 fine-tuning'
    llm.responses = iter([{'candidates': [extraction(text)]},
        {'action': 'ignore', 'targets': [old.memory_id], 'reason': '等价 tombstone'}])
    assert service.extract([{'role': 'user', 'content': text}]) == []
    assert not store.list()


@pytest.mark.parametrize('operation', ['update', 'delete'])
def test_concurrent_manual_operation_invalidates_consolidation_target(tmp_path, operation):
    store, service, llm = memory(tmp_path)
    old = store.add(value('不考虑微调'))
    text = '以后也考虑 LoRA 微调'
    candidate = MemoryCandidate(content=text, source=MemorySource(quote=text), rationale='test')
    revision, items = store.snapshot()
    llm.responses = iter([{'action': 'update', 'targets': [old.memory_id], 'reason': 'later decision'}])
    service._plan(candidate, items)
    if operation == 'update':
        store.update(old.memory_id, value('人工决定只考虑 LoRA'))
    else:
        store.delete(old.memory_id)
    with pytest.raises(MemoryConflict):
        store.commit(candidate, expected_revision=revision)
    # Even a refreshed global fingerprint cannot bypass per-target version/status checks.
    current_revision, current = store.snapshot()
    candidate.scope_fingerprint = scope_fingerprint(current, candidate)
    if operation == 'update':
        with pytest.raises(MemoryConflict):
            store.commit(candidate, expected_revision=current_revision)
        assert store.list()[0].content == '人工决定只考虑 LoRA'
    else:
        assert store.commit(candidate, expected_revision=current_revision) is None
        assert store.list() == []


def test_confirmed_canonical_memory_preserves_message_provenance(tmp_path):
    store, service, llm = memory(tmp_path, {'candidates': [proposal()]}, verdict())
    saved = service.extract(messages(), reference='request-2')[0]
    assert saved.content == CANONICAL and saved.source.quote == CONFIRM
    assert saved.source.reference == 'request-2' and saved.source.relation == 'accepted'
    assert [(m.message_id, m.message_index, m.quote) for m in saved.source.messages] == [
        ('a1', 0, PROPOSAL), ('u2', 1, CONFIRM)]
    assert [c['tag'] for c in llm.calls] == ['memory:extract', 'memory:grounding']
    assert MemoryStore(tmp_path).list()[0].source == saved.source


@pytest.mark.parametrize('relation,confirmation,canonical', [
    ('rejected', '不采纳这个方案，以后本项目改用微服务。', 'SAGE Scholar 不采用单体架构，改用微服务。'),
    ('modified', '采用单体架构，但以后本项目允许使用 Redis，其他按原方案。',
     'SAGE Scholar 采用单体架构，允许使用 Redis，不引入 Celery，优先完善 Evidence 和 Deep Research。'),
])
def test_rejection_and_modification_only_save_final_user_decision(tmp_path, relation, confirmation, canonical):
    # An extractor may mistakenly keep the rejected original: independent grounding rejects it.
    wrong = proposal(quote=confirmation)
    correct = proposal(content=canonical, quote=confirmation, relation=relation)
    store, service, _ = memory(tmp_path, {'candidates': [wrong, correct]},
        verdict() | {'supported': False}, verdict(relation))
    saved = service.extract(messages(confirmation))
    assert [m.content for m in saved] == [canonical]
    assert saved[0].source.relation == relation


@pytest.mark.parametrize('change', [
    {'supported': False}, {'user_committed': False}, {'durable': False}, {'standalone': False},
])
def test_grounding_rejects_unsupported_or_unconfirmed_canonical_memory(tmp_path, change):
    store, service, _ = memory(tmp_path, {'candidates': [proposal()]}, verdict() | change)
    assert service.extract(messages()) == [] and not store.list()


def test_assistant_suggestion_and_bare_confirmation_without_context_are_not_decisions(tmp_path):
    store, service, llm = memory(tmp_path)
    # A plain assistant suggestion cannot authorize a decision, even if the model says durable.
    llm.responses = iter([{'candidates': [extraction(PROPOSAL)]}])
    assert service.extract([{'role': 'assistant', 'content': PROPOSAL}]) == []
    llm.responses = iter([{'candidates': [extraction(CONFIRM)]}])
    assert service.extract([{'role': 'user', 'content': CONFIRM}]) == []
    assert not store.list()


@pytest.mark.parametrize('context_quotes', [[], [{'message_index': 0, 'quote': 'SAGE Scholar 引入 Redis/Celery。'}],
                                          [{'message_index': 1, 'quote': CONFIRM}]])
def test_canonical_requires_exact_prior_context_anchors(tmp_path, context_quotes):
    store, service, llm = memory(tmp_path, {'candidates': [proposal() | {'context_quotes': context_quotes}]})
    assert service.extract(messages()) == []
    assert len(llm.calls) == 1


def test_restart_restores_recent_conversation_and_does_not_reextract_history(tmp_path, monkeypatch):
    request = contract().execution.model_copy(update={'query': CONFIRM, 'request_id': 'req2', 'history': [
        {'role': 'user', 'content': 'SAGE Scholar 应该怎么设计？', 'id': 'u1'},
        {'role': 'assistant', 'content': PROPOSAL, 'id': 'a1'}]})
    # The current user is index 2 in the durable snapshot; the old user is ineligible.
    candidate = proposal() | {'message_index': 2, 'context_quotes': [{'message_index': 1, 'quote': PROPOSAL}]}
    store, service, llm = memory(tmp_path, {'candidates': [extraction(request.history[0]['content']), candidate]}, verdict())
    db = RunStore(tmp_path)
    jobs = MemoryJobStore(db)
    background = MemoryBackground(service, None, jobs)
    monkeypatch.setattr(background, '_wake', lambda: None)  # Simulate exit after enqueue, before inference.
    job = background.conversation(request, '已采纳。')
    assert jobs.get(job['job_id'])['payload']['messages'][1]['content'] == PROPOSAL
    db.close()
    db = RunStore(tmp_path)
    jobs = MemoryJobStore(db)
    restored = MemoryBackground(service, None, jobs)
    async def run():
        await restored.recover()
        await restored.drain()
        assert jobs.get(job['job_id'])['status'] == 'succeeded'
        await restored.close()
    asyncio.run(run())
    assert [m.content for m in store.list()] == [CANONICAL]
    anchors = store.list()[0].source.messages
    assert [(m.message_id, m.history_index) for m in anchors] == [('a1', 1), ('req2:user', 2)]
    assert 'SAGE Scholar 应该怎么设计' in llm.calls[1]['messages'][-1]['content']
    db.close()


def test_research_source_reuses_recent_conversation_without_changing_evidence_provenance(tmp_path):
    db, retrieval, run, note, evidence = durable_research(tmp_path)
    from sage_scholar.runtime.models import RunEvent
    run.execution = run.execution.model_copy(update={'query': CONFIRM, 'history': [messages()[0]]})
    db.append(run, RunEvent(sequence=4, type='run_finished', data={}))
    _, source, _ = ResearchMemorySource(db, retrieval).load(run.run_id)
    assert source[0]['content'] == PROPOSAL and source[0]['extraction_eligible'] is False
    assert source[1]['content'] == CONFIRM and source[1]['extraction_eligible'] is True
    assert source[2]['evidence_ids'] == [evidence.evidence_id] and source[2]['research_note_id'] == note['note_id']
    db.close()


def test_old_prepared_receipt_survives_added_provenance_defaults(tmp_path):
    store, service, _ = memory(tmp_path)
    old = store.add(value('不考虑微调'))
    candidate = MemoryCandidate(content=old.content, source=MemorySource(kind='conversation', quote=old.content), rationale='test')
    source = candidate.source.model_dump(mode='json', exclude={'messages', 'relation', 'grounding_reason'})
    key = 'job:' + hashlib.sha256(json.dumps([source, candidate.content, candidate.scope,
        candidate.workspace_id, candidate.session_id], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    with store.lock:
        revision, records, receipts = store._read()
        receipts[key] = old.memory_id
        store._write(revision, records, receipts, changed=False)
    service.delete(old.memory_id)
    assert service.commit_candidates([candidate], batch_id='job') == []
    assert store.processed(key) and store.list() == []


def test_recall_only_available_on_service_and_keeps_semantic_path(tmp_path):
    store, service, _ = memory(tmp_path)
    assert not hasattr(store, 'recall')
    wanted = store.add(value('不考虑微调'))
    assert service.recall('低秩适配器')[0].memory_id == wanted.memory_id
    service.delete(wanted.memory_id)
    assert service.recall('低秩适配器') == []


def test_recent_context_is_bounded_without_truncating_original_messages():
    history = [{'role': 'user', 'content': '旧资料' * 10000}, messages()[0]]
    request = contract().execution.model_copy(update={'query': CONFIRM, 'history': history})
    source = conversation_messages(request)
    assert [m['content'] for m in source] == [PROPOSAL, CONFIRM]
    assert [m['history_index'] for m in source] == [1, 2]


def test_manually_retired_version_cannot_be_recreated_but_explicit_save_can(tmp_path):
    store, service, llm = memory(tmp_path)
    old = store.add(value('以后采用微服务架构', memory_type='decision'))
    store.update(old.memory_id, value('以后采用单体架构', memory_type='decision'))
    archived = next(r for r in store.list(include_inactive=True) if r.status == 'superseded')
    text = '项目未来采用微服务部署架构'
    llm.responses = iter([{'candidates': [extraction(text)]},
        {'action': 'ignore', 'targets': [archived.memory_id], 'reason': '人工否决的旧版'}])
    assert service.extract([{'role': 'user', 'content': text}]) == []
    assert [r.content for r in store.list()] == ['以后采用单体架构']
    llm.responses = iter([{'action': 'update', 'targets': [old.memory_id], 'reason': '用户主动重新保存'}])
    assert service.add(value(text)).content == text


def test_unrelated_revision_conflict_replans_without_permanent_suppression(tmp_path, monkeypatch):
    store, service, llm = memory(tmp_path)
    old = store.add(value('SAGE Scholar 使用 FastAPI', memory_type='decision'))
    text = 'FNO 边界条件泛化需要核验'
    llm.responses = iter([{'candidates': [extraction(text, memory_type='research_note')]},
        {'action': 'new', 'targets': [], 'reason': '无关的项目配置'}])
    candidates = service.prepare_extraction([{'role': 'user', 'content': text}])
    real_commit = store.commit
    def concurrent_edit(*args, **kwargs):
        store.update(old.memory_id, value('SAGE Scholar 使用 FastAPI 单体架构', memory_type='decision'))
        return real_commit(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(store, 'commit', concurrent_edit)
        with pytest.raises(MemoryConflict):
            service.commit_candidates(candidates, batch_id='job')
    assert service.commit_candidates(candidates, batch_id='job')[0].content == text
