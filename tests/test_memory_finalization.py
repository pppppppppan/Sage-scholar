"""Durable boundaries and lossless semantics, without real model/API calls."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from test_v43_memory import memory, value, extraction
from research_fakes import contract, ScriptedLLM
from sage_scholar.domain import EvidenceRef
from sage_scholar.domain.documents import utc_now
from sage_scholar.infrastructure.evidence_store import JsonEvidenceStore
from sage_scholar.memory import MemoryService, MemoryStore, MemoryRecall
from sage_scholar.memory.background import MemoryBackground
from sage_scholar.memory.jobs import MemoryJobStore
from sage_scholar.memory.research_source import ResearchMemorySource
from sage_scholar.runtime.store import RunStore
from sage_scholar.runtime.models import RunSnapshot, RunEvent


def test_lossless_merge_and_ignore_do_not_lose_fastapi(tmp_path):
    store, service, llm = memory(tmp_path)
    old = store.add(value('SAGE Scholar 使用 FastAPI，并且当前不考虑微调。'))
    llm.responses = iter([{'action':'merge','targets':[old.memory_id],'reason':'compatible'}])
    new = service.add(value('SAGE Scholar 当前不考虑 fine-tuning。'))
    assert old.content in new.content and 'fine-tuning' in new.content
    assert new.supersedes == [old.memory_id]
    before = new.model_dump()
    result = service.add(value(new.content))
    assert result.model_dump() == before
    assert len(store.list()) == 1


def test_merge_too_large_keeps_both_valid_memories(tmp_path):
    store, service, llm = memory(tmp_path)
    old = store.add(value('FastAPI ' * 800))
    text = 'FastAPI 项目补充配置 ' * 200
    llm.responses = iter([{'action':'merge','targets':[old.memory_id],'reason':'complementary'}])
    new = service.add(value(text))
    assert not new.supersedes
    assert {r.content for r in store.list()} == {old.content, text.strip()}


def test_manual_edit_is_protected_even_from_later_automatic_job(tmp_path):
    store, service, llm = memory(tmp_path)
    old = store.add(value('不考虑微调'))
    store.update(old.memory_id, value('手动决定：只考虑 LoRA'))
    text = '以后考虑全量微调'
    llm.responses = iter([{'candidates':[extraction(text)]},
        {'action':'update','targets':[old.memory_id],'reason':'new decision'}])
    assert service.extract([{'role':'user','content':text}]) == []
    assert store.list()[0].content == '手动决定：只考虑 LoRA'


@pytest.mark.parametrize('crash_after_commit', [False, True])
def test_restart_recovers_jobs_without_duplicate_memory(tmp_path, monkeypatch, crash_after_commit):
    text = '这个项目以后不考虑微调'
    store, service, llm = memory(tmp_path, {'candidates':[extraction(text)]})
    db = RunStore(tmp_path)
    jobs = MemoryJobStore(db)
    job = jobs.enqueue('conversation','request-1','default','s1',
        {'messages':[{'role':'user','content':text}]}, utc_now())
    background = MemoryBackground(service, None, jobs)
    if crash_after_commit:
        claimed = jobs.claim()
        real_commit = store.commit
        def crash(*args, **kwargs):
            real_commit(*args, **kwargs)
            raise SystemExit('crash after atomic memory+receipt, before job acknowledgement')
        with monkeypatch.context() as patch:
            patch.setattr(store, 'commit', crash)
            with pytest.raises(SystemExit):
                background._extract(claimed)
        assert jobs.get(job['job_id'])['status'] == 'running' and len(store.list()) == 1
    db.close()
    db = RunStore(tmp_path)
    jobs = MemoryJobStore(db)
    restored = MemoryBackground(service, None, jobs)
    async def scenario():
        await restored.recover()
        await restored.drain()
        assert jobs.get(job['job_id'])['status'] == 'succeeded'
        assert len(store.list()) == 1
        # Same source identity does not enqueue an independent task.
        repeated = jobs.enqueue('conversation','request-1','default','s1',{'messages':[]}, utc_now())
        assert repeated['job_id'] == job['job_id'] and repeated['status'] == 'succeeded'
        await restored.close()
    asyncio.run(scenario())
    assert len(llm.calls) == 1
    db.close()


def test_partial_batch_resume_keeps_temporal_update(tmp_path, monkeypatch):
    old, new = '项目不考虑微调', '以后项目考虑 LoRA 微调'
    store, service, llm = memory(tmp_path, {'candidates':[extraction(old), extraction(new,message_index=1)]})
    db = RunStore(tmp_path)
    jobs = MemoryJobStore(db)
    job = jobs.enqueue('conversation','source','default',None,{'messages':[
        {'role':'user','content':old}, {'role':'user','content':new}]},utc_now())
    background = MemoryBackground(service, None, jobs)
    real_commit = store.commit
    def crash(*args, **kwargs):
        real_commit(*args, **kwargs)
        raise SystemExit()
    with monkeypatch.context() as patch:
        patch.setattr(store, 'commit', crash)
        with pytest.raises(SystemExit):
            background._extract(jobs.claim())
    previous = store.list()[0]
    llm.responses = iter([{'action':'update','targets':[previous.memory_id],'reason':'later source turn'}])
    async def scenario():
        await background.recover()
        await background.drain()
        assert jobs.get(job['job_id'])['status'] == 'succeeded'
        assert len(store.list()) == 1 and store.list()[0].content == new
        assert previous.memory_id in store.list()[0].supersedes
        await background.close()
    asyncio.run(scenario())
    db.close()


def durable_research(tmp_path):
    text = 'FNO 在边界条件变化时可能存在泛化问题，仍需实验核验。'
    evidence_store = JsonEvidenceStore(str(tmp_path))
    evidence = EvidenceRef.create(source_id='https://example.org/paper',original_text=text,
        source_version='v1',parser_version='test',metadata={'kind':'source_content'})
    evidence_store.save(evidence)
    retrieval = SimpleNamespace(evidence_store=evidence_store,documents=None)
    db = RunStore(tmp_path)
    now = utc_now()
    request = contract().execution.model_copy(update={'run_id':'research-1'})
    run = RunSnapshot(run_id='research-1',workflow=request.workflow,execution=request,query=request.query,
        status='partial',created_at=now,updated_at=now,result={'report':'最终报告没有包含这个研究线索','claims':[]},
        remaining_gaps=['外推测试缺失'])
    note = {'note_id':'note_1','task_id':'T1','question':'边界条件泛化','findings':[
        {'text':text,'evidence_ids':[evidence.evidence_id]}],
        'unknowns':['跨数据集未知'],'conflicts':['论文结论不一致'],'limitations':['样本有限']}
    db.append(run, RunEvent(sequence=1,type='research_task_completed',data={'note':note}))
    db.append(run, RunEvent(sequence=2,type='gap_reviewed',data={'missing':['外推测试缺失']}))
    db.append(run, RunEvent(sequence=3,type='run_finished',data={'status':'partial'}))
    return db, retrieval, run, note, evidence


def test_research_reconcile_and_manual_reread_use_persisted_notes(tmp_path):
    db, retrieval, run, note, evidence = durable_research(tmp_path)
    text = note['findings'][0]['text']
    store, service, llm = memory(tmp_path, {'candidates':[extraction(text,message_index=1,memory_type='research_note')]})
    db.close()  # No live events or RunManager survives.
    db = RunStore(tmp_path)
    jobs = MemoryJobStore(db)
    background = MemoryBackground(service,retrieval,jobs)
    async def scenario():
        await background.recover()  # Reconciles finished Run whose callback never ran.
        await background.drain()
        saved = store.list()[0]
        assert (saved.source.run_id,saved.source.research_note_id,saved.source.task_id) == (run.run_id,'note_1','T1')
        assert saved.source.evidence_ids == [evidence.evidence_id]
        payload = json.loads(llm.calls[0]['messages'][-1]['content'])
        assert payload['background']['contract']['objective'] == run.query
        assert payload['background']['remaining_gaps'] == ['外推测试缺失']
        assert payload['background']['research_notes'][0]['conflicts'] == ['论文结论不一致']
        llm.responses = iter([{'candidates':[extraction(text,message_index=1,memory_type='research_note')]}])
        repeated = background.enqueue_research(run.run_id,retry=True)
        await background.drain()
        assert len(store.list()) == 1 and jobs.get(repeated['job_id'])['status'] == 'succeeded'
        assert text in llm.calls[-1]['messages'][-1]['content']
        await background.close()
    asyncio.run(scenario())
    db.close()


def test_checkpoint_research_input_is_authoritative(tmp_path):
    from sage_scholar.runtime.checkpoints import Checkpoints
    from test_v4_recovery import graph
    from sage_scholar.domain.research import ResearchContract
    db, retrieval, run, note, evidence = durable_research(tmp_path)
    checkpoints = Checkpoints(tmp_path)
    actual = ResearchContract.from_request(run.execution.model_copy(update={'query':'持久契约指定比较维度'}))
    list(graph(checkpoints,[]).stream({'research_contract':actual},checkpoints.config(run.run_id),
        interrupt_after=['research_node'],durability='sync'))
    run.checkpoint = checkpoints.reference(run.run_id,run.workflow)
    db.append(run,RunEvent(sequence=4,type='run_finished',data={}))
    _, _, background = ResearchMemorySource(db,retrieval,checkpoints).load(run.run_id)
    assert background['contract']['objective'] == actual.objective
    assert background['research_notes'][0]['question'] != note['question']
    checkpoints.close(); db.close()


def test_failed_job_has_bounded_retry_and_can_be_rearmed(tmp_path):
    db = RunStore(tmp_path)
    jobs = MemoryJobStore(db)
    job = jobs.enqueue('conversation','req','default',None,{'messages':[]},utc_now())
    for i in range(3):
        assert jobs.claim()['attempts'] == i+1
        jobs.fail(job['job_id'],'provider unavailable')
        with db.transaction() as conn:
            conn.execute('UPDATE memory_jobs SET available_at=0')
    assert jobs.get(job['job_id'])['status'] == 'failed' and jobs.claim() is None
    jobs.enqueue('conversation','req','default',None,{'messages':[]},utc_now(),retry=True)
    assert jobs.claim()['attempts'] == 1
    db.close()


def test_deleted_memory_receipt_cannot_be_replayed_after_restart(tmp_path):
    store, service, llm = memory(tmp_path, {'candidates':[extraction('不考虑微调')]})
    candidates = service.prepare_extraction([{'role':'user','content':'不考虑微调'}])
    saved = service.commit_candidates(candidates,batch_id='job')[0]
    store.delete(saved.memory_id)
    restored = MemoryService(MemoryStore(tmp_path), MemoryRecall(tmp_path), lambda: ScriptedLLM())
    assert restored.commit_candidates(candidates,batch_id='job') == []
    assert restored.recall('微调') == []


def test_memory_with_provenance_still_not_a_citation_source(tmp_path):
    from sage_scholar.research.evidence_context import ResearchEvidenceContext
    from sage_scholar.research.verification import CitationVerifier
    from sage_scholar.domain.research import Claim, ReportDraft
    store, service, _ = memory(tmp_path)
    saved = store.add(value('FNO 比 PINN 好'))
    context = ResearchEvidenceContext(JsonEvidenceStore(str(tmp_path)),contract().source_policy)
    result = CitationVerifier(ScriptedLLM(),context).verify(ReportDraft(title='x',claims=[
        Claim(claim_id='C1',text='FNO always wins',evidence_ids=[saved.memory_id])]))
    assert result.claims[0].identity_valid is False
