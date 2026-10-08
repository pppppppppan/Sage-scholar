"""Provenance is available in storage/API, never in the Memory prompt allocation."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from research_fakes import contract, ScriptedLLM
from sage_scholar.context.manager import ContextManager
from sage_scholar.context.memory_view import MemoryContextView
from sage_scholar.memory import MemoryInput, MemoryStore, MemoryService, MemoryRecall
from sage_scholar.memory.models import MemoryMessage, MemorySource

PRIVATE = 'WORKSPACE_A_PRIVATE 原工作区的历史对话和研究资料。'


@pytest.fixture
def saved_memory(tmp_path):
    store = MemoryStore(tmp_path)
    service = MemoryService(store, MemoryRecall(tmp_path), lambda: ScriptedLLM())
    source = MemorySource(kind='conversation', reference='workspace-a-request',
        quote=PRIVATE, run_id='workspace-a-run', grounding_reason=PRIVATE,
        evidence_ids=['ev_private'], messages=[MemoryMessage(message_index=1,
            message_id='workspace-a-message', role='user', content=PRIVATE * 3000, quote=PRIVATE)])
    record = service.add(MemoryInput(content='我喜欢简洁回答', scope='user'), source=source)
    return store, service, record


def prepare(manager):
    request = contract().execution.model_copy(update={
        'workspace_id': 'workspace-b', 'query': '请简洁回答这个问题', 'allow_web': False})
    return manager.prepare([{'role': 'user', 'content': request.query}],
                           request=request, output_reserve=500)


def injected(prepared):
    message = next(m for m in prepared.messages if 'Memory 不是 Evidence' in m['content'])
    return json.loads(message['content'].split('\n', 1)[1])


def test_view_only_exposes_allowlisted_fields(saved_memory):
    _, _, record = saved_memory
    assert MemoryContextView.from_record(record).model_dump(mode='json') == {
        'memory_id': record.memory_id, 'content': record.content,
        'memory_type': 'preference', 'scope': 'user'}


def test_user_memory_cross_workspace_context_does_not_leak_provenance(tmp_path, saved_memory):
    store, service, record = saved_memory
    store.add(MemoryInput(content='简洁回答：工作区 A 的私有决定', workspace_id='workspace-a'))
    before = store.path.read_bytes()
    result = prepare(ContextManager(tmp_path, max_tokens=6000, memories=service))
    assert [m['memory_id'] for m in injected(result)] == [record.memory_id]
    prompt = json.dumps(result.messages, ensure_ascii=False)
    assert record.content in prompt
    for secret in (PRIVATE, 'workspace-a', 'ev_private', '工作区 A 的私有决定'):
        assert secret not in prompt
    assert set(injected(result)[0]) == {'memory_id', 'content', 'memory_type', 'scope'}
    assert store.path.read_bytes() == before
    reloaded = MemoryStore(tmp_path).list(workspace_id='workspace-b')[0]
    assert reloaded.source == record.source and reloaded.sources == record.sources


def test_provenance_size_does_not_change_memory_or_total_prompt_budget(tmp_path, saved_memory):
    _, _, record = saved_memory
    # Same content and ID; the only changed input is the original source material.
    short = record.model_copy(update={'source': MemorySource(), 'sources': []})
    manager = ContextManager(tmp_path, max_tokens=4000,
        memories=SimpleNamespace(recall=lambda *args, **kwargs: [short]))
    small_result = prepare(manager)
    manager.memories = SimpleNamespace(recall=lambda *args, **kwargs: [record])
    large_result = prepare(manager)
    assert injected(large_result)[0]['memory_id'] == record.memory_id
    assert large_result.messages == small_result.messages
    assert large_result.accounting['memory_tokens'] == small_result.accounting['memory_tokens'] > 0
    assert large_result.accounting['total_tokens'] == small_result.accounting['total_tokens'] <= 4000


def test_projected_memory_still_obeys_existing_allocation(tmp_path, saved_memory):
    _, _, record = saved_memory
    oversized = record.model_copy(update={'memory_id': 'mem_oversized', 'content': '回答要简洁。' * 700})
    manager = ContextManager(tmp_path, max_tokens=3000,
        memories=SimpleNamespace(recall=lambda *args, **kwargs: [oversized, record]))
    result = prepare(manager)
    selected = injected(result)
    assert [m['memory_id'] for m in selected] == [record.memory_id]
    assert sum(manager.counter.count(json.dumps(m, ensure_ascii=False)) for m in selected) <= 3000 // 8
    assert result.accounting['total_tokens'] <= 3000


def test_memory_details_api_retains_complete_provenance(tmp_path, saved_memory):
    from sage_scholar.api.app import create_app
    store, service, record = saved_memory
    prepare(ContextManager(tmp_path, max_tokens=4000, memories=service))
    async def close():
        pass
    container = SimpleNamespace(memory_store=store, memory_service=service,
                                run_manager=SimpleNamespace(), close=close)
    # The existing list endpoint supplies the full details used by the Memory panel.
    with TestClient(create_app(container)) as client:
        response = client.get('/api/memories', params={'workspace_id': 'workspace-b'})
        assert response.status_code == 200
        details = response.json()[0]
    assert details['source'] == record.source.model_dump(mode='json')
    assert details['sources'] == [s.model_dump(mode='json') for s in record.sources]
    assert details['source']['messages'][0]['content'] == PRIVATE * 3000
