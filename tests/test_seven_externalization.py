import pytest
from sage_scholar.runtime.journal import WorkflowInterrupted
from test_react_progress import setup, Model, calls, answer


def test_one_externalization_shared_by_success_cache_history_and_resume(tmp_path):
    batch = calls(1, 1)
    batch.tool_calls[1].id = 'call_repeat'
    agent, raw, task, request, progress, store = setup(tmp_path, Model(batch, RuntimeError('stop')))
    externalized = []
    def externalize(text):
        externalized.append(text)
        return '[artifact_id: one] [ev_a] saved original'
    agent._externalize = externalize
    with pytest.raises(WorkflowInterrupted):
        agent.run(task, request)
    assert len(externalized) == 1 and raw.calls == [1]
    saved = progress.load(progress.fingerprint(task, request))
    outputs = [m for m in saved['history'] if m['role'] == 'tool']
    assert len(outputs) == 2 and outputs[0]['content'] == outputs[1]['content']
    store.close()
    repeated = calls(1)
    repeated.tool_calls[0].id = 'call_resume'
    resumed, raw2, task, request, progress, store = setup(tmp_path, Model(repeated, answer()))
    resumed._externalize = lambda _: pytest.fail('completed cached result externalized again')
    assert resumed.run(task, request).source_ids == ['ev_a']
    assert raw2.calls == []
    store.close()
