"""Small offline checks for prefix stability, cache reporting and report reserve."""
from types import SimpleNamespace

import pytest

from sage_scholar.base.cache_usage import cached_input, anthropic_input
from sage_scholar.runtime.control import RunControl, ResearchBudgetReached, RunBudgetExceeded
from sage_scholar.runtime.store import RunStore
from sage_scholar.agents.researcher import Researcher


def test_cache_reporting_preserves_unknown_and_persists_observed_denominator(tmp_path):
    assert cached_input({}) is None
    assert cached_input({'prompt_cache_hit_tokens': 700}) == 700
    assert cached_input({'prompt_tokens_details': {'cached_tokens': 600}}) == 600
    assert anthropic_input({'input_tokens': 100, 'cache_read_input_tokens': 700,
                            'cache_creation_input_tokens': 200}) == (1000, 700)
    store = RunStore(tmp_path)
    store.record_usage('r', 1000, 100, cached_tokens=700)
    store.record_usage('r', 500, 50)
    store.close()
    store = RunStore(tmp_path)
    usage = store.budget('r')
    assert usage['prompt_tokens'] == 1500
    assert usage['cached_input_tokens'] == 700
    assert usage['cache_observed_input_tokens'] == 1000
    assert usage['cache_observed_calls'] == 1
    store.close()


def test_collection_reserve_stops_react_but_allows_report_and_retains_hard_limit(tmp_path):
    store = RunStore(tmp_path)
    request = SimpleNamespace(run_id='r', workflow='deep_research',
        budget={'max_total_tokens': 10000, 'report_reserve_tokens': 3000})
    control = RunControl(request, store, lambda: False)
    store.record_usage('r', 6900, 0)
    with pytest.raises(ResearchBudgetReached):
        control.reserve_model(200, purpose='T1:react')
    assert control.research_exhausted()
    lease = control.reserve_model(2000, purpose='research:writer')
    assert lease
    control.release_model(lease)
    with pytest.raises(RunBudgetExceeded):
        control.reserve_model(4000, purpose='research:writer')
    store.close()


def test_repeated_successful_tool_call_reuses_result_with_current_call_id():
    invocations = []
    class Tool:
        name = 'read'
        def to_schema(self):
            return {'type': 'function', 'function': {'name': 'read'}}
        def run_tool(self, parameters):
            invocations.append(parameters)
            return 'Original source ev_1'
    researcher = Researcher('T1', None, [Tool()])
    researcher._execute_tool('read', {'url': 'source'}, 'call1', 'question')
    researcher._execute_tool('read', {'url': 'source'}, 'call2', 'question')
    assert len(invocations) == 1
    assert researcher._history[-1].tool_call_id == 'call2'
    assert researcher._history[-1].content == 'Original source ev_1'
