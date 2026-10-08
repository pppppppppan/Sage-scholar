"""Lifecycle tests use SQLite, native LangGraph and HTTP; no model/network calls."""
import asyncio
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient

from sage_scholar.api.app import create_app
from sage_scholar.application.query_service import QueryService, QueryCommand
from sage_scholar.conversations.service import ConversationService
from sage_scholar.domain.execution import RunOutcome
from sage_scholar.runtime import RunManager, RunStatus
from sage_scholar.runtime.store import RunStore
from sage_scholar.runtime.checkpoints import Checkpoints
from sage_scholar.runtime.human import HumanResponse, RunWaiting, public_interrupts
from sage_scholar.runtime.interaction import resume_command
from sage_scholar.runtime.control import check_control, cooperative_thread, CURRENT_CONTROL
from sage_scholar.routing import Workflow, QueryRouter
from sage_scholar.graph.graph_builder import build_graph
from sage_scholar.research.continuation import continue_graph
from research_fakes import contract, FakeSupervisor, FakeResearcher, FakeWriter, CompleteReview, SupportedVerifier

class Direct:
    def __init__(self): self.calls = []
    async def answer(self, request):
        self.calls.append(request)
        return "answer: " + request.query

class Runner:
    async def run(self, request, emit, cancelled):
        await emit("write", {"report": "research report"})
        return RunOutcome(result={"report": "research report"})


def setup(tmp_path):
    store, checkpoints = RunStore(tmp_path), Checkpoints(tmp_path)
    direct = Direct()
    manager = RunManager(store=store, checkpoints=checkpoints)
    runners = {Workflow.DEEP_RESEARCH: Runner()}
    query = QueryService(QueryRouter(), direct, None, manager, runners)
    runners[Workflow.DIRECT] = query.inline_runner
    manager.register_runners(runners)
    conversations = ConversationService(store.conversations, manager, query, None)
    async def close():
        await manager.shutdown()
        checkpoints.close(); store.close()
    return SimpleNamespace(run_store=store, checkpoints=checkpoints, run_manager=manager, query_service=query,
                           conversations=conversations, close=close, direct=direct)


def test_http_conversations_crud_history_idempotence_multiple_runs(tmp_path):
    container = setup(tmp_path)
    with TestClient(create_app(container)) as client:
        c = client.post('/api/conversations', json={}).json()['conversation']
        identity = c['conversation_id']
        body = dict(query='hello', workflow='direct', conversation_id=identity, request_id='one')
        run = client.post('/api/queries', json=body).json()['run_id']
        assert client.get(f'/api/runs/{run}/events').status_code == 200
        duplicate = client.post('/api/queries', json=body).json()['run_id']
        assert duplicate == run and len(container.direct.calls) == 1
        bad = {**body, 'query': 'different'}
        assert client.post('/api/queries', json=bad).status_code == 400
        run2 = client.post('/api/queries', json={**body, 'query': '请调研研究方向', 'workflow':'deep_research', 'request_id':'two'}).json()['run_id']
        client.get(f'/api/runs/{run2}/events')
        detail = client.get('/api/conversations/' + identity).json()
        assert len(detail['runs']) == 2 and len(detail['messages']) == 4
        assert detail['messages'][1]['content'] == 'answer: hello'
        assert detail['messages'][3]['content'] == 'research report'
        assert client.patch('/api/conversations/' + identity, json={'title':'renamed'}).status_code == 200
        assert len(client.get('/api/conversations?search=report').json()['conversations']) == 1
        client.patch('/api/conversations/' + identity, json={'archived':True})
        assert client.get('/api/conversations').json()['conversations'] == []
        assert len(client.get('/api/conversations?archived=true').json()['conversations']) == 1
        assert client.post('/api/queries', json={**body,'request_id':'three'}).status_code == 400
        client.patch('/api/conversations/' + identity, json={'archived':False})
        # A fresh conversation receives no context from the first one.
        fresh = client.post('/api/conversations', json={}).json()['conversation']['conversation_id']
        rid = client.post('/api/queries', json={**body,'conversation_id':fresh}).json()['run_id']
        client.get(f'/api/runs/{rid}/events')
        assert container.direct.calls[-1].history == []
        assert client.delete('/api/conversations/' + identity).status_code == 204
        assert client.get('/api/conversations/' + identity).status_code == 404
        assert client.get(f'/api/runs/{run}').status_code == 404
        assert container.run_store.conversations.source_deleted(identity)
        assert container.run_store.conversations.messages(identity) == []
        assert not container.checkpoints.saver.get_tuple(container.checkpoints.config(run))


def test_authoritative_history_and_restart_persistence(tmp_path):
    async def first():
        container = setup(tmp_path)
        identity = container.run_store.conversations.create().conversation_id
        for number in range(2):
            result = await container.conversations.submit(QueryCommand(query=f'hello {number}', workflow=Workflow.DIRECT,
                conversation_id=identity, history=[{'role':'user','content':'injected client history'}]))
            await container.run_manager._tasks[result.run_id]
        assert container.direct.calls[1].history[0]['content'] == 'hello 0'
        assert len(container.direct.calls[1].history) == 2
        await container.close()
        return identity
    identity = asyncio.run(first())
    container = setup(tmp_path)
    assert len(container.conversations.detail(identity)['messages']) == 4
    asyncio.run(container.close())


@pytest.mark.parametrize('workflow', [Workflow.DIRECT, Workflow.QA, Workflow.DEEP_RESEARCH, Workflow.CITATION_AUDIT])
def test_stop_cancels_backend_only_one_run_and_retains_output(tmp_path, workflow):
    async def scenario():
        store = RunStore(tmp_path)
        manager = RunManager(store=store)
        entered, release = asyncio.Event(), asyncio.Event()
        class Slow:
            async def run(self, request, emit, cancelled):
                await emit('answer_delta', {'text':'already written'})
                entered.set()
                await release.wait()
                return RunOutcome(result={'answer':'should not complete'})
        request = contract().execution.model_copy(update={'workflow':workflow})
        stopped = await manager.start(request, Slow())
        await entered.wait()
        other = await manager.start(request, Runner())
        await manager.cancel(stopped.run_id)
        await asyncio.gather(*list(manager._tasks.values()), return_exceptions=True)
        assert manager.get(stopped.run_id).status == RunStatus.CANCELLED
        assert manager.get(stopped.run_id).result['answer'] == 'already written'
        assert manager.get(other.run_id).status == RunStatus.COMPLETED
        again = await manager.cancel(other.run_id)
        assert again.status == RunStatus.COMPLETED
        await manager.shutdown(); store.close()
    asyncio.run(scenario())


def test_pending_cancel_delete_and_sse_disconnect(tmp_path):
    async def scenario():
        container = setup(tmp_path)
        identity = container.run_store.conversations.create().conversation_id
        called = []
        class Never:
            async def run(self,*args): called.append(True); await asyncio.sleep(10)
        req = contract().execution.model_copy(update={'conversation_id':identity})
        run = await container.run_manager.start(req, Never())
        await container.run_manager.cancel(run.run_id)  # before _execute gets its first timeslice
        assert container.run_manager.get(run.run_id).status == RunStatus.CANCELLED
        await container.conversations.delete(identity)
        assert called == []
        run = await container.run_manager.start(contract().execution, Runner())
        stream = container.run_manager.events(run.run_id)
        await anext(stream)
        await stream.aclose()  # closing SSE is not Stop
        await container.run_manager._tasks[run.run_id]
        assert container.run_manager.get(run.run_id).status == RunStatus.COMPLETED
        await container.close()
    asyncio.run(scenario())


class HumanRunner:
    def __init__(self, checkpoints, calls): self.checkpoints,self.calls = checkpoints,calls
    def graph(self):
        outer = self
        class Researcher(FakeResearcher):
            def run(self, *args): outer.calls.append(args[1].objective); return super().run(*args)
        return build_graph(FakeSupervisor(), lambda _: Researcher(), FakeWriter(), gap_reviewer=CompleteReview(),
            verifier=SupportedVerifier(), checkpointer=self.checkpoints.saver,
            interaction=lambda _: {'question':'比较精度还是算力？','options':['精度','算力'],'allow_skip':False})
    async def run(self, request, emit, cancelled):
        g = self.graph(); config = self.checkpoints.config(request.run_id)
        state = await asyncio.to_thread(g.get_state, config)
        value = resume_command(state, request.metadata.get('_resume_input', {}),
            None if state.values else {'research_contract':contract().model_copy(update={'execution':request})})
        output = await asyncio.to_thread(g.invoke, value, config)
        if output.get('__interrupt__'):
            raise RunWaiting(public_interrupts(output['__interrupt__']))
        return RunOutcome(result={'report':output['final_report']})


def test_native_hitl_restart_binding_duplicate_skip_and_resume(tmp_path):
    calls = []
    async def first():
        store, cp = RunStore(tmp_path), Checkpoints(tmp_path)
        identity = store.conversations.create().conversation_id
        manager = RunManager(store=store, checkpoints=cp)
        runner = HumanRunner(cp,calls)
        manager.register_runners({Workflow.DEEP_RESEARCH:runner})
        run = await manager.start(contract().execution.model_copy(update={'conversation_id':identity}), runner)
        events = [e async for e in manager.events(run.run_id)]
        saved = manager.get(run.run_id)
        assert saved.status == RunStatus.WAITING_FOR_USER and events[-1].type == 'waiting_for_user'
        assert cp.pending_interrupts(run.run_id)
        await manager.shutdown(); cp.close(); store.close()
        return saved
    saved = asyncio.run(first())
    async def second():
        store,cp = RunStore(tmp_path),Checkpoints(tmp_path)
        manager=RunManager(store=store,checkpoints=cp)
        manager.register_runners({Workflow.DEEP_RESEARCH:HumanRunner(cp,calls)})
        await manager.recover()
        assert not manager._tasks and calls == []
        assert manager.get(saved.run_id).status == RunStatus.WAITING_FOR_USER
        response = HumanResponse(run_id=saved.run_id, conversation_id=saved.execution.conversation_id,
            interrupt_id=saved.interrupts[0]['interrupt_id'], answer='算力')
        for wrong in [response.model_copy(update={'conversation_id':'wrong'}),response.model_copy(update={'interrupt_id':'old'}),response.model_copy(update={'skip':True})]:
            with pytest.raises(ValueError): await manager.answer(saved.run_id,wrong)
        await manager.answer(saved.run_id,response)
        with pytest.raises(ValueError): await manager.answer(saved.run_id,response)
        await manager._tasks[saved.run_id]
        assert manager.get(saved.run_id).status == RunStatus.COMPLETED
        assert len(calls)==1 and '算力' in calls[0]
        assert manager.get(saved.run_id).consumed_budget['active_seconds'] >= saved.consumed_budget['active_seconds']
        await manager.shutdown();cp.close();store.close()
    asyncio.run(second())


def test_partial_native_continuation_reuses_completed_notes(tmp_path):
    from sage_scholar.domain.research import GapReview, FollowUp, ResearchLimits
    calls=[]
    class Researcher(FakeResearcher):
        def run(self, task, *args): calls.append(task.task_id); return super().run(task,*args)
    class Review:
        def review(self, contract, tasks, notes):
            return GapReview(missing=['compute'], next_tasks=[FollowUp(objective='compute')]) if len(notes)==1 else GapReview()
    cp=Checkpoints(tmp_path)
    graph=build_graph(FakeSupervisor(),lambda _:Researcher(),FakeWriter(),gap_reviewer=Review(),verifier=SupportedVerifier(),checkpointer=cp.saver)
    request=contract().execution.model_copy(update={'run_id':'partial'})
    config=cp.config('partial')
    graph.invoke({'research_contract':contract()},config)
    before=graph.get_state(config)
    note=before.values['research_notes']['T1'].note_id
    assert before.values['report_result'].remaining_gaps and calls==['T1']
    request=request.model_copy(update={'metadata':{'continuation':1}})
    state=continue_graph(graph,config,before,request,ResearchLimits(max_rounds=1))
    assert state.next
    graph.invoke(None,config)
    final=graph.get_state(config)
    assert final.values['research_notes']['T1'].note_id==note and calls[0]=='T1' and calls.count('T1')==1
    assert len(calls)==2 and not final.values['report_result'].remaining_gaps
    cp.close()


def test_cancel_waits_for_owned_thread_and_stops_next_dispatch(tmp_path):
    import threading
    async def scenario():
        store=RunStore(tmp_path); manager=RunManager(store=store)
        entered, stopped=threading.Event(),threading.Event()
        calls=[]
        def work():
            entered.set()
            try:
                while True:
                    check_control()
                    stopped.wait(.01)
            finally: stopped.set()
        class Slow:
            async def run(self,request,emit,cancelled):
                await cooperative_thread(work)
                calls.append('next tool')
        run=await manager.start(contract().execution,Slow())
        await asyncio.to_thread(entered.wait,1)
        await manager.cancel(run.run_id)
        await asyncio.gather(*list(manager._tasks.values()),return_exceptions=True)
        assert stopped.is_set() and not calls
        assert manager.get(run.run_id).status==RunStatus.CANCELLED
        store.close()
    asyncio.run(scenario())


def test_cumulative_budget_persisted_and_not_context_limit(tmp_path):
    async def scenario():
        store=RunStore(tmp_path); manager=RunManager(store=store)
        class Limited:
            async def run(self,request,emit,cancelled):
                await emit('answer_delta',{'text':'progress'})
                control=CURRENT_CONTROL.get()
                store.record_usage(request.run_id,4090,0)
                control.reserve_model(100)
        run=await manager.start(contract().execution.model_copy(update={'budget':{'max_total_tokens':4096}}),Limited())
        await manager._tasks[run.run_id]
        saved=manager.get(run.run_id)
        assert saved.status==RunStatus.PARTIAL and saved.result['answer']=='progress'
        assert 'Context Window' in saved.reason and saved.consumed_budget['prompt_tokens']==4090
        lease=store.reserve_model(run.run_id,100,10000)
        assert lease
        store.clear_reservations(run.run_id)
        assert store.budget(run.run_id)['prompt_tokens']==4190
        store.close()
    asyncio.run(scenario())
