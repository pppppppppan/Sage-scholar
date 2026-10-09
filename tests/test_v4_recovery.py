import asyncio
import json

import pytest

from sage_scholar.domain.execution import RunOutcome
from sage_scholar.runtime import RunManager
from sage_scholar.runtime.models import RunEvent, RunSnapshot, RunStatus, utc_now
from sage_scholar.runtime.store import RunStore
from sage_scholar.runtime.checkpoints import Checkpoints
from sage_scholar.runtime.journal import JournaledTool, UnknownToolExecution, DurableResearcher
from sage_scholar.research.budget import ResearchBudget, BudgetedTool
from sage_scholar.graph.graph_builder import build_graph
from research_fakes import contract, FakeSupervisor, FakeResearcher, FakeWriter, CompleteReview, FakeSynthesis, SupportedReview
from sage_scholar.tools.base_tool import BaseTool


def graph(checkpoints, calls):
    class Researcher(FakeResearcher):
        def run(self, *args, **kwargs):
            calls.append(True)
            return super().run(*args, **kwargs)
    return build_graph(FakeSupervisor(), lambda _: Researcher(), FakeWriter(),
                       gap_reviewer=CompleteReview(), synthesizer=FakeSynthesis(), report_reviewer=SupportedReview(), checkpointer=checkpoints.saver)


def test_restart_resumes_same_native_checkpoint_notes_budget_and_sse(tmp_path):
    calls = []
    store, checkpoints = RunStore(str(tmp_path)), Checkpoints(str(tmp_path))
    run_id = "durable-run"
    request = contract().execution.model_copy(update={"run_id": run_id})
    g = graph(checkpoints, calls)
    list(g.stream({"research_contract": contract()}, checkpoints.config(run_id), interrupt_after=["research_node"], durability="sync"))
    assert calls == [True]
    assert store.reserve(run_id, 5)
    now = utc_now()
    snapshot = RunSnapshot(run_id=run_id, workflow=request.workflow, query=request.query, execution=request,
        status=RunStatus.RUNNING, created_at=now, updated_at=now, event_cursor=1,
        checkpoint=checkpoints.reference(run_id, request.workflow))
    store.append(snapshot, RunEvent(sequence=1, type="research_task_completed", data={"task_id": "T1"}))
    checkpoints.close(); store.close()
    # Fresh objects/connections simulate a new process, with no prior message list.
    store, checkpoints = RunStore(str(tmp_path)), Checkpoints(str(tmp_path))
    class Runner:
        async def run(self, request, emit, cancelled):
            result = await asyncio.to_thread(graph(checkpoints, calls).invoke, None, checkpoints.config(request.run_id))
            assert result["research_notes"]["T1"].evidence_ids == ["ev_a"]
            await emit("write", {"report": result["final_report"]})
            return RunOutcome(result={"report": result["final_report"]})
    async def scenario():
        manager = RunManager(store=store, checkpoints=checkpoints)
        manager.register_runners({request.workflow: Runner()})
        await manager.recover()
        events = [event async for event in manager.events(run_id, after=1)]
        assert events[0].type == "run_resumed" and events[-1].type == "run_finished"
        assert manager.get(run_id).status == RunStatus.COMPLETED
        assert manager.get(run_id).consumed_budget["tool_calls"] == 1
        assert calls == [True]  # Finished ResearchNote not regenerated.
        await manager.shutdown()
        again = RunManager(store=store, checkpoints=checkpoints)
        assert again.get(run_id).result["report"]
        replay = [e async for e in again.events(run_id, after=1)]
        assert [e.model_dump() for e in replay] == [e.model_dump() for e in events]
        assert [e async for e in again.events(run_id, after=events[-1].sequence)] == []
    asyncio.run(scenario())
    checkpoints.close(); store.close()


@pytest.mark.parametrize("schema", [4, 999])
def test_missing_or_incompatible_checkpoint_interrupts_instead_of_restarting(tmp_path, schema):
    store, checkpoints = RunStore(str(tmp_path)), Checkpoints(str(tmp_path))
    request, now = contract().execution, utc_now()
    snapshot = RunSnapshot(run_id="lost", workflow=request.workflow, query="q", execution=request,
        status=RunStatus.RUNNING, schema_version=schema, event_cursor=1, created_at=now, updated_at=now)
    store.append(snapshot, RunEvent(sequence=1,type="run_started"))
    async def scenario():
        manager = RunManager(store=store, checkpoints=checkpoints)
        await manager.recover()
        assert manager.get("lost").status == RunStatus.INTERRUPTED
        assert manager.get("lost").restart_required
        with pytest.raises(ValueError, match="restart_required"):
            await manager.resume("lost")
    asyncio.run(scenario())
    checkpoints.close(); store.close()


def test_committed_results_reused_and_manual_unknown_calls_not_replayed(tmp_path):
    store = RunStore(str(tmp_path))
    calls = []
    class Tool(BaseTool):
        source_access = "public"
        def __init__(self): super().__init__("fetch", "fetch", [])
        def run_tool(self, parameters):
            calls.append(parameters)
            return "source evidence"
    budget = ResearchBudget(5, store, "run")
    tool = JournaledTool(BudgetedTool(Tool(), budget), store, "run", "task:0")
    assert tool.run_tool({"url": "a"}) == tool.run_tool({"url": "a"})
    assert len(calls) == budget.used == 1
    from hashlib import sha256
    key = "task:0:" + sha256(json.dumps(["fetch", {"url":"b"}], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    store.call_started("run", key)
    assert store.unknown_calls("run", mark=True) == [key]
    with pytest.raises(UnknownToolExecution): tool.run_tool({"url":"b"})
    assert len(calls) == 1
    store.close()


def test_read_only_unknown_call_is_retried_and_legacy_record_does_not_block_recovery(tmp_path):
    store = RunStore(str(tmp_path))
    calls = []
    class ReadTool(BaseTool):
        source_access = "public"
        replay_policy = "safe_retry"
        def __init__(self): super().__init__("read_arxiv_paper", "read", [])
        def run_tool(self, parameters):
            calls.append(parameters)
            return "paper text"

    from hashlib import sha256
    parameters = {"paper_id": "2306.08543"}
    key = "T1:0:" + sha256(json.dumps(["read_arxiv_paper", parameters], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    # Legacy rows have no replay_policy, as in Runs created before this fix.
    store.call_started("run", key, {"tool": "read_arxiv_paper", "parameters": parameters,
                                     "task": "T1:0", "source_access": "public"})
    assert store.unknown_call_records("run", mark=True)[0]["status"] == "unknown"
    tool = JournaledTool(ReadTool(), store, "run", "T1:0")
    assert tool.run_tool(parameters) == "paper text"
    assert calls == [parameters]
    assert store.call("run", key)[0] == "completed"
    store.close()


def test_recovery_blocks_only_unknown_calls_that_require_manual_review(tmp_path):
    store = RunStore(str(tmp_path))
    request, now = contract().execution, utc_now()
    snapshot = RunSnapshot(
        run_id="run", workflow=request.workflow, query="q",
        execution=request.model_copy(update={"run_id": "run"}),
        status=RunStatus.INTERRUPTED, created_at=now, updated_at=now,
        restart_required=True,
        reason="外部工具执行状态未知，不能自动重放；请查看记录后明确重新开始",
    )
    store.append(snapshot, RunEvent(sequence=1, type="run_interrupted"))
    store.call_started("run", "safe", {"tool": "read_arxiv_paper"})

    class CompatibleCheckpoints:
        @staticmethod
        def compatible(_snapshot): return True

    manager = RunManager(store=store, checkpoints=CompatibleCheckpoints())
    manager.register_runners({request.workflow: object()})
    assert manager._recovery_issue(manager._runs["run"]) is None

    store.call_started("run", "unsafe", {"tool": "submit_external_job", "replay_policy": "manual"})
    assert "unknown_execution_state" in manager._recovery_issue(manager._runs["run"])
    store.close()


def test_note_commits_before_graph_checkpoint_and_reuses_on_retry(tmp_path):
    from sage_scholar.domain.research import ResearchTask
    calls = []
    class Researcher(FakeResearcher):
        def run(self, *args):
            calls.append(True)
            return super().run(*args)
    store = RunStore(str(tmp_path))
    task = ResearchTask(task_id="T1", objective="read")
    note = DurableResearcher(lambda _: Researcher(), store, "run", lambda: False).run(task, contract())
    store.close()
    store = RunStore(str(tmp_path))
    restored = DurableResearcher(lambda _: Researcher(), store, "run", lambda: False).run(task, contract())
    assert restored == note and calls == [True]
    store.close()
