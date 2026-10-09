"""V3.1: resource capacity is not completeness; labels cannot assert unchecked facts."""
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from sage_scholar.application.research_service import ResearchService
from sage_scholar.domain.research import (
    FollowUp, GapReview, ResearchTask,
)
from sage_scholar.graph.graph_builder import build_graph
from sage_scholar.research.budget import ResearchBudget
from sage_scholar.research.runtime import ResearchRuntime
from sage_scholar.runtime import RunManager, RunStatus
from sage_scholar.tools import ToolCallError
from research_fakes import (
    contract, FakeSupervisor, FakeResearcher, FakeWriter, CompleteReview, FakeSynthesis, SupportedReview,
)


def run_through_service(graph, request):
    """Keep actual ResearchService/RunManager status decisions and SSE event projection."""
    captured = {}
    class Runtime:
        to_public_events = staticmethod(ResearchRuntime.to_public_events)
        def usage_snapshot(self):
            return {}
        def run_research(self, execution):
            assert execution.run_id  # Lifecycle attaches identity without changing the research request.
            assert execution.model_dump(exclude={"run_id", "budget"}) == request.execution.model_dump(exclude={"run_id", "budget"})
            for event in graph.stream({"research_contract": request}):
                captured.update(event)
                yield event
            report = captured["finalize_node"]["report_result"]
            yield {"outcome_node": {"remaining_gaps": report.remaining_gaps}}
    class Provider:
        def get(self, require_external):
            return Runtime()
    async def scenario():
        manager = RunManager()
        run = await manager.start(request.execution, ResearchService(Provider()))
        events = [event async for event in manager.events(run.run_id)]
        snapshot = manager.get(run.run_id)
        await manager.shutdown()
        return snapshot, events
    snapshot, events = asyncio.run(scenario())
    return snapshot, events, captured


@pytest.mark.parametrize("used_before,calls", [(0, 3)])
def test_exactly_full_budget_without_required_work_is_completed(used_before, calls):
    budget = ResearchBudget(3)
    for _ in range(used_before):
        budget.reserve()
    class Supervisor(FakeSupervisor):
        def plan_tasks(self, request):
            task = super().plan_tasks(request)[0]
            if not calls:
                task.status = "completed"
            return [task]
    class Researcher(FakeResearcher):
        def run(self, *args):
            for _ in range(calls):
                budget.reserve()
            return super().run(*args)
    graph = build_graph(Supervisor(), lambda _: Researcher(), FakeWriter(),
        gap_reviewer=CompleteReview(), synthesizer=FakeSynthesis(), report_reviewer=SupportedReview(), budget=budget)
    snapshot, events, state = run_through_service(graph, contract())
    assert budget.used == 3 and budget.remaining == 0 and budget.exhausted
    assert budget.denied_calls == 0 and not budget.blocked_by_budget
    assert not state["review_node"]["blocked_by_budget"]
    assert snapshot.status == RunStatus.COMPLETED and snapshot.remaining_gaps == []
    assert events[-1].data["status"] == "completed"


def test_denied_call_without_any_evidence_cannot_publish_fake_report():
    budget = ResearchBudget(3)
    class Researcher(FakeResearcher):
        def run(self, *args):
            for _ in range(4):
                budget.reserve()
            return super().run(*args)
    graph = build_graph(FakeSupervisor(), lambda _: Researcher(), FakeWriter(),
        gap_reviewer=CompleteReview(), synthesizer=FakeSynthesis(), report_reviewer=SupportedReview(), budget=budget)
    snapshot, events, state = run_through_service(graph, contract())
    assert budget.used == 3 and budget.denied_calls == 1 and budget.blocked_by_budget
    assert state["review_node"]["blocked_by_budget"]
    assert snapshot.status == RunStatus.FAILED
    assert snapshot.result is None
    assert "可用原文证据" in snapshot.error


def test_required_followup_blocked_without_making_denied_call():
    budget = ResearchBudget(1)
    attempts = []
    class Researcher(FakeResearcher):
        def run(self, task, *args):
            attempts.append(task.task_id)
            budget.reserve()
            return super().run(task, *args)
    class Review:
        def review(self, *args):
            return GapReview(missing=["compute budget"], next_tasks=[FollowUp(objective="check compute")])
    graph = build_graph(FakeSupervisor(), lambda _: Researcher(), FakeWriter(),
        gap_reviewer=Review(), synthesizer=FakeSynthesis(), report_reviewer=SupportedReview(), budget=budget)
    snapshot, _, state = run_through_service(graph, contract(max_rounds=3))
    assert attempts == ["T1"] and budget.denied_calls == 0 and budget.blocked_by_budget
    assert state["review_node"]["blocked_by_budget"]
    assert snapshot.status == RunStatus.PARTIAL
    assert "compute budget" in snapshot.remaining_gaps


def test_budget_blocks_ready_dependent_task_only_after_last_slot_used():
    budget = ResearchBudget(1)
    attempts = []
    class Supervisor:
        def plan_tasks(self, request):
            return [ResearchTask(task_id="A", objective="read"),
                    ResearchTask(task_id="B", objective="verify", dependencies=["A"])]
    class Researcher(FakeResearcher):
        def run(self, task, *args):
            attempts.append(task.task_id)
            budget.reserve()
            return super().run(task, *args)
    graph = build_graph(Supervisor(), lambda _: Researcher(), FakeWriter(),
        gap_reviewer=CompleteReview(), synthesizer=FakeSynthesis(), report_reviewer=SupportedReview(), budget=budget)
    snapshot, _, state = run_through_service(graph, contract())
    assert attempts == ["A"] and budget.blocked_by_budget and budget.denied_calls == 0
    assert state["dispatch_node"]["tasks"]["B"].status == "blocked"
    assert snapshot.status == RunStatus.PARTIAL


def test_parallel_reservations_preserve_cap_and_count_rejections():
    budget = ResearchBudget(10)
    def reserve(_):
        try:
            budget.reserve()
            return True
        except ToolCallError:
            return False
    with ThreadPoolExecutor(max_workers=16) as pool:
        accepted = list(pool.map(reserve, range(100)))
    assert sum(accepted) == budget.used == 10
    assert budget.denied_calls == 90 and budget.remaining == 0 and budget.blocked_by_budget
