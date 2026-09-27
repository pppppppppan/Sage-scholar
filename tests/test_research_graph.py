import threading

import pytest

from sage_scholar.domain.research import FollowUp, GapReview, ResearchTask
from sage_scholar.graph.graph_builder import build_graph
from sage_scholar.research.tasks import follow_up_tasks, validate_tasks
from research_fakes import (
    contract, FakeSupervisor, FakeResearcher, FakeWriter, CompleteReview, SupportedVerifier,
)


def test_research_returns_report_without_rag_write_dependency():
    graph = build_graph(FakeSupervisor(), lambda _: FakeResearcher(), FakeWriter(),
                        gap_reviewer=CompleteReview(), verifier=SupportedVerifier())
    result = graph.invoke({"research_contract": contract()})
    assert result["final_report"].startswith("# 最终报告")
    assert result["report_result"].citations == {1: "ev_a"}
    assert result["report_result"].remaining_gaps == []
    assert result["research_notes"]["T1"].evidence_ids == ["ev_a"]
    # No retrieval/pipeline is injected into the graph or Writer: reports cannot be ingested.


def test_missing_review_entries_are_reported_as_gaps():
    class MissingReview(CompleteReview):
        def review(self, *args):
            return GapReview(missing=["未完成审查"], next_tasks=[FollowUp(objective="补证")])
    graph = build_graph(FakeSupervisor(), lambda _: FakeResearcher(), FakeWriter(),
                        gap_reviewer=MissingReview(), verifier=SupportedVerifier())
    result = graph.invoke({"research_contract": contract()})
    assert "未完成审查" in result["report_result"].remaining_gaps
    assert "已达到研究补查轮数上限" in result["final_report"]


def test_independent_tasks_are_parallel_and_dependencies_wait():
    barrier = threading.Barrier(2, timeout=3)
    finished = set()
    class Supervisor:
        def plan_tasks(self, _):
            return [ResearchTask(task_id="A", objective="a"), ResearchTask(task_id="B", objective="b"),
                    ResearchTask(task_id="C", objective="c", dependencies=["A", "B"])]
    class Researcher(FakeResearcher):
        def run(self, task, contract, dependency_notes):
            if task.task_id in {"A", "B"}:
                barrier.wait()
                finished.add(task.task_id)
            else:
                assert finished == {"A", "B"}
                assert {note.task_id for note in dependency_notes} == {"A", "B"}
            return super().run(task, contract, dependency_notes)
    graph = build_graph(Supervisor(), lambda _: Researcher(), FakeWriter(),
                        gap_reviewer=CompleteReview(), verifier=SupportedVerifier())
    result = graph.invoke({"research_contract": contract()})
    assert {task.status for task in result["tasks"].values()} == {"completed"}


@pytest.mark.parametrize("tasks", [
    [ResearchTask(task_id="A", objective="a", dependencies=["A"])],
    [ResearchTask(task_id="A", objective="a", dependencies=["missing"])],
    [ResearchTask(task_id="A", objective="a"), ResearchTask(task_id="A", objective="duplicate")],
])
def test_dag_invalid_plans_fail_explicitly(tasks):
    with pytest.raises(ValueError):
        validate_tasks(tasks)


def test_retry_retains_identity_and_invalidates_descendants_once():
    tasks = validate_tasks([ResearchTask(task_id="A", objective="a", status="completed"),
        ResearchTask(task_id="B", objective="b", dependencies=["A"], status="completed")])
    updated, changed = follow_up_tasks(tasks, [FollowUp(task_id="A", objective="check data"),
        FollowUp(task_id="A", objective="check metric")], 1, 10)
    assert changed == {"A", "B"}
    assert updated["A"].revision == updated["B"].revision == 1
    assert updated["A"].status == updated["B"].status == "pending"
    assert "check data" in updated["A"].objective and "check metric" in updated["A"].objective
    assert tasks["A"].status == "completed"


def test_task_count_limit_stops_unproductive_replanning():
    class MissingReview:
        calls = 0
        def review(self, *args):
            self.calls += 1
            return GapReview(missing=["missing new dimension"], next_tasks=[FollowUp(objective="new task")])
    review = MissingReview()
    graph = build_graph(FakeSupervisor(), lambda _: FakeResearcher(), FakeWriter(),
                        gap_reviewer=review, verifier=SupportedVerifier())
    report = graph.invoke({"research_contract": contract(max_tasks=1, max_rounds=10)})["report_result"]
    assert review.calls == 2
    assert "任务数量预算已耗尽，无法新增补查任务" in report.remaining_gaps
