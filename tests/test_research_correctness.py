"""Resource capacity is not research completeness."""
from concurrent.futures import ThreadPoolExecutor
from sage_scholar.research.budget import ResearchBudget
from sage_scholar.graph.graph_builder import build_graph
from sage_scholar.tools import ToolCallError
from sage_scholar.domain.research import SupervisorReview, SubQuestion
from research_fakes import research_request, FakeSupervisor, FakeResearcher, FakeWriter


def test_exactly_full_budget_without_required_work_is_completed():
    budget = ResearchBudget(1)
    class Researcher(FakeResearcher):
        def run(self, *args):
            budget.reserve()
            return super().run(*args)
    result = build_graph(FakeSupervisor(), lambda _: Researcher(), FakeWriter(), budget=budget).invoke(
        {"execution": research_request()})
    assert budget.exhausted and not result["report_result"].partial


def test_required_followup_blocked_still_produces_report():
    budget = ResearchBudget(1)
    class Researcher(FakeResearcher):
        def run(self, *args):
            budget.reserve()
            return super().run(*args)
    class Supervisor(FakeSupervisor):
        def review(self, *args):
            return SupervisorReview(remaining_gaps=["compute"], follow_up_questions=[SubQuestion(objective="compute")])
    result = build_graph(Supervisor(), lambda _: Researcher(), FakeWriter(), budget=budget).invoke(
        {"execution": research_request(max_rounds=3)})
    assert len(result["research_notes"]) == 1 and result["report_result"].partial
    assert result["final_report"] and "compute" in result["report_result"].remaining_gaps


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
