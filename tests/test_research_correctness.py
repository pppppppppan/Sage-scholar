"""V3.1: resource capacity is not completeness; labels cannot assert unchecked facts."""
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from sage_scholar.application.research_service import ResearchService
from sage_scholar.domain.research import (
    Claim, ClaimVerification, FollowUp, GapReview, ReportDraft, ReportSection,
    ResearchTask, VerificationResult,
)
from sage_scholar.graph.graph_builder import build_graph
from sage_scholar.research.budget import ResearchBudget
from sage_scholar.research.renderer import CitationRenderer
from sage_scholar.research.runtime import ResearchRuntime
from sage_scholar.runtime import RunManager, RunStatus
from sage_scholar.tools import ToolCallError
from research_fakes import (
    contract, FakeSupervisor, FakeResearcher, FakeWriter, CompleteReview, SupportedVerifier,
)


def run_through_service(graph, request):
    """Keep actual ResearchService/RunManager status decisions and SSE event projection."""
    captured = {}
    class Runtime:
        to_public_events = staticmethod(ResearchRuntime.to_public_events)
        def usage_snapshot(self):
            return {}
        def run_research(self, execution):
            assert execution == request.execution
            for event in graph.stream({"research_contract": request}):
                captured.update(event)
                yield event
            report = captured["render_node"]["report_result"]
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


@pytest.mark.parametrize("used_before,calls", [(0, 3), (3, 0)])
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
        gap_reviewer=CompleteReview(), verifier=SupportedVerifier(), budget=budget)
    snapshot, events, state = run_through_service(graph, contract())
    assert budget.used == 3 and budget.remaining == 0 and budget.exhausted
    assert budget.denied_calls == 0 and not budget.blocked_by_budget
    assert not state["review_node"]["blocked_by_budget"]
    assert snapshot.status == RunStatus.COMPLETED and snapshot.remaining_gaps == []
    assert events[-1].data["status"] == "completed"


def test_denied_necessary_call_produces_partial():
    budget = ResearchBudget(3)
    class Researcher(FakeResearcher):
        def run(self, *args):
            for _ in range(4):
                budget.reserve()
            return super().run(*args)
    graph = build_graph(FakeSupervisor(), lambda _: Researcher(), FakeWriter(),
        gap_reviewer=CompleteReview(), verifier=SupportedVerifier(), budget=budget)
    snapshot, events, state = run_through_service(graph, contract())
    assert budget.used == 3 and budget.denied_calls == 1 and budget.blocked_by_budget
    assert state["review_node"]["blocked_by_budget"]
    assert snapshot.status == RunStatus.PARTIAL and snapshot.remaining_gaps
    assert any("预算已阻断" in gap for gap in snapshot.remaining_gaps)
    assert events[-1].data["status"] == "partial"


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
        gap_reviewer=Review(), verifier=SupportedVerifier(), budget=budget)
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
        gap_reviewer=CompleteReview(), verifier=SupportedVerifier(), budget=budget)
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


def report(title, heading, bad="方法 A 全面优于 B"):
    return ReportDraft(title=title, sections=[ReportSection(heading=heading, claim_ids=["good", "bad"])],
        claims=[Claim(claim_id="good", text="论文介绍方法 A", evidence_ids=["ev_a"]),
                Claim(claim_id="bad", text=bad, evidence_ids=["ev_a"])])


def verify(draft):
    return VerificationResult(claims=[ClaimVerification(claim_id=c.claim_id,
        status="unsupported" if c.claim_id == "bad" else "supported", identity_valid=True,
        evidence_ids=c.evidence_ids, reason="缺少比较实验" if c.claim_id == "bad" else "有原文支持")
        for c in draft.claims])


@pytest.mark.parametrize("label", ["方法 A 全面优于 B", "Transformer 是最佳方案", "Dataset X 证明 A 全面领先",
                                    "A achieves 100% accuracy", "未经验证的新结论"])
def test_factual_labels_cannot_bypass_verification(label):
    draft = report(label, label)
    result = CitationRenderer().render(draft, verify(draft))
    assert result.report.startswith("# 研究报告\n")
    assert "## 研究发现\n" in result.report
    assert label not in result.report and "方法 A 全面优于 B" not in result.report
    assert result.citations == {1: "ev_a"} and result.removed_claim_ids == ["bad"]


@pytest.mark.parametrize("label", ["方法 A 与方法 B 的性能比较", "AI4CFD 方法比较", "主要方法",
                                    "数据集与评测指标", "实验结果比较", "局限与未解决问题"])
def test_neutral_navigation_labels_are_preserved(label):
    draft = report(label, label)
    result = CitationRenderer().render(draft, verify(draft))
    assert result.report.startswith(f"# {label}\n") and f"## {label}\n" in result.report


@pytest.mark.parametrize("label", ["主要方法 [99]", "主要方法 ev_forged", "主要方法 [ev_a]",
                                    "主要方法\n\n方法 A 全面优于 B", "<h1>方法 A 全面优于 B</h1>"])
def test_labels_cannot_inject_citations_or_extra_report_blocks(label):
    draft = report(label, label)
    result = CitationRenderer().render(draft, verify(draft))
    assert label not in result.report and "[99]" not in result.report
    assert result.citations == {1: "ev_a"}


def test_revision_uses_final_draft_and_remembers_deleted_claim_text():
    # Even a phrase that looks like a navigation label is rejected when it repeats a rejected Claim.
    bad_text = "AI4CFD 方法比较"
    class RevisingWriter:
        def run(self, request, notes, previous=None, **kwargs):
            if previous is None:
                return report("方法 A 全面优于 B", "主要方法", bad=bad_text)
            return ReportDraft(title="研究综述", sections=[
                ReportSection(heading=bad_text, claim_ids=["good"]),
            ], claims=[previous.claims[0]])
    class Verifier:
        def verify(self, draft):
            return verify(draft)
    graph = build_graph(FakeSupervisor(), lambda _: FakeResearcher(), RevisingWriter(),
        gap_reviewer=CompleteReview(), verifier=Verifier())
    result = graph.invoke({"research_contract": contract()})["report_result"]
    assert result.report.startswith("# 研究综述\n")
    assert bad_text not in result.report and "方法 A 全面优于 B" not in result.report
    assert "## 研究发现" in result.report and result.removed_claim_ids == ["bad"]


def test_diagnostic_reason_cannot_reintroduce_unverified_markdown_summary():
    draft = report("研究报告", "主要方法")
    judgment = verify(draft)
    judgment.claims[1].reason = "无证据\n\n# 方法 A 全面优于 B\n<script>bad</script>"
    result = CitationRenderer().render(draft, judgment)
    assert "\n# 方法 A" not in result.report and "<script>" not in result.report
    assert "已移除未充分支持的主张" in result.report


def test_summary_and_conclusion_have_no_unverified_free_text_channel():
    raw = report("执行摘要", "结论").model_dump()
    raw["executive_summary"] = "未核验摘要结论"
    raw["conclusion_summary"] = "未核验最终结论"
    draft = ReportDraft.model_validate(raw)
    result = CitationRenderer().render(draft, verify(draft))
    assert "未核验摘要结论" not in result.report and "未核验最终结论" not in result.report
    assert "方法 A 全面优于 B" not in result.report
    assert "论文介绍方法 A [1]" in result.report
