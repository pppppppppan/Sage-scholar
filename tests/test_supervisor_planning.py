"""Planning contracts/dispatch, with deterministic model outputs (no paid model calls)."""
import json

import pytest
from pydantic import ValidationError

from research_fakes import ScriptedLLM, research_request, FakeSupervisor, FakeResearcher, FakeWriter
from sage_scholar.agents.supervisor import Supervisor, ResearchPlan
from sage_scholar.domain.research import SubQuestion, ResearchTask, SupervisorReview
from sage_scholar.graph.graph_builder import build_graph
from sage_scholar.research.planning import consolidate, within_budget, novel_followups, check_coverage
from sage_scholar.research.continuation import continue_graph
from sage_scholar.runtime.checkpoints import Checkpoints


def question(objective, scope, covers=None, **extra):
    return {"objective": objective, "scope_keys": [scope], "expected_output": "具体回答：" + objective,
            "covers": covers or [objective], **extra}


@pytest.mark.parametrize("query,objectives", [
    ("OPD 和传统知识蒸馏有什么区别？", ["比较采样分布、教师反馈及训练目标"]),
    ("调研 OPD 的核心原理", ["解释 OPD 的采样与反馈机制"]),
    ("调研 OPD 的代表性方法及实验效果", ["比较代表方法机制", "独立核对代表方法的实验条件及效果"]),
    ("全面调研 OPD 的原理、代表方法、实验进展及局限", ["原理及理论局限", "方法及适用边界", "实验进展及评估局限"]),
    ("调研 OPD，并分析如何结合 PASA", ["OPD 机制与现状", "自行核实 PASA 必要前提并提出 OPD 结合方案，不重做背景综述"]),
])
def test_minimum_plan_flows_without_forced_padding(query, objectives):
    llm = ScriptedLLM({"requirements": objectives, "questions": [question(v, f"outcome:{i}") for i, v in enumerate(objectives)]})
    request = research_request().model_copy(update={"query": query})
    planned = Supervisor(llm, 8).plan(request)
    assert len(planned) == len(objectives) and len(llm.calls) == 1
    class Plan(FakeSupervisor):
        def plan(self, request):
            return planned
    result = build_graph(Plan(), lambda _: FakeResearcher(), FakeWriter()).invoke({"execution": request})
    assert len(result["tasks"]) == len(objectives) == len(result["research_notes"])
    assert result["final_report"] and all(t.status == "completed" for t in result["tasks"].values())


def test_semantic_scope_merge_retains_distinct_requirements_and_outputs():
    requirements = ["采样原理", "与传统蒸馏比较"]
    llm = ScriptedLLM({"requirements": requirements, "questions": [
        question("解释学生自身轨迹采样", "opd:distribution", [requirements[0]]),
        question("分析 on-policy 相对固定数据蒸馏的差异", "opd:distribution", [requirements[1]])]})
    planned = Supervisor(llm, 8).plan(research_request())
    assert len(planned) == 1 and len(llm.calls) == 1
    assert planned[0].covers == requirements
    assert "学生自身轨迹" in planned[0].objective and "固定数据" in planned[0].objective
    assert "固定数据" in planned[0].expected_output
    check_coverage(requirements, planned)


def test_same_paper_with_different_outcomes_is_not_a_duplicate():
    values = [SubQuestion(**question("阅读 GKD 分析方法", "gkd:method")),
              SubQuestion(**question("阅读 GKD 核对实验条件", "gkd:experiment"))]
    assert len(consolidate(values)) == 2
    assert len(consolidate([*values, values[0].model_copy(update={"objective": "  阅读 GKD 分析方法  "})])) == 2


def test_budget_merge_keeps_coverage_and_all_scope_aliases():
    values = [SubQuestion(**question(v, v)) for v in ["原理", "实验", "成本"]]
    merged = within_budget(consolidate(values), 1)
    assert len(merged) == 1 and set(merged[0].covers) == {"原理", "实验", "成本"}
    previous = ResearchTask(task_id="T1", status="completed", **merged[0].model_dump())
    assert not novel_followups([SubQuestion(**question("换句话说再研究实验", "实验"))], [previous])


def test_uncovered_requirement_rejected_without_keyword_inference():
    with pytest.raises(ValidationError, match="遗漏"):
        ResearchPlan.model_validate({"requirements": ["原理", "局限"],
            "questions": [question("说明原理", "opd:principle", ["原理"])]})


@pytest.mark.parametrize("status", ["completed", "pending", "partial"])
def test_review_filters_existing_semantic_scope_but_allows_specific_new_gap(status):
    task = ResearchTask(task_id="T1", status=status, **question("解释 OPD 核心机制", "opd:principle"))
    duplicate = question("更详细调研 on-policy 蒸馏原理", "opd:principle", research_gap="尚未解释清楚分布差异")
    fresh = question("核实某实验在长序列上的训练成本", "opd:long-sequence-cost", research_gap="已有训练流程，但缺长序列成本，影响适用性判断")
    llm = ScriptedLLM({"follow_up_questions": [duplicate, fresh]})
    review = Supervisor(llm, 8).review(research_request(), [task], [])
    assert len(llm.calls) == 1 and [q.objective for q in review.follow_up_questions] == [fresh["objective"]]
    assert "尚未解释清楚分布差异" in review.remaining_gaps  # Filtering work does not claim the gap is solved.
    payload = json.loads(llm.calls[0]["messages"][-1]["content"])
    assert payload["questions"][0]["scope_keys"] == ["opd:principle"]


def test_review_requires_concrete_gap_for_new_model_followup():
    from sage_scholar.agents.supervisor import ReviewResponse
    with pytest.raises(ValidationError, match="research_gap"):
        ReviewResponse.model_validate({"follow_up_questions": [question("再写详细一点", "style")]})


def test_duplicate_review_does_not_dispatch_or_consume_extra_round():
    dispatched, reviews = [], []
    class Plan(FakeSupervisor):
        def plan(self, request):
            return [SubQuestion(**question("原理", "opd:principle"))]
        def review(self, request, tasks, notes):
            reviews.append(1)
            return SupervisorReview(follow_up_questions=[SubQuestion(**question("换说法再问原理", "opd:principle"))])
    class Researcher(FakeResearcher):
        def run(self, task, request):
            dispatched.append(task.task_id)
            return super().run(task, request)
    result = build_graph(Plan(), lambda _: Researcher(), FakeWriter()).invoke({"execution": research_request(max_rounds=4)})
    assert dispatched == ["T1"] and len(reviews) == 1
    assert result["final_report"] and not result["report_result"].partial


def test_checkpoint_keeps_scope_identity_and_continuation_reviews_gaps(tmp_path):
    calls, reviews = [], []
    class Plan(FakeSupervisor):
        def plan(self, request):
            return [SubQuestion(**question("OPD 机制", "opd:principle"))]
        def review(self, request, tasks, notes):
            reviews.append(1)
            return SupervisorReview(remaining_gaps=["训练成本未确认"])
    class Researcher(FakeResearcher):
        def run(self, task, request):
            calls.append(task.task_id)
            return super().run(task, request)
    request = research_request().model_copy(update={"run_id": "plan-resume"})
    cp = Checkpoints(tmp_path)
    config = cp.config(request.run_id)
    graph = build_graph(Plan(), lambda _: Researcher(), FakeWriter(), checkpointer=cp.saver)
    graph.invoke({"execution": request}, config)
    cp.close()
    cp = Checkpoints(tmp_path)
    graph = build_graph(Plan(), lambda _: Researcher(), FakeWriter(), checkpointer=cp.saver)
    state = graph.get_state(config)
    assert state.values["tasks"]["T1"].scope_keys == ["opd:principle"]
    request = request.model_copy(update={"metadata": {"continuation": 1}})
    continue_graph(graph, config, state, request)
    result = graph.invoke(None, config)
    assert calls == ["T1"] and len(reviews) == 2  # A gap string is not a new task.
    assert result["report_result"].partial and len(result["tasks"]) == 1
    cp.close()
    # Old checkpoints without planning metadata still validate.
    assert ResearchTask.model_validate({"task_id": "old", "objective": "旧任务"}).scope_keys == []


def test_partial_scope_overlap_does_not_discard_new_requirement():
    existing = ResearchTask(task_id="T1", status="completed", **question("OPD 原理", "principle"))
    followup = SubQuestion(**{**question("比较原理并核对新的成本数据", "cost"),
                             "scope_keys": ["principle", "cost"]})
    assert novel_followups([followup], [existing])[0].scope_keys == ["principle", "cost"]
