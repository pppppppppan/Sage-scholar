from sage_scholar.domain.research import Finding, FollowUp, GapReview, ResearchNote, ResearchTask
from sage_scholar.research.gap_review import GapReviewer
from research_fakes import ScriptedLLM, contract
from test_research_evidence import evidence_context


def setup_note(context):
    ids = context.ids()
    task = ResearchTask(task_id="T1", objective="compare methods", status="completed", dimensions=["methods"])
    note = ResearchNote(task_id="T1", question=task.objective, evidence_ids=ids,
        findings=[Finding(text="method result", evidence_ids=ids, dimensions=["methods"])])
    return task, note


def test_missing_dimension_causes_targeted_followup(evidence_context):
    task, note = setup_note(evidence_context)
    request = contract().model_copy(update={"requested_dimensions": ["methods", "ablation"]})
    review = GapReviewer(ScriptedLLM(GapReview(covered=["methods", "ablation"], next_tasks=[FollowUp(objective="读取消融实验", dimensions=["ablation"])])), evidence_context)
    result = review.review(request, [task], [note])
    assert result.covered == ["methods"]
    assert "尚需补充或解释：ablation" in result.missing
    assert any("ablation" in item.dimensions for item in result.next_tasks)


def test_all_dimensions_supported_does_not_replan(evidence_context):
    task, note = setup_note(evidence_context)
    request = contract().model_copy(update={"requested_dimensions": ["methods"]})
    result = GapReviewer(ScriptedLLM(GapReview(covered=["methods"])), evidence_context).review(request, [task], [note])
    assert result.gaps == result.next_tasks == []


def test_unknown_conflict_and_missing_evidence_target_original_task(evidence_context):
    task, note = setup_note(evidence_context)
    note.unknowns, note.conflicts = ["preprocessing"], ["split mismatch"]
    note.findings[0].evidence_ids = []
    result = GapReviewer(ScriptedLLM(GapReview()), evidence_context).review(contract(), [task], [note])
    assert result.unknowns == note.unknowns and result.conflicts == note.conflicts
    assert any("缺少可追溯" in gap for gap in result.missing)
    assert result.next_tasks == []  # Diagnostics alone must not force repeated research.


def test_missing_task_note_cannot_be_approved(evidence_context):
    task, _ = setup_note(evidence_context)
    result = GapReviewer(ScriptedLLM(GapReview()), evidence_context).review(contract(), [task], [])
    assert result.missing and result.next_tasks[0].task_id == "T1"


def test_contract_preserves_execution_and_limits():
    original = contract(max_rounds=2, max_tool_calls=7)
    assert original.execution.allow_web == original.source_policy.allow_web
    assert original.execution.execution_mode.value == "run"
    assert original.limits.max_tool_calls == 7 and original.limits.max_rounds == 2
