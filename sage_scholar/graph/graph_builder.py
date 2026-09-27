"""The existing LangGraph plan/parallel research/review/write workflow, with typed boundaries."""
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.types import Send

from sage_scholar.domain.research import (
    Claim, ClaimVerification, GapReview, ReportDraft, ResearchContract, ResearchNote,
    ResearchReport, ResearchTask, VerificationResult,
)
from sage_scholar.research.renderer import CitationRenderer
from sage_scholar.research.tasks import follow_up_tasks, validate_tasks


def merge_dicts(a, b):
    return {**(a or {}), **(b or {})}


class State(TypedDict):
    research_contract: ResearchContract
    tasks: Annotated[dict[str, ResearchTask], merge_dicts]
    research_notes: Annotated[dict[str, ResearchNote | None], merge_dicts]
    ready_tasks: list[str]
    evidence_ids: list[str]
    gap_review: GapReview
    refine_round: int
    draft: ReportDraft
    verification_result: VerificationResult
    verification_history: list[VerificationResult]
    removed_during_revision: list[str]
    writer_revision: int
    draft_gaps: list[str]
    task_gaps: list[str]
    planning_exhausted: bool
    report_result: ResearchReport
    final_report: str


class InputSchema(TypedDict):
    research_contract: ResearchContract


class OutputSchema(TypedDict):
    final_report: str
    report_result: ResearchReport
    research_notes: dict[str, ResearchNote | None]
    tasks: dict[str, ResearchTask]


def build_graph(supervisor, create_researcher, writer, *, gap_reviewer, verifier, budget=None):
    builder = StateGraph(State, input_schema=InputSchema, output_schema=OutputSchema)

    def notes(state):
        return [note for note in state.get("research_notes", {}).values() if note is not None]

    def plan_node(state):
        contract = state["research_contract"]
        previous = state.get("tasks", {})
        try:
            if previous:
                tasks, changed = follow_up_tasks(
                    previous, state["gap_review"].next_tasks,
                    state.get("refine_round", 0), contract.limits.max_tasks,
                )
                return {"tasks": tasks, "research_notes": {identity: None for identity in changed},
                        "planning_exhausted": not changed,
                        "task_gaps": [] if changed else ["任务数量预算已耗尽，无法新增补查任务"]}
            planned = supervisor.plan_tasks(contract)
            tasks = validate_tasks(planned)
            if len(tasks) > contract.limits.max_tasks:
                # Dropping dependent tasks silently would corrupt the plan.
                raise ValueError("计划超过 max_tasks 限制")
            return {"tasks": tasks, "task_gaps": [] if tasks else ["Planner 未产生任务"]}
        except Exception as exc:
            return {"tasks": {}, "task_gaps": [f"任务规划失败：{exc}"], "planning_exhausted": True}

    def dispatch_node(state):
        tasks = state.get("tasks", {})
        completed = {key for key, task in tasks.items() if task.status == "completed"}
        ready = [key for key, task in tasks.items()
                 if task.status == "pending" and set(task.dependencies) <= completed]
        if budget and budget.exhausted:
            ready = []
        updates = {}
        if not ready:
            for key, task in tasks.items():
                if task.status == "pending":
                    updates[key] = task.model_copy(update={"status": "blocked"})
        return {"ready_tasks": ready, "tasks": updates}

    def dispatch_route(state):
        if not state["ready_tasks"]:
            return "review_node"
        return [Send("research_node", {
            "task": state["tasks"][identity], "research_contract": state["research_contract"],
            "dependency_notes": [
                note for key, note in state.get("research_notes", {}).items()
                if key in state["tasks"][identity].dependencies and note is not None
            ],
        }) for identity in state["ready_tasks"]]

    def research_node(state):
        task = state["task"]
        try:
            researcher = create_researcher(task.task_id)
            note = researcher.run(task, state["research_contract"], state["dependency_notes"])
            if not isinstance(note, ResearchNote):
                raise TypeError("Researcher 必须返回 ResearchNote")
            if note.task_id != task.task_id:
                raise ValueError("ResearchNote task_id 不匹配")
            status = "completed"
        except Exception as exc:
            note = ResearchNote(task_id=task.task_id, question=task.objective, revision=task.revision,
                                unknowns=[f"任务失败：{exc}"], metadata={"task_error": str(exc)})
            status = "failed"
        return {"research_notes": {task.task_id: note},
                "tasks": {task.task_id: task.model_copy(update={"status": status})}}

    def review_node(state):
        contract = state["research_contract"]
        current_notes = notes(state)
        review = gap_reviewer.review(contract, list(state.get("tasks", {}).values()), current_notes)
        review.missing.extend(state.get("task_gaps", []))
        if budget and budget.exhausted:
            review.missing.append("工具调用预算已耗尽")
        return {
            "gap_review": review, "refine_round": state.get("refine_round", 0) + 1,
            "evidence_ids": sorted({identity for note in current_notes for identity in note.evidence_ids}),
        }

    def reviewer_route(state):
        review, contract = state["gap_review"], state["research_contract"]
        if (review.gaps and review.next_tasks and state["refine_round"] <= contract.limits.max_rounds
                and not state.get("planning_exhausted", False)
                and not (budget and budget.exhausted)):
            return "plan_node"
        return "write_node"

    def write_node(state):
        get_stream_writer()({"progress_node": {"event": "report_drafting", "phase": "started"}})
        previous = state.get("draft")
        draft_gaps = list(state.get("draft_gaps", []))
        try:
            draft = writer.run(state["research_contract"], notes(state),
                               previous=previous, verification=state.get("verification_result"))
        except Exception as exc:
            draft_gaps.append(f"Writer 生成或修订失败：{exc}")
            # Preserve the last draft; otherwise verify only actual note findings.
            draft = previous or ReportDraft(title="研究报告", claims=[
                Claim(claim_id=f"C{index}", text=finding.text, evidence_ids=finding.evidence_ids)
                for index, finding in enumerate(
                    [finding for note in notes(state) for finding in note.findings], 1,
                )
            ])
        removed = list(state.get("removed_during_revision", []))
        if previous and state.get("verification_result"):
            new_ids = {claim.claim_id for claim in draft.claims}
            removed.extend(item.claim_id for item in state["verification_result"].unresolved
                           if item.claim_id not in new_ids)
        return {"draft": draft, "draft_gaps": draft_gaps,
                "removed_during_revision": list(dict.fromkeys(removed))}

    def verify_node(state):
        get_stream_writer()({"progress_node": {"event": "citation_verifying", "phase": "started"}})
        try:
            verification = verifier.verify(state["draft"])
        except Exception as exc:
            verification = VerificationResult(claims=[
                ClaimVerification(claim_id=claim.claim_id, status="uncertain", reason=str(exc),
                                  identity_valid=False, evidence_ids=claim.evidence_ids)
                for claim in state["draft"].claims
            ], issues=[f"引用核验失败：{exc}"])
        return {"verification_result": verification,
                "verification_history": [*state.get("verification_history", []), verification]}

    def verification_route(state):
        if (state["verification_result"].unresolved
                and state.get("writer_revision", 0) < state["research_contract"].limits.max_writer_revisions):
            return "revision_node"
        return "render_node"

    def revision_node(state):
        return {"writer_revision": state.get("writer_revision", 0) + 1}

    def render_node(state):
        gaps = [*state["gap_review"].gaps, *state.get("draft_gaps", [])]
        if state["gap_review"].gaps and state["refine_round"] > state["research_contract"].limits.max_rounds:
            gaps.append("已达到研究补查轮数上限")
        if state["verification_result"].unresolved:
            gaps.append("已达到报告修订上限，剩余未充分支持的主张不纳入正文")
        removed = state.get("removed_during_revision", [])
        gaps.extend(f"修订中移除未充分支持的主张 {identity}" for identity in removed)
        report = CitationRenderer().render(state["draft"], state["verification_result"], remaining_gaps=gaps)
        report.removed_claim_ids = list(dict.fromkeys([*removed, *report.removed_claim_ids]))
        report.verification_history = state.get("verification_history", [])
        return {"report_result": report, "final_report": report.report}

    for name, node in [
        ("plan_node", plan_node), ("dispatch_node", dispatch_node), ("research_node", research_node),
        ("review_node", review_node), ("write_node", write_node), ("verify_node", verify_node),
        ("revision_node", revision_node), ("render_node", render_node),
    ]:
        builder.add_node(name, node)
    builder.add_edge(START, "plan_node")
    builder.add_edge("plan_node", "dispatch_node")
    builder.add_conditional_edges("dispatch_node", dispatch_route, ["research_node", "review_node"])
    builder.add_edge("research_node", "dispatch_node")
    builder.add_conditional_edges("review_node", reviewer_route, ["plan_node", "write_node"])
    builder.add_edge("write_node", "verify_node")
    builder.add_conditional_edges("verify_node", verification_route, ["revision_node", "render_node"])
    builder.add_edge("revision_node", "write_node")
    builder.add_edge("render_node", END)
    return builder.compile()
