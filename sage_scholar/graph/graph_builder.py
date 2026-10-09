"""Durable research → synthesis → section writing → local report review."""
import logging
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.types import Send
from langgraph.errors import GraphBubbleUp
from sage_scholar.context.manager import ContextBudgetExceeded
from sage_scholar.runtime.journal import WorkflowInterrupted
from sage_scholar.runtime.control import CURRENT_CONTROL, ResearchBudgetReached, check_control
from sage_scholar.domain.research import (
    GapReview, ResearchContract, ResearchNote, ResearchReport, ResearchTask,
    Synthesis, ReportOutline, WrittenSection, ReportReview,
)
from sage_scholar.research.tasks import follow_up_tasks, validate_tasks
from sage_scholar.research.materials import batches, note_units, encoded
from sage_scholar.research.note_view import sanitize_note
from sage_scholar.research.references import reference_catalog, check_section, finalize_report
from sage_scholar.context.token_counter import TokenCounter

logger = logging.getLogger(__name__)


def merge_dicts(a, b):
    return {**(a or {}), **(b or {})}


class State(TypedDict):
    research_contract: ResearchContract
    tasks: Annotated[dict[str, ResearchTask], merge_dicts]
    research_notes: Annotated[dict[str, ResearchNote], merge_dicts]
    ready_tasks: list[str]
    evidence_ids: list[str]
    gap_review: GapReview
    refine_round: int
    task_gaps: list[str]
    planning_exhausted: bool
    blocked_by_budget: bool
    reviewed_evidence_ids: list[str]
    reviewed_dimensions: list[str]
    human_question: dict | None
    human_answer: dict | None
    continuation_generation: int
    resume_report: bool
    synthesis_queue: list[list[dict]]
    synthesis_parts: list[Synthesis]
    synthesis_level: int
    synthesis: Synthesis
    outline: ReportOutline
    references: dict[int, str]
    sections: dict[str, WrittenSection]
    section_reviews: dict[str, ReportReview]
    section_index: int
    review_index: int
    section_failures: list[str]
    report_followup_count: int
    final_report: str
    report_result: ResearchReport


class InputSchema(TypedDict):
    research_contract: ResearchContract


class OutputSchema(TypedDict):
    final_report: str
    report_result: ResearchReport
    research_notes: dict[str, ResearchNote]
    tasks: dict[str, ResearchTask]
    blocked_by_budget: bool


def build_graph(supervisor, create_researcher, writer, *, gap_reviewer, synthesizer,
                report_reviewer, budget=None, checkpointer=None, interaction=None):
    builder = StateGraph(State, input_schema=InputSchema, output_schema=OutputSchema)
    context = writer.context

    def progress(event, **data):
        get_stream_writer()({"progress_node": {"event": event, **data}})

    def notes(state):
        return [sanitize_note(n) for n in state.get("research_notes", {}).values() if n is not None]

    def assess_node(state):
        return {"human_question": interaction(state["research_contract"].execution) if interaction else None}

    def human_node(state):
        from sage_scholar.runtime.interaction import ask
        answer = ask(state.get("human_question"))
        if not answer:
            return {}
        contract = state["research_contract"]
        clarification = "用户补充决定：" + (answer.get("answer") or "跳过问题，由研究员自主确定")
        execution = contract.execution.model_copy(update={"query": contract.execution.query + "\n" + clarification})
        return {"human_answer": answer, "research_contract": contract.model_copy(update={
            "execution": execution, "objective": contract.objective + "\n" + clarification,
            "constraints": [*contract.constraints, clarification]})}

    def continuation_node(state):
        return {}

    def continuation_route(state):
        if state.get("resume_report"):
            return "write_node"
        if any(t.status == "pending" for t in state.get("tasks", {}).values()):
            return "dispatch_node"
        return "plan_node" if state.get("gap_review") and state["gap_review"].next_tasks else "review_node"

    def plan_node(state):
        contract, previous = state["research_contract"], state.get("tasks", {})
        if previous:
            tasks, changed = follow_up_tasks(previous, state["gap_review"].next_tasks,
                state.get("refine_round", 0), contract.limits.max_tasks)
            return {"tasks": tasks, "planning_exhausted": not changed,
                    "task_gaps": [] if changed else ["仍有研究问题未能补齐"]}
        try:
            tasks = validate_tasks(supervisor.plan_tasks(contract))
            if len(tasks) > contract.limits.max_tasks:
                raise ValueError("计划超过任务数量上限")
            dimensions = list(dict.fromkeys([*contract.requested_dimensions,
                                             *(d for t in tasks.values() for d in t.dimensions)]))
            return {"tasks": tasks, "research_contract": contract.model_copy(update={"requested_dimensions": dimensions}),
                    "task_gaps": [] if tasks else ["未形成有效研究计划"]}
        except (WorkflowInterrupted, ResearchBudgetReached, ContextBudgetExceeded, GraphBubbleUp):
            raise
        except Exception:
            logger.exception("Research planning failed")
            raise ValueError("研究计划生成失败，请重试；未生成报告") from None

    def dispatch_node(state):
        tasks = state.get("tasks", {})
        completed = {k for k, t in tasks.items() if t.status == "completed"}
        ready = [k for k, t in tasks.items() if t.status == "pending" and set(t.dependencies) <= completed]
        control = CURRENT_CONTROL.get()
        if control and control.research_exhausted():
            ready = []
        if ready and budget and budget.exhausted:
            budget.mark_blocked()
            ready = []
        updates = {k: t.model_copy(update={"status": "blocked"}) for k, t in tasks.items()
                   if not ready and t.status == "pending"}
        return {"ready_tasks": ready, "tasks": updates}

    def dispatch_route(state):
        if not state["ready_tasks"]:
            return "review_node"
        return [Send("research_node", {"task": state["tasks"][identity],
            "research_contract": state["research_contract"],
            "dependency_notes": [n for k, n in state.get("research_notes", {}).items()
                                 if n and (k == identity or k in state["tasks"][identity].dependencies)],
        }) for identity in state["ready_tasks"]]

    def research_node(state):
        task = state["task"]
        try:
            note = create_researcher(task.task_id).run(task, state["research_contract"], state["dependency_notes"])
            if not isinstance(note, ResearchNote) or note.task_id != task.task_id:
                raise ValueError("研究笔记身份不匹配")
            # Follow-up findings supplement existing work instead of erasing whole subtopics.
            prior = next((n for n in state["dependency_notes"] if n.task_id == task.task_id), None)
            if prior:
                seen = {f.text for f in note.findings}
                note.findings.extend(f for f in prior.findings if f.text not in seen)
                note.evidence_ids = list(dict.fromkeys([*note.evidence_ids, *prior.evidence_ids]))
            status = "completed"
        except (WorkflowInterrupted, ResearchBudgetReached, ContextBudgetExceeded, GraphBubbleUp):
            raise
        except Exception:
            logger.exception("Research task %s failed", task.task_id)
            note = ResearchNote(task_id=task.task_id, question=task.objective, revision=task.revision,
                                unknowns=["此研究问题暂未取得足够可靠资料"], metadata={"task_error": True})
            status = "failed"
        return {"research_notes": {task.task_id: sanitize_note(note)},
                "tasks": {task.task_id: task.model_copy(update={"status": status})}}

    def review_node(state):
        review = gap_reviewer.review(state["research_contract"], list(state.get("tasks", {}).values()), notes(state))
        review.missing.extend(state.get("task_gaps", []))
        for task in state.get("tasks", {}).values():
            if task.status in {"blocked", "failed"}:
                review.missing.append(f"尚未完成：{task.objective}")
        identities = sorted({i for n in notes(state) for i in n.evidence_ids})
        control = CURRENT_CONTROL.get()
        blocked = bool(budget and budget.blocked_by_budget)
        if (control and control.research_exhausted()) or (budget and budget.exhausted):
            if review.next_tasks:
                review.missing.append("受本次研究资源限制，部分问题仍需进一步调查")
                if budget and budget.exhausted:
                    budget.mark_blocked()
                    blocked = True
            review.next_tasks = []
        elif state.get("refine_round", 0) and not (set(identities) - set(state.get("reviewed_evidence_ids", []))):
            review.next_tasks = []
        return {"gap_review": review, "refine_round": state.get("refine_round", 0) + 1,
                "evidence_ids": identities, "reviewed_evidence_ids": identities,
                "reviewed_dimensions": review.covered, "blocked_by_budget": blocked}

    def review_route(state):
        if (state["gap_review"].next_tasks and state["refine_round"] <= state["research_contract"].limits.max_rounds
                and not state.get("planning_exhausted")):
            return "plan_node"
        return "prepare_synthesis_node"

    def prepare_synthesis_node(state):
        queue = list(batches(note_units(notes(state))))
        if not state.get("evidence_ids") or not queue:
            raise ValueError("未取得可用原文证据，无法生成可信研究报告")
        return {"synthesis_queue": queue, "synthesis_parts": [], "synthesis_level": 0,
                "sections": {}, "section_reviews": {}, "section_index": 0, "review_index": 0,
                "section_failures": []}

    def synthesis_node(state):
        import re
        progress("research_synthesizing", phase="started")
        materials = state["synthesis_queue"][0]
        allowed = set(state["evidence_ids"])
        ids = [i for i in dict.fromkeys(re.findall(r"ev_[a-zA-Z0-9_-]+", encoded(materials))) if i in allowed]
        part = synthesizer.run(state["research_contract"], materials, allowed_ids=ids)
        return {"synthesis_queue": state["synthesis_queue"][1:],
                "synthesis_parts": [*state["synthesis_parts"], part]}

    def synthesis_route(state):
        if state["synthesis_queue"]:
            return "synthesis_node"
        return "combine_synthesis_node"

    def combine_synthesis_node(state):
        blocks = [b for p in state["synthesis_parts"] for b in p.blocks]
        if len(state["synthesis_parts"]) == 1 and len(blocks) <= 8 and TokenCounter().count(encoded([b.model_dump() for b in blocks])) <= 6500:
            return {"synthesis": Synthesis(blocks=blocks), "synthesis_queue": []}
        if state["synthesis_level"] >= 3:
            raise ValueError("研究综合未能在预算内收敛，材料已保存，请缩小研究范围")
        return {"synthesis_queue": list(batches([b.model_dump() for b in blocks])),
                "synthesis_parts": [], "synthesis_level": state["synthesis_level"] + 1}

    def combine_route(state):
        return "synthesis_node" if state["synthesis_queue"] else "outline_node"

    def outline_node(state):
        outline = writer.outline(state["research_contract"], state["synthesis"])
        return {"outline": outline, "references": reference_catalog(context,
                [i for b in state["synthesis"].blocks for i in b.evidence_ids])}

    def write_node(state):
        plan = state["outline"].sections[state["section_index"]]
        if state["section_reviews"].get(plan.section_id) and state["section_reviews"][plan.section_id].acceptable:
            return {"section_index": state["section_index"] + 1, "review_index": state["section_index"]}
        control = CURRENT_CONTROL.get()
        if control and control.store and any(r.acceptable for r in state["section_reviews"].values()):
            available = control.request.budget["max_total_tokens"] - control.store.model_load(control.request.run_id)
            if available < 70_000:
                unfinished = state["outline"].sections[state["section_index"]:]
                return {"section_index": len(state["outline"].sections), "review_index": len(state["outline"].sections),
                        "section_failures": [*state["section_failures"], *[f"本次预算不足，尚未完成：{p.heading}" for p in unfinished]]}
        plan = state["outline"].sections[state["section_index"]]
        progress("report_drafting", phase="started", section=plan.heading)
        sections, failures = dict(state["sections"]), list(state["section_failures"])
        try:
            section = writer.write_section(state["research_contract"], state["synthesis"], state["outline"], plan, state["references"])
            sections[plan.section_id] = section
        except (WorkflowInterrupted, ResearchBudgetReached, ContextBudgetExceeded, GraphBubbleUp):
            raise
        except Exception:
            logger.exception("Writing section %s failed", plan.section_id)
            failures.append(f"{plan.heading}尚未完成写作")
        return {"sections": sections, "section_failures": failures, "section_index": state["section_index"] + 1,
                "review_index": state["section_index"]}

    def write_route(state):
        if state["review_index"] >= len(state["outline"].sections):
            return "finalize_node"
        key = state["outline"].sections[state["review_index"]].section_id
        if state["section_reviews"].get(key) and state["section_reviews"][key].acceptable:
            return "advance_review_node"
        return "report_review_node"

    def report_review_node(state):
        plan = state["outline"].sections[state["review_index"]]
        progress("report_reviewing", phase="started", section=plan.heading)
        section = state["sections"].get(plan.section_id)
        reviews = dict(state["section_reviews"])
        if section:
            ids = {i for index in plan.block_indices for i in state["synthesis"].blocks[index].evidence_ids}
            issues = check_section(section, state["references"], context, ids)
            try:
                review = (ReportReview(acceptable=False, issues=issues, revision_instructions="；".join(issues))
                          if issues else report_reviewer.review(state["research_contract"], state["outline"],
                              section, state["references"], state["sections"]))
            except (WorkflowInterrupted, ResearchBudgetReached, ContextBudgetExceeded, GraphBubbleUp):
                raise
            except Exception:
                logger.exception("Report review failed")
                review = ReportReview(acceptable=False, issues=["本节尚未完成质量审查"])
            reviews[plan.section_id] = review
        return {"section_reviews": reviews}

    def report_review_route(state):
        key = state["outline"].sections[state["review_index"]].section_id
        section, review = state["sections"].get(key), state["section_reviews"].get(key)
        control = CURRENT_CONTROL.get()
        if (section and review and review.source_request and state.get("report_followup_count", 0) < 1
                and len(state["synthesis"].blocks) < 8
                and section.revision < state["research_contract"].limits.max_writer_revisions
                and not (budget and budget.exhausted) and not (control and control.research_exhausted())):
            return "request_section_sources_node"
        if section and review and not review.acceptable and section.revision < state["research_contract"].limits.max_writer_revisions:
            return "revise_section_node"
        return "advance_review_node"

    def request_section_sources_node(state):
        # Commit intent/count before external work; at most one report-level follow-up per Run.
        return {"report_followup_count": state.get("report_followup_count", 0) + 1}

    def section_sources_node(state):
        progress("research_started", phase="supplementing", message="正在定向补读关键原文…")
        plan = state["outline"].sections[state["review_index"]]
        review = state["section_reviews"][plan.section_id]
        task = ResearchTask(task_id=f"report-{plan.section_id}", objective=review.source_request,
                            label=plan.heading, revision=state["report_followup_count"])
        return research_node({"task": task, "research_contract": state["research_contract"],
                              "dependency_notes": []})

    def section_synthesis_node(state):
        plan = state["outline"].sections[state["review_index"]]
        note = state["research_notes"][f"report-{plan.section_id}"]
        if not note.evidence_ids:
            return {}  # No invented support: local revision must narrow the conclusion.
        blocks = [state["synthesis"].blocks[i] for i in plan.block_indices]
        identities = list(dict.fromkeys([*note.evidence_ids, *(i for b in blocks for i in b.evidence_ids)]))
        merged = synthesizer.run(state["research_contract"],
            [*[b.model_dump() for b in blocks], *note_units([note])],
            allowed_ids=identities, max_blocks=1)
        synthesis = Synthesis(blocks=[*state["synthesis"].blocks, *merged.blocks])
        outline = state["outline"].model_copy(deep=True)
        outline.sections[state["review_index"]].block_indices = [len(synthesis.blocks) - 1]
        # Existing chapter numbers stay stable when adding newly read evidence.
        references = dict(state["references"])
        for identity in identities:
            context.get(identity)
            if identity not in references.values():
                references[len(references) + 1] = identity
        return {"synthesis": synthesis, "outline": outline, "references": references,
                "evidence_ids": list(dict.fromkeys([*state["evidence_ids"], *identities]))}

    def revise_section_node(state):
        plan = state["outline"].sections[state["review_index"]]
        progress("report_revising", phase="started", section=plan.heading)
        previous = state["sections"][plan.section_id]
        sections = dict(state["sections"])
        try:
            sections[plan.section_id] = writer.write_section(state["research_contract"], state["synthesis"],
                state["outline"], plan, state["references"], previous=previous,
                review=state["section_reviews"][plan.section_id])
        except (WorkflowInterrupted, ResearchBudgetReached, ContextBudgetExceeded, GraphBubbleUp):
            raise
        except Exception:
            logger.exception("Section revision failed")
            sections[plan.section_id] = previous.model_copy(update={"revision": previous.revision + 1})
        return {"sections": sections}

    def advance_review_node(state):
        return {"review_index": state["review_index"] + 1}

    def advance_review_route(state):
        return "write_node" if state["section_index"] < len(state["outline"].sections) else "finalize_node"

    def finalize_node(state):
        result = finalize_report(state["outline"], state["sections"], state["references"], context,
            state["section_reviews"], [*state["gap_review"].gaps, *state["section_failures"]])
        return {"report_result": result, "final_report": result.report}

    nodes = {key: value for key, value in locals().copy().items() if key.endswith("_node")}
    for name, fn in nodes.items():
        def checked(state, fn=fn):
            check_control()
            return fn(state)
        builder.add_node(name, checked)
    for a, b in [(START, "assess_node"), ("assess_node", "human_node"), ("human_node", "plan_node"),
                 ("plan_node", "dispatch_node"), ("research_node", "dispatch_node"),
                 ("prepare_synthesis_node", "synthesis_node"), ("outline_node", "write_node"),
                 ("request_section_sources_node", "section_sources_node"),
                 ("section_sources_node", "section_synthesis_node"),
                 ("section_synthesis_node", "revise_section_node"),
                 ("revise_section_node", "report_review_node"), ("finalize_node", END)]:
        builder.add_edge(a, b)
    for name, route, targets in [
        ("continuation_node", continuation_route, ["dispatch_node", "plan_node", "review_node", "write_node"]),
        ("dispatch_node", dispatch_route, ["research_node", "review_node"]),
        ("review_node", review_route, ["plan_node", "prepare_synthesis_node"]),
        ("synthesis_node", synthesis_route, ["synthesis_node", "combine_synthesis_node"]),
        ("combine_synthesis_node", combine_route, ["synthesis_node", "outline_node"]),
        ("write_node", write_route, ["finalize_node", "report_review_node", "advance_review_node"]),
        ("report_review_node", report_review_route, ["request_section_sources_node", "revise_section_node", "advance_review_node"]),
        ("advance_review_node", advance_review_route, ["write_node", "finalize_node"]),
    ]:
        builder.add_conditional_edges(name, route, targets)
    return builder.compile(checkpointer=checkpointer)
