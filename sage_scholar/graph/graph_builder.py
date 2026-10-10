"""Supervisor → parallel researchers → supervisor review → whole-report writer."""
import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.types import Send
from langgraph.errors import GraphBubbleUp
from sage_scholar.context.manager import ContextBudgetExceeded
from sage_scholar.context.manager import request_context
from sage_scholar.runtime.journal import WorkflowInterrupted
from sage_scholar.runtime.control import CURRENT_CONTROL, ResearchBudgetReached, check_control
from sage_scholar.domain.execution import ExecutionRequest
from sage_scholar.domain.research import ResearchNote, ResearchReport, ResearchTask, SupervisorReview

logger = logging.getLogger(__name__)


def merge_dicts(a, b):
    return {**(a or {}), **(b or {})}


class State(TypedDict):
    execution: ExecutionRequest
    tasks: Annotated[dict[str, ResearchTask], merge_dicts]
    research_notes: Annotated[dict[str, ResearchNote], merge_dicts]
    ready_tasks: list[str]
    supervisor_review: SupervisorReview
    review_round: int
    round_base: int
    planning_exhausted: bool
    blocked_by_budget: bool
    human_question: dict | None
    human_answer: dict | None
    continuation_generation: int
    report_result: ResearchReport
    final_report: str


class InputSchema(TypedDict):
    execution: ExecutionRequest


class OutputSchema(TypedDict):
    final_report: str
    report_result: ResearchReport
    research_notes: dict[str, ResearchNote]
    tasks: dict[str, ResearchTask]
    blocked_by_budget: bool


def build_graph(supervisor, create_researcher, writer, *, budget=None, checkpointer=None, interaction=None, execution_request=None):
    builder = StateGraph(State, input_schema=InputSchema, output_schema=OutputSchema)

    def notes(state):
        return [n for n in state.get("research_notes", {}).values() if n is not None]

    def exhausted():
        control = CURRENT_CONTROL.get()
        return bool((control and control.research_exhausted()) or (budget and budget.exhausted))

    def assess_node(state):
        return {"human_question": interaction(state["execution"]) if interaction else None}

    def human_node(state):
        from sage_scholar.runtime.interaction import ask
        answer = ask(state.get("human_question"))
        if not answer:
            return {}
        request = state["execution"]
        clarification = "用户补充决定：" + (answer.get("answer") or "跳过问题，由研究员自主确定")
        metadata = {**request.metadata, "constraints": [*request.metadata.get("constraints", []), clarification]}
        return {"human_answer": answer, "execution": request.model_copy(update={
            "query": request.query + "\n" + clarification, "metadata": metadata})}

    def continuation_node(state):
        return {}

    def continuation_route(state):
        if any(t.status == "pending" for t in state.get("tasks", {}).values()):
            return "dispatch_node"
        if state.get("supervisor_review") and state["supervisor_review"].follow_up_questions:
            return "plan_node"
        return "review_node"

    def plan_node(state):
        request, previous = state["execution"], state.get("tasks", {})
        questions = (state["supervisor_review"].follow_up_questions if previous else supervisor.plan(request))
        seen = {t.objective.strip().casefold() for t in previous.values()}
        available = max(0, request.budget["max_tasks"] - len(previous))
        tasks = {}
        for question in questions:
            objective = question.objective.strip()
            if not objective or objective.casefold() in seen or len(tasks) >= available:
                continue
            identity = f"T{len(previous) + len(tasks) + 1}"
            tasks[identity] = ResearchTask(task_id=identity, objective=objective, label=question.label)
            seen.add(objective.casefold())
        if not tasks and not previous:
            raise ValueError("未形成有效研究子问题，请重试")
        return {"tasks": tasks, "planning_exhausted": not tasks}

    def dispatch_node(state):
        pending = [t for t in state.get("tasks", {}).values() if t.status == "pending"]
        blocked = bool(pending and exhausted())
        if blocked and budget:
            budget.mark_blocked()
        return {"ready_tasks": [] if blocked else [t.task_id for t in pending],
                "tasks": {t.task_id: t.model_copy(update={"status": "blocked"}) for t in pending} if blocked else {},
                "blocked_by_budget": state.get("blocked_by_budget", False) or blocked}

    def dispatch_route(state):
        if not state["ready_tasks"]:
            return "review_node"
        return [Send("research_node", {"task": state["tasks"][identity], "execution": state["execution"]})
                for identity in state["ready_tasks"]]

    def research_node(state):
        task = state["task"]
        try:
            note = create_researcher(task.task_id).run(task, state["execution"])
            if not isinstance(note, ResearchNote) or note.task_id != task.task_id:
                raise ValueError("研究笔记身份不匹配")
            status = note.status
        except (WorkflowInterrupted, ResearchBudgetReached, ContextBudgetExceeded, GraphBubbleUp):
            raise
        except Exception:
            logger.exception("Research task %s failed", task.task_id)
            note = ResearchNote(task_id=task.task_id, question=task.objective, content="此问题未能完成资料收集。", status="failed")
            status = "failed"
        return {"research_notes": {task.task_id: note},
                "tasks": {task.task_id: task.model_copy(update={"status": status})}}

    def review_node(state):
        get_stream_writer()({"progress_node": {"event": "research_reviewing", "phase": "started"}})
        tasks = list(state.get("tasks", {}).values())
        try:
            review = supervisor.review(state["execution"], tasks, notes(state))
        except (WorkflowInterrupted, ResearchBudgetReached, ContextBudgetExceeded, GraphBubbleUp):
            raise
        except Exception:
            logger.exception("Supervisor review failed; preserve notes for writing")
            review = SupervisorReview(remaining_gaps=["研究主管未能完成覆盖审查，报告范围仍需确认"])
        # Failed collection is visible, but ordinary literature limitations never automatically become backlog.
        for task in tasks:
            if task.status in {"failed", "blocked"}:
                review.remaining_gaps.append(f"尚未完成：{task.objective}")
        for question in review.follow_up_questions:
            if question.objective not in review.remaining_gaps:
                review.remaining_gaps.append(question.objective)
        review.remaining_gaps = list(dict.fromkeys(review.remaining_gaps))
        blocked = bool(review.follow_up_questions and exhausted())
        if blocked and budget:
            budget.mark_blocked()
        return {"supervisor_review": review, "review_round": state.get("review_round", 0) + 1,
                "blocked_by_budget": state.get("blocked_by_budget", False) or blocked}

    def review_route(state):
        request, review = state["execution"], state["supervisor_review"]
        local_round = state["review_round"] - state.get("round_base", 0)
        if (review.follow_up_questions and local_round <= request.budget["max_rounds"]
                and len(state["tasks"]) < request.budget["max_tasks"]
                and not state.get("planning_exhausted") and not exhausted()):
            return "plan_node"
        return "write_node"

    def write_node(state):
        get_stream_writer()({"progress_node": {"event": "report_drafting", "phase": "started"}})
        usable = [n for n in notes(state) if n.status != "failed" and n.content.strip()]
        if not any(n.source_ids for n in usable):
            raise ValueError("未取得可回读的来源，无法生成有来源的研究报告；已有记录已保存")
        result = writer.run(state["execution"], usable, state["supervisor_review"].remaining_gaps)
        return {"report_result": result, "final_report": result.report}

    def finalize_node(state):
        # No model calls or a second review. This checkpoint precedes publishing the report.
        return {"report_result": state["report_result"], "final_report": state["final_report"]}

    for name, fn in list(locals().items()):
        if not name.endswith("_node"):
            continue
        def checked(state, fn=fn):
            check_control()
            if execution_request is not None:
                original = state["execution"]
                state = {**state, "execution": original.model_copy(update={
                    "budget": execution_request.budget,
                    "metadata": {**execution_request.metadata, "constraints": original.metadata.get("constraints", [])},
                })}
            with request_context(state["execution"]):
                result = fn(state)
            # Parallel research branches only merge their note/task. A serial node persists updated limits.
            if fn is not research_node:
                result.setdefault("execution", state["execution"])
            return result
        builder.add_node(name, checked)
    for a, b in [(START, "assess_node"), ("assess_node", "human_node"), ("human_node", "plan_node"),
                 ("plan_node", "dispatch_node"), ("research_node", "review_node"),
                 ("write_node", "finalize_node"), ("finalize_node", END)]:
        builder.add_edge(a, b)
    builder.add_conditional_edges("dispatch_node", dispatch_route, ["research_node", "review_node"])
    builder.add_conditional_edges("review_node", review_route, ["plan_node", "write_node"])
    builder.add_conditional_edges("continuation_node", continuation_route, ["dispatch_node", "plan_node", "review_node"])
    return builder.compile(checkpointer=checkpointer)
