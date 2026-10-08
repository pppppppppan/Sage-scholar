"""Re-open an ended graph at a durable workflow boundary, preserving completed work."""
from sage_scholar.domain.research import FollowUp, GapReview


def continue_graph(graph, config, snapshot, request, limits):
    generation = int(request.metadata.get("continuation", 0))
    if not snapshot.values or snapshot.next or generation <= snapshot.values.get("continuation_generation", 0):
        return snapshot
    state = snapshot.values
    old = state["research_contract"]
    tasks = {key: task.model_copy(update={"status": "pending", "revision": task.revision + 1})
             for key, task in state.get("tasks", {}).items() if task.status in {"failed", "blocked"}}
    review = state.get("gap_review") or GapReview()
    if not review.next_tasks and state.get("report_result"):
        gaps = state["report_result"].remaining_gaps
        review = review.model_copy(update={"next_tasks": [FollowUp(objective="针对未覆盖项补充证据：" + gap) for gap in gaps[:6]]})
    contract = old.model_copy(update={"execution": request, "limits": limits})
    graph.update_state(config, {"research_contract": contract, "tasks": tasks,
        "gap_review": review, "refine_round": 0, "writer_revision": 0, "draft_gaps": [],
        "task_gaps": [], "planning_exhausted": False, "blocked_by_budget": False,
        "continuation_generation": generation}, as_node="continuation_node")
    return graph.get_state(config)
