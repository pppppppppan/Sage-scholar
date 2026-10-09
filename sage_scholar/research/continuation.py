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
    contract = old.model_copy(update={"execution": request, "limits": limits})
    if not tasks and not review.gaps and state.get("outline"):
        sections, reviews = state.get("sections", {}), state.get("section_reviews", {})
        unfinished = [index for index, plan in enumerate(state["outline"].sections)
                      if plan.section_id not in sections or not reviews.get(plan.section_id)
                      or not reviews[plan.section_id].acceptable]
        if unfinished:
            graph.update_state(config, {"research_contract": contract, "resume_report": True,
                "section_index": unfinished[0], "review_index": unfinished[0], "section_failures": [],
                "continuation_generation": generation}, as_node="continuation_node")
            return graph.get_state(config)
    if not review.next_tasks and state.get("report_result"):
        gaps = state["report_result"].remaining_gaps
        review = review.model_copy(update={"next_tasks": [FollowUp(objective="针对未覆盖项补充证据：" + gap) for gap in gaps[:6]]})
    graph.update_state(config, {"research_contract": contract, "tasks": tasks,
        "gap_review": review, "refine_round": 0, "resume_report": False,
        "task_gaps": [], "reviewed_evidence_ids": [], "reviewed_dimensions": [], "planning_exhausted": False, "blocked_by_budget": False,
        "continuation_generation": generation}, as_node="continuation_node")
    return graph.get_state(config)
