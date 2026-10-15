"""Resume the same graph with current budgets and preserved completed Markdown notes."""


def continue_graph(graph, config, snapshot, request):
    if not snapshot.values:
        return snapshot
    state = snapshot.values
    # Preserve the original scope and HITL decision; only lifecycle metadata/budgets are refreshed.
    old = state["execution"]
    execution = old.model_copy(update={"budget": request.budget,
        "metadata": {**request.metadata, "constraints": old.metadata.get("constraints", [])}})
    generation = int(request.metadata.get("continuation", 0))
    if snapshot.next:
        # Leave pending writes/Send tasks intact. Node entry refreshes budgets from the current Run.
        return snapshot
    if generation <= state.get("continuation_generation", 0):
        return snapshot
    tasks = {key: task.model_copy(update={"status": "pending", "revision": task.revision + 1})
             for key, task in state.get("tasks", {}).items() if task.status in {"failed", "blocked"}}
    review = state.get("supervisor_review")
    # Gaps are not executable tasks. With no already-reviewed follow-ups, continuation
    # routes back through Supervisor Review using all preserved notes and task identities.
    graph.update_state(config, {"execution": execution, "tasks": tasks, "supervisor_review": review,
        "round_base": state.get("review_round", 0), "planning_exhausted": False, "blocked_by_budget": False,
        "continuation_generation": generation}, as_node="continuation_node")
    return graph.get_state(config)
