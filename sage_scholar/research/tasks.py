from sage_scholar.domain.research import ResearchTask


def validate_tasks(tasks):
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Planner task_id 重复")
    completed = set()
    while len(completed) < len(tasks):
        ready = [
            task.task_id for task in tasks
            if task.task_id not in completed and set(task.dependencies) <= completed
        ]
        if not ready:
            raise ValueError("Task DAG 存在循环或未知依赖")
        completed.update(ready)
    return by_id


def follow_up_tasks(existing, suggestions, revision, limit):
    tasks = {key: value.model_copy(deep=True) for key, value in existing.items()}
    changed = set()
    for suggestion in suggestions:
        identity = suggestion.task_id
        if identity in tasks:
            if identity in changed:
                tasks[identity].objective += "；" + suggestion.objective
                continue
            task = tasks[identity]
            task.objective = suggestion.objective
            task.dimensions = list(dict.fromkeys([*task.dimensions, *suggestion.dimensions]))
            task.status, task.revision = "pending", task.revision + 1
            changed.add(identity)
        elif len(tasks) < limit:
            identity = f"R{revision}-T{len(tasks) + 1}"
            tasks[identity] = ResearchTask(task_id=identity, objective=suggestion.objective,
                                          dimensions=suggestion.dimensions, revision=revision)
            changed.add(identity)
    # Invalidated dependencies force their descendants to rerun, never keep stale notes.
    while True:
        descendants = {key for key, task in tasks.items()
                       if key not in changed and set(task.dependencies) & changed}
        if not descendants:
            break
        for identity in descendants:
            tasks[identity].status = "pending"
            tasks[identity].revision += 1
        changed.update(descendants)
    validate_tasks(list(tasks.values()))
    return tasks, changed
