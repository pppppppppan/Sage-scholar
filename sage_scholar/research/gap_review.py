import json

from sage_scholar.domain.research import FollowUp, GapReview
from .structured import structured_response


class GapReviewer:
    def __init__(self, llm, context):
        self.llm, self.context = llm, context

    def review(self, contract, tasks, notes):
        try:
            result = structured_response(self.llm, GapReview, [
                {"role": "system", "content": (
                    "检查研究契约的维度和交付项是否被 findings 覆盖，检查 unknowns/conflicts。"
                    "不要按文风、篇幅或引用数量判断完成。主要结论必须关联证据 ID。"
                    "输出 covered/missing/conflicts/unknowns/next_tasks；next_tasks 指向具体补证任务，"
                    "可以用已有 task_id 重试，不得改变资料约束。只评价本次输入，不进行搜索。"
                )},
                {"role": "user", "content": json.dumps({
                    "contract": contract.model_dump(),
                    "tasks": [task.model_dump() for task in tasks],
                    "notes": [note.model_dump() for note in notes],
                }, ensure_ascii=False)},
            ], tag="research:gap_review")
        except Exception as exc:
            result = GapReview(missing=[f"缺口审查失败：{exc}"])
        # Enforce coverage/identity independently of an optimistic model review.
        backed_dimensions = set()
        note_by_task = {note.task_id: note for note in notes}
        for task in tasks:
            note = note_by_task.get(task.task_id)
            if not note or task.status in {"failed", "blocked"}:
                result.missing.append(f"任务未完成：{task.objective}")
                result.next_tasks.append(FollowUp(task_id=task.task_id, objective=task.objective))
                continue
            for finding in note.findings:
                valid = bool(finding.evidence_ids)
                for identity in finding.evidence_ids:
                    try:
                        self.context.get(identity)
                    except ValueError:
                        valid = False
                if valid:
                    backed_dimensions.update(finding.dimensions)
                else:
                    result.missing.append(f"缺少证据：{finding.text}")
            if not note.findings or not note.evidence_ids:
                result.missing.append(f"缺少来源：{task.objective}")
                result.next_tasks.append(FollowUp(task_id=task.task_id, objective=task.objective))
            result.unknowns.extend(note.unknowns)
            result.conflicts.extend(note.conflicts)
            result.missing.extend(note.metadata.get("validation_issues", []))
            result.missing.extend(note.limitations)
            task_gaps = [*note.unknowns, *note.conflicts, *note.limitations,
                         *note.metadata.get("validation_issues", [])]
            task_gaps.extend(f"缺少证据：{finding.text}" for finding in note.findings
                             if not finding.evidence_ids)
            if task_gaps:
                result.next_tasks.append(FollowUp(
                    task_id=task.task_id, objective=task.objective + "；核实：" + "; ".join(task_gaps),
                    dimensions=task.dimensions,
                ))
        result.covered = [dim for dim in result.covered if dim in backed_dimensions]
        for dim in contract.requested_dimensions:
            if dim not in result.covered:
                result.missing.append(f"未覆盖维度：{dim}")
                result.next_tasks.append(FollowUp(objective=f"补充来源以回答 {dim}：{contract.objective}", dimensions=[dim]))
        result.missing = list(dict.fromkeys(result.missing))
        result.unknowns = list(dict.fromkeys(result.unknowns))
        result.conflicts = list(dict.fromkeys(result.conflicts))
        if result.gaps and not result.next_tasks:
            result.next_tasks = [FollowUp(objective="针对以下缺口补充原文证据：" + "; ".join(result.gaps))]
        if not result.gaps:
            result.next_tasks = []
        return result
