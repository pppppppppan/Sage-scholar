"""Bounded supervisor review; limitations do not automatically trigger re-search."""
from sage_scholar.context.protection import prompt_contract
from sage_scholar.domain.research import GapReview, FollowUp
from .structured import structured_response
from .materials import batches, note_units, encoded


class GapReviewer:
    def __init__(self, llm, context):
        self.llm, self.context = llm, context

    def review(self, contract, tasks, notes):
        result = GapReview()
        for group in batches(note_units(notes)):
            checked = structured_response(self.llm, GapReview, [
                {"role": "system", "content": (
                    "作为研究主管，评价这批材料对用户问题的贡献及关键缺口。"
                    "仅标记实际覆盖的维度；缺少其他批次不等于全局缺失。"
                    "评价内容、方法机制、比较条件、反例与解释，不按引用密度或字数评分。"
                    "常规局限和诚实未知不触发重复研究。next_tasks 仅提出可通过定向读取新来源解决、"
                    "且影响交付的具体缺口，不重复全局调研；不得放宽资料策略。"
                    "搜索摘要或综述不能冒充已读一手原文。不要输出原文、内部错误或日志。"
                )},
                {"role": "user", "content": encoded({
                    "contract": prompt_contract(contract),
                    "tasks": [{"task_id": t.task_id, "objective": t.objective,
                               "completion_criteria": t.completion_criteria} for t in tasks],
                    "materials": group,
                })},
            ], tag="supervisor:gap_review", max_tokens=1800)
            for key in ("covered", "missing", "conflicts", "unknowns"):
                getattr(result, key).extend(getattr(checked, key))
            result.next_tasks.extend(checked.next_tasks)
        backed = set()
        for note in notes:
            result.unknowns.extend(note.unknowns)
            result.conflicts.extend(note.conflicts)
            for finding in note.findings:
                try:
                    if finding.evidence_ids:
                        for identity in finding.evidence_ids:
                            self.context.get(identity)
                        backed.update(finding.dimensions)
                    else:
                        result.missing.append("部分发现缺少可追溯的原文证据")
                except ValueError:
                    result.missing.append("部分发现缺少可追溯的原文证据")
        result.covered = list(dict.fromkeys(d for d in result.covered if d in backed))
        for task in tasks:
            if task.status in {"blocked", "failed"} or not any(n.task_id == task.task_id and n.findings for n in notes):
                result.missing.append(f"尚未完成：{task.objective}")
                if task.revision < 1:
                    result.next_tasks.append(FollowUp(task_id=task.task_id, objective=task.objective, dimensions=task.dimensions))
        for dim in contract.requested_dimensions:
            if dim not in result.covered:
                result.missing.append(f"尚需补充或解释：{dim}")
        previous, seen, selected = {t.task_id: t for t in tasks}, set(), []
        for item in result.next_tasks:
            target = previous.get(item.task_id)
            if item.task_id and (not target or target.revision >= 1):
                continue
            key = item.task_id or item.objective.strip().casefold()
            if key not in seen:
                seen.add(key)
                selected.append(item)
        result.next_tasks = selected[:3]
        for key in ("missing", "conflicts", "unknowns"):
            setattr(result, key, list(dict.fromkeys(getattr(result, key))))
        return result
