"""Plan research questions; evidence coverage review lives in research.gap_review."""
from pydantic import BaseModel, Field
from sage_scholar.domain.research import ResearchTask
from sage_scholar.context.protection import prompt_contract
from sage_scholar.research.structured import structured_response
from sage_scholar.research.materials import encoded


class ResearchPlan(BaseModel):
    analysis: str
    tasks: list[ResearchTask] = Field(min_length=1, max_length=12)


class Supervisor:
    def __init__(self, llm, max_steps, review_llm=None):
        self.llm, self.max_steps = llm, max_steps

    def plan_tasks(self, contract):
        result = structured_response(self.llm, ResearchPlan, [
            {"role": "system", "content": (
                "规划科研调研，先分析用户真正要了解什么，再分解为相互补充、少量可执行的研究任务。"
                "领域综述通常覆盖定义动机、代表方法与机制、实验与适用条件、局限和建议；"
                "两论文比较围绕共同问题、方法、实验可比性与差异，不套空泛模板。"
                "每个任务有唯一 task_id、简短 label、具体 objective、dimensions 和 completion_criteria。"
                "完成标准描述需要解释的问题和原文证据，不是凑引用数量。"
                "独立任务 dependencies=[]，需要前序结果才设置依赖；不得循环。"
                "不能只是搜索不同关键词，不安排重复全局综述。status=pending，revision=0。"
                "严格遵守资料权限与用户限制。"
            )},
            {"role": "user", "content": encoded({"contract": prompt_contract(contract),
                "max_tasks": contract.limits.max_tasks, "steps_per_task": self.max_steps})},
        ], tag="supervisor:plan", max_tokens=3500)
        if len(result.tasks) > contract.limits.max_tasks:
            raise ValueError("规划任务数量超过预算")
        return [task.model_copy(update={"status": "pending", "revision": 0}) for task in result.tasks]
