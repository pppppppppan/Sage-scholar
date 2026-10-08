"""Native LangGraph decision/interrupt nodes shared by long workflows."""
import json
from langgraph.types import interrupt, Command
from pydantic import BaseModel
from sage_scholar.research.structured import structured_response
from .human import HumanQuestion
from .control import check_control


class Decision(BaseModel):
    needs_user: bool = False
    question: HumanQuestion | None = None


def assess(request, llm):
    check_control()
    explicit = request.metadata.get("human_question")
    if explicit:
        return HumanQuestion.model_validate(explicit).model_dump()
    decision = structured_response(llm, Decision, [
        {"role": "system", "content": "判断执行研究/引用审查前是否必须让用户作重要选择。仅当研究范围有互斥方向且无法合理自主确定、关键假设/已给资料与要求明显冲突或缺少执行必需输入时 needs_user=true。普通事实未知、证据不足、低风险选择不要询问，应自主检索。不要要求用户授权不允许的来源。问题必须具体，最多提供4个选项，也允许文字回答。"},
        {"role": "user", "content": json.dumps({"query": request.query, "history": request.history[-6:],
            "source_policy": request.source_policy.model_dump(mode="json"),
            "constraints": request.metadata.get("constraints", [])}, ensure_ascii=False)},
    ], tag="interaction:assess")
    return decision.question.model_dump() if decision.needs_user and decision.question else None


def ask(question):
    check_control()
    return interrupt(question) if question else None


def resume_command(snapshot, saved_answers, fallback):
    pending = {i.id for task in snapshot.tasks for i in task.interrupts} if snapshot else set()
    answers = {key: value for key, value in saved_answers.items() if key in pending}
    return Command(resume=answers) if answers else fallback
