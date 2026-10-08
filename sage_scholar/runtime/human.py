"""Public interaction contract; workflow state stays in LangGraph."""
from pydantic import BaseModel, Field


class HumanQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    reason: str = Field(default="", max_length=2000)
    options: list[str] = Field(default_factory=list, max_length=8)
    allow_skip: bool = False


class HumanResponse(BaseModel):
    conversation_id: str
    run_id: str
    interrupt_id: str
    answer: str = Field(default="", max_length=10000)
    skip: bool = False


class RunWaiting(Exception):
    def __init__(self, interrupts):
        self.interrupts = interrupts
        super().__init__("等待用户补充信息")


def public_interrupts(interrupts):
    return [{"interrupt_id": item.id, **HumanQuestion.model_validate(item.value).model_dump()} for item in interrupts]
