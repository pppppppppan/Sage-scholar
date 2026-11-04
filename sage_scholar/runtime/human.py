"""Public interaction contract; workflow state stays in LangGraph."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from sage_scholar.routing.models import SourceScope


class RoutingSettings(BaseModel):
    """User-controlled settings only; never populated from an LLM response."""
    model_config = ConfigDict(extra="forbid")
    source_scope: SourceScope | None = None
    allow_web: StrictBool | None = None
    document_ids: list[str] = Field(default_factory=list, max_length=50)


class HumanQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    reason: str = Field(default="", max_length=2000)
    options: list[str] = Field(default_factory=list, max_length=8)
    allow_skip: bool = False


class RouterQuestion(HumanQuestion):
    """Code-authored Router controls; absent from the Research LLM's schema."""
    kind: Literal["information", "source_policy"] = "information"
    routing_settings: RoutingSettings | None = None


class HumanResponse(BaseModel):
    conversation_id: str
    run_id: str
    interrupt_id: str
    answer: str = Field(default="", max_length=10000)
    skip: bool = False
    routing_update: RoutingSettings | None = None


def validate_routing_response(question: dict, response: HumanResponse):
    """Only the bound Router interrupt can accept a typed user settings update."""
    update = response.routing_update
    if update is not None:
        current = question.get("routing_settings")
        if current is None or response.skip:
            raise ValueError("该问题不接受资料设置修改")
        changes = update.model_dump(mode="json", exclude_unset=True)
        if not changes or all(current.get(k) == v for k, v in changes.items()):
            raise ValueError("资料设置没有变化，请修改冲突的设置后提交")
    elif question.get("kind") == "source_policy":
        raise ValueError("请通过资料设置明确修改范围或联网权限后再继续；文字回答不会修改权限")


class RunWaiting(Exception):
    def __init__(self, interrupts):
        self.interrupts = interrupts
        super().__init__("等待用户补充信息")


def public_interrupts(interrupts):
    return [{"interrupt_id": item.id, **RouterQuestion.model_validate(item.value).model_dump()} for item in interrupts]
