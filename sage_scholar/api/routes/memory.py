import asyncio
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator
from sage_scholar.memory import MemoryInput
from sage_scholar.memory.store import MemoryConflict
from ..dependencies import Container
from ..schemas import ChatMessage

router = APIRouter(prefix="/api/memories")


def service(container):
    return getattr(container, "memory_service", container.memory_store)


async def operate(fn, *args, **kwargs):
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except KeyError as exc:
        raise HTTPException(404, "memory/candidate not found") from exc
    except MemoryConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "记忆服务暂不可用，请稍后重试；原记忆未因模型错误而覆盖。") from exc


@router.get("")
async def list_memory(container: Container, session_id: str | None = None, workspace_id: str = "default", include_inactive: bool = False):
    return await operate(service(container).list, session_id, workspace_id=workspace_id, include_inactive=include_inactive)


@router.post("")
async def add_memory(body: MemoryInput, container: Container):
    return await operate(service(container).add, body)


class ExtractRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list, max_length=200)
    run_id: str | None = None
    workspace_id: str = Field(default="default", min_length=1, max_length=200)
    session_id: str | None = None

    @model_validator(mode="after")
    def bounded_source(self):
        if bool(self.messages) == bool(self.run_id):
            raise ValueError("提供 conversation messages 或 run_id 其中一种来源")
        if sum(len(m.content) for m in self.messages) > 100000:
            raise ValueError("提取输入过长，请选择需要整理的对话片段")
        return self


class Approval(BaseModel):
    confirmed: bool = False
    revision: int | None = None


@router.post("/extract")
async def extract_candidates(body: ExtractRequest, container: Container):
    messages = [m.model_dump() for m in body.messages]
    workspace, session, research = body.workspace_id, body.session_id, bool(body.run_id)
    if body.run_id:
        run = container.run_manager.get(body.run_id)
        if not run or run.status not in {"completed", "partial"} or not run.result:
            raise HTTPException(400, "请选择已经结束且有结果的研究任务")
        workspace, session = run.execution.workspace_id, run.execution.session_id
        # The original user request can support decisions; the report supports research leads only.
        messages = [{"role": "user", "content": run.execution.query},
                    {"role": "assistant", "content": json.dumps(run.result, ensure_ascii=False)}]
        if sum(len(m["content"]) for m in messages) > 100000:
            raise HTTPException(400, "研究结果过长，请选择关键笔记后提取")
    return await operate(service(container).extract, messages, workspace_id=workspace, session_id=session,
                         reference=body.run_id or session, research=research)


@router.get("/candidates")
async def list_candidates(container: Container, workspace_id: str = "default", session_id: str | None = None):
    return await operate(service(container).candidates, workspace_id=workspace_id, session_id=session_id)


@router.post("/candidates/{identity}/approve")
async def approve_candidate(identity: str, body: Approval, container: Container):
    return await operate(service(container).approve, identity, confirmed=body.confirmed, revision=body.revision)


@router.delete("/candidates/{identity}")
async def reject_candidate(identity: str, container: Container):
    await operate(service(container).reject, identity)
    return {"rejected": identity}


@router.put("/{identity}")
async def update_memory(identity: str, body: MemoryInput, container: Container):
    return await operate(service(container).update, identity, body)


@router.delete("/{identity}")
async def delete_memory(identity: str, container: Container):
    await operate(service(container).delete, identity)
    return {"deleted": identity}
