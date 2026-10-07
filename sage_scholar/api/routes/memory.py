import asyncio
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
        raise HTTPException(404, "memory/job not found") from exc
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


def public_job(job):
    return {key: job[key] for key in ('job_id','source_type','source_id','workspace_id','session_id',
        'status','attempts','created_at','updated_at')} | {
        "error": "记忆整理失败，可重新提交；已有记忆不会被清空。" if job['error'] else None}


@router.post("/extract", status_code=202)
async def extract_memory(body: ExtractRequest, container: Container):
    try:
        if body.run_id:
            job = container.memory_background.enqueue_research(body.run_id, retry=True)
        else:
            job = container.memory_background.enqueue_conversation([m.model_dump() for m in body.messages],
                workspace_id=body.workspace_id, session_id=body.session_id, retry=True)
        return public_job(job)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "无法保存记忆整理任务，请稍后重试。") from exc


@router.get("/jobs")
async def list_jobs(container: Container, workspace_id: str = "default", session_id: str | None = None):
    return [public_job(job) for job in container.memory_jobs.list(workspace_id, session_id)]


@router.get("/jobs/{identity}")
async def get_job(identity: str, container: Container):
    job = container.memory_jobs.get(identity)
    if not job:
        raise HTTPException(404, "memory job not found")
    return public_job(job)


@router.put("/{identity}")
async def update_memory(identity: str, body: MemoryInput, container: Container):
    return await operate(service(container).update, identity, body)


@router.delete("/{identity}")
async def delete_memory(identity: str, container: Container):
    await operate(service(container).delete, identity)
    return {"deleted": identity}
