from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from ..dependencies import Container

router = APIRouter(prefix="/api/conversations")

class CreateConversation(BaseModel):
    workspace_id: str = Field(default="default", min_length=1, max_length=200)
    title: str = Field(default="新对话", min_length=1, max_length=200)

class UpdateConversation(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    archived: bool | None = None

@router.get("")
async def list_conversations(container: Container, search: str = Query(default="", max_length=200), archived: bool = False, workspace_id: str = "default"):
    return {"conversations": container.run_store.conversations.list(workspace_id, search, archived)}

@router.post("", status_code=201)
async def create_conversation(body: CreateConversation, container: Container):
    return {"conversation": container.run_store.conversations.create(body.workspace_id, body.title)}

@router.get("/{identity}")
async def get_conversation(identity: str, container: Container):
    try:
        return container.conversations.detail(identity)
    except KeyError as exc:
        raise HTTPException(404, "conversation not found") from exc

@router.patch("/{identity}")
async def update_conversation(identity: str, body: UpdateConversation, container: Container):
    try:
        async with container.conversations.lock(identity):
            return {"conversation": container.run_store.conversations.update(identity, **body.model_dump(exclude_unset=True))}
    except KeyError as exc:
        raise HTTPException(404, "conversation not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@router.delete("/{identity}", status_code=204)
async def delete_conversation(identity: str, container: Container):
    try:
        await container.conversations.delete(identity)
    except KeyError as exc:
        raise HTTPException(404, "conversation not found") from exc
