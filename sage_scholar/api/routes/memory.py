import asyncio
from fastapi import APIRouter, HTTPException
from sage_scholar.memory import MemoryInput
from ..dependencies import Container

router = APIRouter(prefix="/api/memories")


@router.get("")
async def list_memory(container: Container, session_id: str | None = None):
    return await asyncio.to_thread(container.memory_store.list, session_id)


@router.post("")
async def add_memory(body: MemoryInput, container: Container):
    try:
        return await asyncio.to_thread(container.memory_store.add, body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.put("/{identity}")
async def update_memory(identity: str, body: MemoryInput, container: Container):
    try:
        return await asyncio.to_thread(container.memory_store.update, identity, body)
    except KeyError as exc:
        raise HTTPException(404, "memory not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/{identity}")
async def delete_memory(identity: str, container: Container):
    try:
        await asyncio.to_thread(container.memory_store.delete, identity)
    except KeyError as exc:
        raise HTTPException(404, "memory not found") from exc
    return {"deleted": identity}
