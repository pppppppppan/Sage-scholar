import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from ..dependencies import Container
from ..schemas import RunResponse

router = APIRouter()

@router.get("/api/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, container: Container):
    run = container.run_manager.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return RunResponse(run=run)

@router.get("/api/runs/{run_id}/events")
async def stream_run_events(run_id: str, request: Request, container: Container, after: int = 0):
    if container.run_manager.get(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")

    async def generate():
        try:
            async for event in container.run_manager.events(run_id, after=after):
                if await request.is_disconnected():
                    return
                payload = json.dumps(event.data, ensure_ascii=False)
                yield f"id: {event.sequence}\nevent: {event.type}\ndata: {payload}\n\n"
        except KeyError:
            return

    return StreamingResponse(generate(), media_type="text/event-stream")

@router.post("/api/runs/{run_id}/cancel", response_model=RunResponse)
async def cancel_run(run_id: str, container: Container):
    run = await container.run_manager.cancel(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return RunResponse(run=run)
