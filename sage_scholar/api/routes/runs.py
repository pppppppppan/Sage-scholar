import json
from fastapi import APIRouter, HTTPException, Request, Query
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
async def stream_run_events(run_id: str, request: Request, container: Container, after: int | None = Query(default=None, ge=0)):
    if container.run_manager.get(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")

    if after is None:
        try:
            after = max(0, int(request.headers.get("last-event-id", "0")))
        except ValueError as exc:
            raise HTTPException(400, "invalid Last-Event-ID") from exc

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


@router.get("/api/runs")
async def list_runs(container: Container):
    return {"runs": container.run_manager.list()}


@router.post("/api/runs/{run_id}/resume", response_model=RunResponse)
async def resume_run(run_id: str, container: Container):
    try:
        return RunResponse(run=await container.run_manager.resume(run_id))
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/api/runs/{run_id}/restart", response_model=RunResponse)
async def restart_run(run_id: str, container: Container):
    try:
        return RunResponse(run=await container.run_manager.restart(run_id))
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
