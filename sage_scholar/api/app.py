from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from sage_scholar.application import QueryCommand
from sage_scholar.base import setup_logging
from sage_scholar.infrastructure import ApplicationContainer

from .schemas import IngestRequest, IngestResult, QueryRequest, QueryResponse, RunResponse, SaveReportRequest


def create_app(container: ApplicationContainer | None = None) -> FastAPI:
    application_container = container or ApplicationContainer()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(enable_display=False)
        app.state.container = application_container
        yield
        await application_container.close()

    app = FastAPI(title="SAGE Scholar", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "sage-scholar"}

    @app.post("/api/queries", response_model=QueryResponse)
    async def submit_query(body: QueryRequest, response: Response):
        result = await application_container.query_service.execute(
            QueryCommand(
                query=body.query,
                workflow=body.workflow,
                source_scope=body.source_scope,
                document_ids=body.document_ids,
                allow_web=body.allow_web,
                timezone=body.timezone,
                history=[message.model_dump() for message in body.history],
            )
        )
        if result.run_id:
            response.status_code = status.HTTP_202_ACCEPTED
        return QueryResponse(
            request_id=result.request_id,
            route=result.route,
            status=result.status.value,
            answer=result.answer,
            evidence=result.evidence,
            run_id=result.run_id,
            message=result.message,
        )

    @app.get("/api/runs/{run_id}", response_model=RunResponse)
    async def get_run(run_id: str):
        run = application_container.run_manager.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return RunResponse(run=run)

    @app.get("/api/runs/{run_id}/events")
    async def stream_run_events(run_id: str, request: Request, after: int = 0):
        if application_container.run_manager.get(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")

        async def generate():
            try:
                async for event in application_container.run_manager.events(run_id, after=after):
                    if await request.is_disconnected():
                        return
                    payload = json.dumps(event.data, ensure_ascii=False)
                    yield f"id: {event.sequence}\nevent: {event.type}\ndata: {payload}\n\n"
            except KeyError:
                return

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/api/runs/{run_id}/cancel", response_model=RunResponse)
    async def cancel_run(run_id: str):
        run = await application_container.run_manager.cancel(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return RunResponse(run=run)

    @app.get("/api/library")
    async def list_documents():
        return await application_container.document_service.list_documents()

    @app.post("/api/library/ingest", response_model=IngestResult)
    async def ingest_document(body: IngestRequest):
        try:
            return await application_container.document_service.ingest(
                body.src,
                custom_title=body.custom_title,
                overwrite=body.overwrite,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/library/upload", response_model=IngestResult)
    async def upload_document(file: UploadFile):
        filename = (file.filename or "upload.pdf").replace("/", "_").replace("\\", "_")
        temp_dir = tempfile.mkdtemp(dir=application_container.config.data_dir)
        source = os.path.join(temp_dir, filename)
        try:
            with open(source, "wb") as output:
                while chunk := await file.read(1024 * 1024):
                    output.write(chunk)
            return await application_container.document_service.ingest(source, overwrite=True)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @app.post("/api/library/save-report", response_model=IngestResult)
    async def save_report(body: SaveReportRequest):
        safe_title = body.title.replace("/", "_").replace("\\", "_")
        temp_dir = tempfile.mkdtemp(dir=application_container.config.data_dir)
        source = os.path.join(temp_dir, f"{safe_title}.md")
        try:
            with open(source, "w", encoding="utf-8") as output:
                output.write(body.content)
            return await application_container.document_service.ingest(
                source,
                custom_title=body.title,
                overwrite=True,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @app.delete("/api/library/{title}")
    async def delete_document(title: str):
        await application_container.document_service.delete(title)
        return {"message": f"deleted: {title}"}

    return app


app = create_app()
