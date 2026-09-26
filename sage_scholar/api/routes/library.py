import os
import shutil
import tempfile
from uuid import uuid4
from fastapi import APIRouter, Form, HTTPException, UploadFile
from ..dependencies import Container
from ..schemas import IngestRequest, IngestResult, SaveReportRequest

router = APIRouter()

@router.get("/api/library")
async def list_documents(container: Container):
    return await container.document_service.list_documents()

@router.post("/api/library/ingest", response_model=IngestResult)
async def ingest_document(body: IngestRequest, container: Container):
    try:
        return await container.document_service.ingest(
            body.src,
            custom_title=body.custom_title,
            overwrite=body.overwrite, document_id=body.document_id,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/api/library/upload", response_model=IngestResult)
async def upload_document(file: UploadFile, container: Container, document_id: str | None = Form(None)):
    filename = (file.filename or "upload.pdf").replace("/", "_").replace("\\", "_")
    os.makedirs(container.config.data_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=container.config.data_dir)
    source = os.path.join(temp_dir, filename)
    try:
        with open(source, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                output.write(chunk)
        return await container.document_service.ingest(
            source, overwrite=True, document_id=document_id,
            source_identifier="upload:" + uuid4().hex,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@router.post("/api/library/save-report", response_model=IngestResult)
async def save_report(body: SaveReportRequest, container: Container):
    safe_title = body.title.replace("/", "_").replace("\\", "_")
    os.makedirs(container.config.data_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=container.config.data_dir)
    source = os.path.join(temp_dir, f"{safe_title}.md")
    try:
        with open(source, "w", encoding="utf-8") as output:
            output.write(body.content)
        return await container.document_service.ingest(
            source,
            custom_title=body.title,
            overwrite=True, document_id=body.document_id,
            source_identifier="report:" + uuid4().hex,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@router.delete("/api/library/{title}")
async def delete_document(title: str, container: Container):
    try:
        await container.document_service.delete(title)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    return {"message": f"deleted: {title}"}
