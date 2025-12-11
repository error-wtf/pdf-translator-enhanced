from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .jobs import (
    BASE_DIR,
    DATA_DIR,
    JOBS_DIR,
    create_job,
    get_job,
    start_job_thread,
)
from .models import JobInfo

logger = logging.getLogger("pdf_translator.main")
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PDF Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UploadResponse(BaseModel):
    job: JobInfo


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    pdf: UploadFile = File(...),
    target_language: str = Form("de"),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    logger.info(
        "Upload received: filename=%s, target_language=%s, use_openai=%s, api_key_provided=%s",
        pdf.filename,
        target_language,
        use_openai,
        bool(openai_api_key),
    )

    job = create_job(
        target_language=target_language,
        use_openai=use_openai,
        openai_api_key=openai_api_key,
    )

    job_dir = JOBS_DIR / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / "original.pdf"
    contents = await pdf.read()
    pdf_path.write_bytes(contents)
    logger.info("Original PDF for job %s written to %s", job.job_id, pdf_path)

    start_job_thread(
        job_id=job.job_id,
        pdf_path=pdf_path,
        target_language=target_language,
        use_openai=use_openai,
        openai_api_key=openai_api_key,
    )

    return UploadResponse(job=job)


@app.get("/job/{job_id}", response_model=JobInfo)
async def job_status(job_id: str):
    try:
        return get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/job/{job_id}/pdf")
async def download_result(job_id: str):
    job_dir = JOBS_DIR / job_id
    pdf_path = job_dir / "main.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Result PDF not found")
    logger.info("Serving result PDF for job %s from %s", job_id, pdf_path)
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename="translated.pdf",
    )
