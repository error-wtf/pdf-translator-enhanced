from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, Optional, List
from uuid import uuid4

from .models import JobInfo, JobStatus, Block

logger = logging.getLogger("pdf_translator.jobs")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

_jobs: Dict[str, JobInfo] = {}


def _save_job(job: JobInfo) -> JobInfo:
    _jobs[job.job_id] = job
    return job


def create_job(
    target_language: str = "de",
    use_openai: bool = False,
    openai_api_key: Optional[str] = None,
) -> JobInfo:
    job_id = str(uuid4())

    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    job = JobInfo(
        job_id=job_id,
        status=JobStatus.queued,
        progress=0,
        message=None,
        target_language=target_language,
        use_openai=use_openai,
        openai_api_key_set=bool(openai_api_key),
    )
    logger.info(
        "Created job %s (target_language=%s, use_openai=%s, api_key_set=%s)",
        job_id,
        target_language,
        use_openai,
        bool(openai_api_key),
    )
    return _save_job(job)


def get_job(job_id: str) -> JobInfo:
    if job_id not in _jobs:
        raise KeyError(f"Unknown job_id: {job_id}")
    return _jobs[job_id]


def update_job(
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
) -> JobInfo:
    job = get_job(job_id)

    data = job.model_dump()
    if status is not None:
        data["status"] = status
    if progress is not None:
        data["progress"] = progress
    if message is not None:
        data["message"] = message

    updated = JobInfo(**data)
    _save_job(updated)

    logger.info(
        "Job %s updated: status=%s, progress=%s, message=%s",
        job_id,
        updated.status,
        updated.progress,
        updated.message,
    )
    return updated


def start_job_thread(
    job_id: str,
    pdf_path: Path,
    target_language: str,
    use_openai: bool,
    openai_api_key: Optional[str],
) -> None:
    logger.info(
        "Starting background thread for job %s (pdf=%s, target_language=%s, use_openai=%s, api_key_set=%s)",
        job_id,
        pdf_path,
        target_language,
        use_openai,
        bool(openai_api_key),
    )
    thread = threading.Thread(
        target=run_job_in_background,
        args=(job_id, pdf_path, target_language, use_openai, openai_api_key),
        daemon=True,
    )
    thread.start()


def run_job_in_background(
    job_id: str,
    pdf_path: Path,
    target_language: str,
    use_openai: bool,
    openai_api_key: Optional[str],
) -> None:
    from . import pdf_processing, latex_build

    logger.info("Job %s: background pipeline started", job_id)

    try:
        update_job(
            job_id,
            status=JobStatus.analyzing,
            progress=10,
            message="Analyzing PDF…",
        )

        blocks, detected_source_language = pdf_processing.analyze_pdf(pdf_path)
        logger.info(
            "Job %s: analyze_pdf finished (blocks=%d, detected_source_language=%s)",
            job_id,
            len(blocks),
            detected_source_language,
        )

        update_job(
            job_id,
            status=JobStatus.translating,
            progress=40,
            message=(
                f"Translating from {detected_source_language or 'unknown'} "
                f"to {target_language}…"
            ),
        )

        translated_blocks: List[Block] = pdf_processing.translate_blocks(
            blocks,
            source_language=detected_source_language,
            target_language=target_language,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
        )

        num_blocks = len(blocks)
        num_translated = sum(
            1
            for b in translated_blocks
            if (b.translated_latex or "").strip()
            and (b.translated_latex != (b.content or ""))
        )
        logger.info(
            "Job %s: translation step finished (blocks=%d, translated_blocks=%d)",
            job_id,
            num_blocks,
            num_translated,
        )

        if use_openai and openai_api_key and num_blocks > 0 and num_translated == 0:
            raise RuntimeError(
                "Translation appears to have failed for all blocks (no translated_latex set)."
            )

        update_job(
            job_id,
            status=JobStatus.latex_build,
            progress=70,
            message="Building LaTeX and compiling PDF…",
        )

        latex_build.build_and_compile(
            job_id=job_id,
            blocks=translated_blocks,
            source_language=detected_source_language,
            target_language=target_language,
        )
        logger.info("Job %s: LaTeX build & pdflatex finished", job_id)

        update_job(
            job_id,
            status=JobStatus.done,
            progress=100,
            message="Finished successfully.",
        )
        logger.info("Job %s: completed successfully", job_id)

    except Exception as exc:
        import traceback

        traceback.print_exc()
        logger.exception("Job %s: pipeline failed with exception", job_id)
        update_job(
            job_id,
            status=JobStatus.error,
            progress=100,
            message=f"Error: {exc}",
        )
