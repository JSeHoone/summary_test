"""Job API routes."""

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.core.errors import AppError, JobNotFoundError, ValidationError
from app.core.logging import get_logger
from app.jobs.job_models import (
    ErrorResponse,
    JobCreateResponse,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
    SummaryOutput,
    TranscriptSegment,
)
from app.jobs.job_service import job_service

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/jobs", tags=["Jobs"])


@router.post(
    "",
    response_model=JobCreateResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Create a transcription job",
    description="Upload an audio file to create a new transcription job. "
    "The file will be processed asynchronously.",
)
async def create_job(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
) -> JobCreateResponse:
    """Create a new transcription job.

    Args:
        background_tasks: FastAPI background tasks.
        audio_file: The uploaded audio file.

    Returns:
        Job creation response with job ID and status.
    """
    try:
        job = await job_service.create(audio_file)

        # Only schedule processing if this is a new job (not reused)
        if job.status == JobStatus.QUEUED:
            background_tasks.add_task(job_service.process_pipeline, job.id)
            logger.info("Job scheduled for processing", job_id=job.id)

        return JobCreateResponse(
            job_id=job.id,
            status=job.status,
            created_at=job.created_at,
        )

    except ValidationError as e:
        logger.warning("Validation error", error=e.message)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except AppError as e:
        logger.error("Application error", error=e.message, code=e.code)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("Unexpected error creating job", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Get job status",
    description="Get the current status and progress of a transcription job.",
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get the status of a job.

    Args:
        job_id: The job identifier.

    Returns:
        Job status response.
    """
    try:
        job = await job_service.get_status(job_id)

        return JobStatusResponse(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error=job.error,
        )

    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except AppError as e:
        logger.error("Application error", error=e.message, code=e.code)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("Unexpected error getting job status", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{job_id}/result",
    response_model=JobResultResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Job not completed"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Get job result",
    description="Get the transcript and summary for a completed job.",
)
async def get_job_result(job_id: str) -> JobResultResponse:
    """Get the result of a completed job.

    Args:
        job_id: The job identifier.

    Returns:
        Job result with transcript and summary.
    """
    try:
        job = await job_service.get_result(job_id)

        if not job.result:
            raise ValidationError("Job result is empty", details={"job_id": job_id})

        # Parse segments from result
        segments_data = job.result.get("segments", [])
        segments = [TranscriptSegment(**seg) for seg in segments_data]

        return JobResultResponse(
            job_id=job.id,
            transcript=job.result.get("transcript", ""),
            segments=segments,
            summary=SummaryOutput(**job.result.get("summary", {})),
            metrics=job.metrics,
        )

    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except AppError as e:
        logger.error("Application error", error=e.message, code=e.code)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("Unexpected error getting job result", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/{job_id}/retry",
    response_model=JobStatusResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Job cannot be retried"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Retry a failed job",
    description="Retry a failed transcription job.",
)
async def retry_job(
    job_id: str,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    """Retry a failed job.

    Args:
        job_id: The job identifier.
        background_tasks: FastAPI background tasks.

    Returns:
        Updated job status.
    """
    try:
        job = await job_service.get_status(job_id)

        if job.status != JobStatus.FAILED:
            raise ValidationError(
                f"Only failed jobs can be retried. Current status: {job.status.value}",
                details={"job_id": job_id, "status": job.status.value},
            )

        # Reset job status
        from app.jobs.job_repository import job_repository

        job = await job_repository.update(
            job_id,
            status=JobStatus.QUEUED,
            progress=0,
            error=None,
        )

        # Schedule reprocessing
        background_tasks.add_task(job_service.process_pipeline, job.id)
        logger.info("Job scheduled for retry", job_id=job.id)

        return JobStatusResponse(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error=job.error,
        )

    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except AppError as e:
        logger.error("Application error", error=e.message, code=e.code)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("Unexpected error retrying job", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
