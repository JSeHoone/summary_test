"""Job repository for data persistence."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from app.core.config import settings
from app.core.errors import JobNotFoundError, StorageError
from app.core.logging import get_logger
from app.jobs.job_models import Job, JobStatus

logger = get_logger(__name__)


class JobRepository:
    """Repository for job persistence using filesystem storage."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        """Initialize the job repository.

        Args:
            storage_dir: Directory for job storage. Defaults to settings.JOB_STORAGE_DIR.
        """
        self.storage_dir = storage_dir or settings.JOB_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        """Get the path to a job's JSON file."""
        return self.storage_dir / f"{job_id}.json"

    async def create(
        self,
        job_id: str,
        file_hash: str,
        original_filename: str,
        audio_path: str | None = None,
    ) -> Job:
        """Create a new job.

        Args:
            job_id: Unique job identifier.
            file_hash: SHA-256 hash of the uploaded file.
            original_filename: Original name of the uploaded file.
            audio_path: Path to the stored audio file.

        Returns:
            The created Job instance.
        """
        job = Job(
            id=job_id,
            file_hash=file_hash,
            original_filename=original_filename,
            audio_path=audio_path,
            status=JobStatus.QUEUED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        await self._save(job)
        logger.info("Job created", job_id=job_id, file_hash=file_hash[:16])
        return job

    async def get(self, job_id: str) -> Job:
        """Get a job by ID.

        Args:
            job_id: The job identifier.

        Returns:
            The Job instance.

        Raises:
            JobNotFoundError: If job doesn't exist.
        """
        job_path = self._job_path(job_id)
        if not job_path.exists():
            raise JobNotFoundError(job_id)

        try:
            async with aiofiles.open(job_path, encoding="utf-8") as f:
                content = await f.read()
                if not content.strip():
                    raise StorageError("Job file is empty", path=str(job_path))
                data = json.loads(content)
                return Job(**data)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in job file", job_id=job_id, error=str(e))
            raise StorageError(f"Invalid job data: {e}", path=str(job_path))
        except StorageError:
            raise
        except Exception as e:
            logger.error("Failed to read job", job_id=job_id, error=str(e))
            raise StorageError(f"Failed to read job: {e}", path=str(job_path))

    async def find_by_hash(self, file_hash: str) -> Job | None:
        """Find an existing job by file hash.

        Args:
            file_hash: SHA-256 hash of the file.

        Returns:
            The Job instance if found, None otherwise.
        """
        for job_file in self.storage_dir.glob("*.json"):
            try:
                async with aiofiles.open(job_file, encoding="utf-8") as f:
                    content = await f.read()
                    if not content.strip():
                        continue
                    data = json.loads(content)
                    if data.get("file_hash") == file_hash:
                        job = Job(**data)
                        # Only return if job completed successfully
                        if job.status == JobStatus.DONE:
                            logger.info(
                                "Found existing job by hash",
                                job_id=job.id,
                                file_hash=file_hash[:16],
                            )
                            return job
            except Exception:
                continue
        return None

    async def update(
        self,
        job_id: str,
        status: JobStatus | None = None,
        progress: int | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Job:
        """Update a job's fields.

        Args:
            job_id: The job identifier.
            status: New status (optional).
            progress: New progress percentage (optional).
            error: Error message (optional).
            result: Processing result (optional).
            metrics: Processing metrics (optional).

        Returns:
            The updated Job instance.
        """
        job = await self.get(job_id)

        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = progress
        if error is not None:
            job.error = error
        if result is not None:
            job.result = result
        if metrics is not None:
            job.metrics.update(metrics)

        job.updated_at = datetime.utcnow()
        await self._save(job)

        logger.debug(
            "Job updated",
            job_id=job_id,
            status=job.status.value,
            progress=job.progress,
        )
        return job

    async def update_status(self, job_id: str, status: JobStatus) -> Job:
        """Update job status.

        Args:
            job_id: The job identifier.
            status: New status.

        Returns:
            The updated Job instance.
        """
        return await self.update(job_id, status=status)

    async def update_progress(
        self,
        job_id: str,
        current: int,
        total: int,
    ) -> Job:
        """Update job progress.

        Args:
            job_id: The job identifier.
            current: Current step number.
            total: Total number of steps.

        Returns:
            The updated Job instance.
        """
        progress = int((current / total) * 100) if total > 0 else 0
        return await self.update(job_id, progress=progress)

    async def complete(
        self,
        job_id: str,
        result: dict[str, Any],
        metrics: dict[str, float] | None = None,
    ) -> Job:
        """Mark a job as complete with result.

        Args:
            job_id: The job identifier.
            result: The processing result.
            metrics: Processing metrics (optional).

        Returns:
            The updated Job instance.
        """
        logger.info("Job completed", job_id=job_id)
        return await self.update(
            job_id,
            status=JobStatus.DONE,
            progress=100,
            result=result,
            metrics=metrics,
        )

    async def fail(self, job_id: str, error: str) -> Job:
        """Mark a job as failed.

        Args:
            job_id: The job identifier.
            error: The error message.

        Returns:
            The updated Job instance.
        """
        logger.error("Job failed", job_id=job_id, error=error)
        return await self.update(
            job_id,
            status=JobStatus.FAILED,
            error=error,
        )

    async def delete(self, job_id: str) -> None:
        """Delete a job.

        Args:
            job_id: The job identifier.
        """
        job_path = self._job_path(job_id)
        if job_path.exists():
            job_path.unlink()
            logger.info("Job deleted", job_id=job_id)

    async def find_before(self, cutoff: datetime) -> list[Job]:
        """Find jobs created before a cutoff date.

        Args:
            cutoff: The cutoff datetime.

        Returns:
            List of jobs created before the cutoff.
        """
        old_jobs: list[Job] = []
        for job_file in self.storage_dir.glob("*.json"):
            try:
                async with aiofiles.open(job_file, encoding="utf-8") as f:
                    content = await f.read()
                    if not content.strip():
                        continue
                    data = json.loads(content)
                    job = Job(**data)
                    if job.created_at < cutoff:
                        old_jobs.append(job)
            except Exception:
                continue
        return old_jobs

    async def _save(self, job: Job) -> None:
        """Save a job to disk.

        Args:
            job: The job to save.
        """
        import asyncio
        import os

        job_path = self._job_path(job.id)
        temp_path = job_path.with_suffix(".tmp")
        max_retries = 5
        retry_delay = 0.1  # Start with 100ms

        for attempt in range(max_retries):
            try:
                # Write to temp file first, then rename (atomic operation)
                async with aiofiles.open(temp_path, mode="w", encoding="utf-8") as f:
                    await f.write(job.model_dump_json(indent=2))
                    await f.flush()

                # On Windows, we need to remove the target file first if it exists
                # because os.replace/Path.replace can fail with PermissionError
                if os.name == "nt" and job_path.exists():
                    try:
                        job_path.unlink()
                    except PermissionError:
                        # File is locked, wait and retry
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue

                # Atomic rename
                temp_path.replace(job_path)
                return  # Success

            except PermissionError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "File locked, retrying save",
                        job_id=job.id,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Clean up temp file if exists
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except Exception:
                            pass
                    logger.error("Failed to save job after retries", job_id=job.id, error=str(e))
                    raise StorageError(f"Failed to save job: {e}", path=str(job_path))

            except Exception as e:
                # Clean up temp file if exists
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                logger.error("Failed to save job", job_id=job.id, error=str(e))
                raise StorageError(f"Failed to save job: {e}", path=str(job_path))


# Global repository instance
job_repository = JobRepository()
