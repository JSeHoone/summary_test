"""Job repository for data persistence."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import settings
from app.core.errors import JobNotFoundError, StorageError
from app.core.logging import get_logger
from app.jobs.job_models import Job, JobStatus

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base for SQLAlchemy models."""


class JobRecord(Base):
    """Database model for jobs."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=JobStatus.QUEUED.value)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    audio_object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    error: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False, default=dict)


engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Lightweight migration: add audio_object_key if missing
        result = await conn.execute(
            text(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name='jobs' AND column_name='audio_object_key'
                """
            )
        )
        if result.scalar() is None:
            await conn.execute(text("ALTER TABLE jobs ADD COLUMN audio_object_key VARCHAR(1024)"))


class JobRepository:
    """Repository for job persistence using PostgreSQL."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None = None) -> None:
        """Initialize the job repository.

        Args:
            session_factory: SQLAlchemy async session factory.
        """
        self.session_factory = session_factory or SessionLocal

    @staticmethod
    def _to_job(record: JobRecord) -> Job:
        """Convert a JobRecord to Job model."""
        status_value = (
            record.status.value if isinstance(record.status, JobStatus) else record.status
        )
        return Job(
            id=record.id,
            status=JobStatus(status_value),
            progress=record.progress,
            file_hash=record.file_hash,
            original_filename=record.original_filename,
            audio_path=record.audio_path,
            audio_object_key=record.audio_object_key,
            created_at=record.created_at,
            updated_at=record.updated_at,
            error=record.error,
            result=record.result,
            metrics=record.metrics or {},
        )

    async def create(
        self,
        job_id: str,
        file_hash: str,
        original_filename: str,
        audio_path: str | None = None,
        audio_object_key: str | None = None,
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
        now = datetime.utcnow()
        record = JobRecord(
            id=job_id,
            file_hash=file_hash,
            original_filename=original_filename,
            audio_path=audio_path,
            audio_object_key=audio_object_key,
            status=JobStatus.QUEUED.value,
            progress=0,
            created_at=now,
            updated_at=now,
            metrics={},
        )
        try:
            async with self.session_factory() as session:
                session.add(record)
                await session.commit()
                await session.refresh(record)
            logger.info("Job created", job_id=job_id, file_hash=file_hash[:16])
            return self._to_job(record)
        except Exception as e:
            logger.error("Failed to create job", job_id=job_id, error=str(e))
            raise StorageError(f"Failed to create job: {e}", path="jobs")

    async def get(self, job_id: str) -> Job:
        """Get a job by ID.

        Args:
            job_id: The job identifier.

        Returns:
            The Job instance.

        Raises:
            JobNotFoundError: If job doesn't exist.
        """
        try:
            async with self.session_factory() as session:
                result = await session.execute(select(JobRecord).where(JobRecord.id == job_id))
                record = result.scalar_one_or_none()
                if record is None:
                    raise JobNotFoundError(job_id)
                return self._to_job(record)
        except JobNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to read job", job_id=job_id, error=str(e))
            raise StorageError(f"Failed to read job: {e}", path="jobs")

    async def find_by_hash(self, file_hash: str) -> Job | None:
        """Find an existing job by file hash.

        Args:
            file_hash: SHA-256 hash of the file.

        Returns:
            The Job instance if found, None otherwise.
        """
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(JobRecord)
                    .where(JobRecord.file_hash == file_hash)
                    .where(JobRecord.status == JobStatus.DONE.value)
                    .order_by(JobRecord.created_at.desc())
                )
                record = result.scalars().first()
                if record is None:
                    return None
                logger.info(
                    "Found existing job by hash",
                    job_id=record.id,
                    file_hash=file_hash[:16],
                )
                return self._to_job(record)
        except Exception as e:
            logger.error("Failed to find job by hash", error=str(e))
            raise StorageError(f"Failed to find job by hash: {e}", path="jobs")

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
        try:
            async with self.session_factory() as session:
                result_record = await session.execute(
                    select(JobRecord).where(JobRecord.id == job_id)
                )
                record = result_record.scalar_one_or_none()
                if record is None:
                    raise JobNotFoundError(job_id)

                if status is not None:
                    record.status = status.value
                if progress is not None:
                    record.progress = progress
                if error is not None:
                    record.error = error
                if result is not None:
                    record.result = result
                if metrics is not None:
                    current_metrics = record.metrics or {}
                    current_metrics.update(metrics)
                    record.metrics = current_metrics

                record.updated_at = datetime.utcnow()
                await session.commit()
                await session.refresh(record)

                logger.debug(
                    "Job updated",
                    job_id=job_id,
                    status=record.status,
                    progress=record.progress,
                )
                return self._to_job(record)

        except JobNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update job", job_id=job_id, error=str(e))
            raise StorageError(f"Failed to update job: {e}", path="jobs")

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
        try:
            async with self.session_factory() as session:
                await session.execute(delete(JobRecord).where(JobRecord.id == job_id))
                await session.commit()
            logger.info("Job deleted", job_id=job_id)
        except Exception as e:
            logger.error("Failed to delete job", job_id=job_id, error=str(e))
            raise StorageError(f"Failed to delete job: {e}", path="jobs")

    async def find_before(self, cutoff: datetime) -> list[Job]:
        """Find jobs created before a cutoff date.

        Args:
            cutoff: The cutoff datetime.

        Returns:
            List of jobs created before the cutoff.
        """
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    select(JobRecord).where(JobRecord.created_at < cutoff)
                )
                records = result.scalars().all()
                return [self._to_job(record) for record in records]
        except Exception as e:
            logger.error("Failed to find old jobs", error=str(e))
            raise StorageError(f"Failed to find old jobs: {e}", path="jobs")


# Global repository instance
job_repository = JobRepository()
