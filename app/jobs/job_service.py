"""Job service for business logic."""

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import aiofiles
from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import FileTooLargeError, UnsupportedFormatError, ValidationError
from app.core.logging import get_logger
from app.jobs.job_models import Job, JobStatus
from app.jobs.job_repository import JobRepository, job_repository
from app.storage.minio_storage import MinioStorage, create_minio_storage

if TYPE_CHECKING:
    from app.stt.whisper_engine import WhisperEngine
    from app.summarize.summarizer_interface import SummarizerInterface

logger = get_logger(__name__)


class JobService:
    """Service for managing transcription jobs."""

    def __init__(
        self,
        repository: JobRepository | None = None,
        whisper_engine: "WhisperEngine | None" = None,
        summarizer: "SummarizerInterface | None" = None,
    ) -> None:
        """Initialize the job service.

        Args:
            repository: Job repository instance.
            whisper_engine: Whisper transcription engine.
            summarizer: Summarization engine.
        """
        self.repository = repository or job_repository
        self.whisper_engine = whisper_engine
        self.summarizer = summarizer
        self.work_dir = settings.AUDIO_WORK_DIR
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.storage: MinioStorage | None
        if settings.STORAGE_BACKEND == "minio":
            self.storage = create_minio_storage()
        else:
            self.storage = None

    async def init_storage(self) -> None:
        """Initialize storage backend (e.g., ensure buckets)."""
        if self.storage is not None:
            await self.storage.ensure_bucket()

    def set_whisper_engine(self, engine: "WhisperEngine") -> None:
        """Set the Whisper engine (called after app startup)."""
        self.whisper_engine = engine

    def set_summarizer(self, summarizer: "SummarizerInterface") -> None:
        """Set the summarizer (called after app startup)."""
        self.summarizer = summarizer

    async def create(self, audio_file: UploadFile) -> Job:
        """Create a new transcription job.

        Args:
            audio_file: The uploaded audio file.

        Returns:
            The created job.

        Raises:
            ValidationError: If file validation fails.
            FileTooLargeError: If file exceeds size limit.
            UnsupportedFormatError: If file format not supported.
        """
        # Validate file
        await self._validate_upload(audio_file)

        # Calculate file hash and check for existing job
        file_hash = await self._calculate_file_hash(audio_file)

        # Check for existing completed job with same hash (idempotency)
        existing_job = await self.repository.find_by_hash(file_hash)
        if existing_job:
            logger.info(
                "Reusing existing job",
                job_id=existing_job.id,
                file_hash=file_hash[:16],
            )
            return existing_job

        # Generate job ID and save file
        job_id = str(uuid4())
        audio_path = await self._save_upload(audio_file, job_id)

        # Upload original audio to object storage (if configured)
        audio_object_key = await self._upload_to_object_storage(audio_path, job_id)

        # Create job
        job = await self.repository.create(
            job_id=job_id,
            file_hash=file_hash,
            original_filename=audio_file.filename or "unknown",
            audio_path=str(audio_path),
            audio_object_key=audio_object_key,
        )

        logger.info(
            "Job created",
            job_id=job_id,
            filename=audio_file.filename,
            file_hash=file_hash[:16],
        )
        return job

    async def get_status(self, job_id: str) -> Job:
        """Get the status of a job.

        Args:
            job_id: The job identifier.

        Returns:
            The job.
        """
        return await self.repository.get(job_id)

    async def get_result(self, job_id: str) -> Job:
        """Get the result of a completed job.

        Args:
            job_id: The job identifier.

        Returns:
            The job with result.

        Raises:
            ValidationError: If job is not completed.
        """
        job = await self.repository.get(job_id)

        if job.status == JobStatus.FAILED:
            raise ValidationError(
                f"Job failed: {job.error}",
                details={"job_id": job_id, "error": job.error},
            )

        if job.status != JobStatus.DONE:
            raise ValidationError(
                f"Job not completed. Current status: {job.status.value}",
                details={"job_id": job_id, "status": job.status.value},
            )

        return job

    async def process_pipeline(self, job_id: str) -> None:
        """Run the full processing pipeline for a job.

        Args:
            job_id: The job identifier.
        """
        metrics: dict[str, float] = {}

        try:
            job = await self.repository.get(job_id)
            audio_path = Path(job.audio_path) if job.audio_path else None

            if not audio_path or not audio_path.exists():
                raise ValidationError(f"Audio file not found for job {job_id}")

            # 1. Preprocessing
            logger.info("Starting preprocessing", job_id=job_id)
            await self.repository.update_status(job_id, JobStatus.PREPROCESSING)
            start = time.time()

            # Import here to avoid circular imports
            from app.audio.preprocessing import preprocess_audio

            processed_path = await preprocess_audio(audio_path, job_id)
            metrics["preprocessing_seconds"] = time.time() - start

            # 2. Transcription
            logger.info("Starting transcription", job_id=job_id)
            await self.repository.update_status(job_id, JobStatus.TRANSCRIBING)
            start = time.time()

            if self.whisper_engine is None:
                raise ValidationError("Whisper engine not initialized")

            transcript, segments = await self._transcribe_with_progress(job_id, processed_path)
            metrics["transcription_seconds"] = time.time() - start

            # 3. Postprocessing
            logger.info("Starting postprocessing", job_id=job_id)
            await self.repository.update_status(job_id, JobStatus.POSTPROCESSING)
            start = time.time()

            from app.audio.preprocessing import clean_transcript

            cleaned_transcript = clean_transcript(transcript)
            metrics["postprocessing_seconds"] = time.time() - start

            # 4. Summarization
            logger.info("Starting summarization", job_id=job_id)
            await self.repository.update_status(job_id, JobStatus.SUMMARIZING)
            start = time.time()

            if self.summarizer is None:
                raise ValidationError("Summarizer not initialized")

            summary = await self.summarizer.summarize(cleaned_transcript)
            metrics["summarization_seconds"] = time.time() - start

            # 5. Complete
            result = {
                "transcript": cleaned_transcript,
                "segments": segments,
                "summary": summary.model_dump(),
            }
            await self.repository.complete(job_id, result=result, metrics=metrics)

            logger.info(
                "Pipeline completed",
                job_id=job_id,
                total_seconds=sum(metrics.values()),
                **metrics,
            )

        except Exception as e:
            logger.error("Pipeline failed", job_id=job_id, error=str(e))
            await self.repository.fail(job_id, error=str(e))
            raise

    async def _transcribe_with_progress(
        self, job_id: str, audio_path: Path
    ) -> tuple[str, list[dict]]:
        """Transcribe audio with progress updates.

        Args:
            job_id: The job identifier.
            audio_path: Path to the audio file.

        Returns:
            Tuple of (full transcript text, list of segments with timestamps).
        """
        from app.audio.chunking import chunk_audio

        # Chunk the audio
        chunks = await chunk_audio(
            audio_path,
            chunk_seconds=settings.CHUNK_SECONDS,
            overlap_seconds=settings.CHUNK_OVERLAP_SECONDS,
        )
        total_chunks = len(chunks)

        logger.info(
            "Transcribing chunks",
            job_id=job_id,
            total_chunks=total_chunks,
        )

        transcripts: list[str] = []
        all_segments: list[dict] = []

        for i, chunk in enumerate(chunks):
            # Transcribe chunk with timestamps
            chunk_segments = await self.whisper_engine.transcribe_with_timestamps(chunk.path)

            # Adjust timestamps to real audio timeline (add chunk start time)
            for seg in chunk_segments:
                adjusted_segment = {
                    "start": round(chunk.start_time + seg["start"], 3),
                    "end": round(chunk.start_time + seg["end"], 3),
                    "text": seg["text"],
                }
                all_segments.append(adjusted_segment)

            # Extract text for full transcript
            chunk_text = " ".join(seg["text"] for seg in chunk_segments)
            transcripts.append(chunk_text)

            # Update progress
            await self.repository.update_progress(job_id, i + 1, total_chunks)

            logger.debug(
                "Chunk transcribed",
                job_id=job_id,
                chunk=i + 1,
                total=total_chunks,
            )

        # Merge transcripts (remove duplicates from overlap)
        from app.audio.chunking import merge_transcripts

        merged_transcript = merge_transcripts(transcripts)

        # Remove duplicate segments from overlap regions
        unique_segments = self._deduplicate_segments(all_segments)

        return merged_transcript, unique_segments

    def _deduplicate_segments(self, segments: list[dict]) -> list[dict]:
        """Remove duplicate segments from overlap regions.

        Args:
            segments: List of segments with timestamps.

        Returns:
            Deduplicated list of segments.
        """
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x["start"])

        unique: list[dict] = [sorted_segments[0]]
        for seg in sorted_segments[1:]:
            last = unique[-1]
            # If this segment starts before the last one ends, it's likely overlap
            if seg["start"] >= last["end"] - 0.5:  # 0.5s tolerance
                unique.append(seg)
            elif seg["end"] > last["end"]:
                # Segment extends beyond last one, might have new content
                # Only add if text is substantially different
                if seg["text"].strip() != last["text"].strip():
                    unique.append(seg)

        return unique

    async def _validate_upload(self, audio_file: UploadFile) -> None:
        """Validate an uploaded file.

        Args:
            audio_file: The uploaded file.

        Raises:
            ValidationError: If validation fails.
        """
        if not audio_file.filename:
            raise ValidationError("Filename is required")

        # Check extension
        ext = Path(audio_file.filename).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise UnsupportedFormatError(ext, settings.ALLOWED_EXTENSIONS)

        # Check file size (stream to avoid memory load)
        max_size = settings.MAX_UPLOAD_MB * 1024 * 1024
        size = 0

        await audio_file.seek(0)
        while chunk := await audio_file.read(8192):
            size += len(chunk)
            if size > max_size:
                raise FileTooLargeError(
                    size_mb=size / (1024 * 1024),
                    max_mb=settings.MAX_UPLOAD_MB,
                )

        await audio_file.seek(0)

    async def _calculate_file_hash(self, audio_file: UploadFile) -> str:
        """Calculate SHA-256 hash of a file.

        Args:
            audio_file: The uploaded file.

        Returns:
            The hex-encoded hash.
        """
        hasher = hashlib.sha256()

        await audio_file.seek(0)
        while chunk := await audio_file.read(8192):
            hasher.update(chunk)

        await audio_file.seek(0)
        return hasher.hexdigest()

    async def _save_upload(self, audio_file: UploadFile, job_id: str) -> Path:
        """Save an uploaded file to disk.

        Args:
            audio_file: The uploaded file.
            job_id: The job identifier.

        Returns:
            Path to the saved file.
        """
        ext = Path(audio_file.filename or "audio").suffix.lower()
        file_path = self.work_dir / f"{job_id}{ext}"

        await audio_file.seek(0)
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await audio_file.read(8192):
                await f.write(chunk)

        logger.debug("File saved", job_id=job_id, path=str(file_path))
        return file_path

    async def _upload_to_object_storage(self, file_path: Path, job_id: str) -> str | None:
        """Upload file to object storage and return object key.

        Args:
            file_path: Local path to file.
            job_id: Job identifier.

        Returns:
            Object key if uploaded, else None.
        """
        if self.storage is None:
            return None

        object_key = f"jobs/{job_id}/original{file_path.suffix.lower()}"
        await self.storage.save_file(file_path, object_key)
        logger.info("Uploaded original audio to MinIO", job_id=job_id, object_key=object_key)
        return object_key


# Global service instance
job_service = JobService()
