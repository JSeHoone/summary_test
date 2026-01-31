"""Application error hierarchy."""

from typing import Any


class AppError(Exception):
    """Base application error."""

    def __init__(
        self,
        message: str,
        code: str = "APP_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize application error.

        Args:
            message: Human-readable error message.
            code: Machine-readable error code.
            status_code: HTTP status code.
            details: Additional error details.
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ValidationError(AppError):
    """Validation error for invalid input."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=400,
            details=details,
        )


class NotFoundError(AppError):
    """Resource not found error."""

    def __init__(self, resource: str, resource_id: str) -> None:
        super().__init__(
            message=f"{resource} with id '{resource_id}' not found",
            code="NOT_FOUND",
            status_code=404,
            details={"resource": resource, "resource_id": resource_id},
        )


class JobNotFoundError(NotFoundError):
    """Job not found error."""

    def __init__(self, job_id: str) -> None:
        super().__init__(resource="Job", resource_id=job_id)


class ProcessingError(AppError):
    """Error during audio/transcription processing."""

    def __init__(self, message: str, job_id: str | None = None) -> None:
        super().__init__(
            message=message,
            code="PROCESSING_ERROR",
            status_code=500,
            details={"job_id": job_id} if job_id else {},
        )


class TranscriptionError(ProcessingError):
    """Error during transcription."""

    def __init__(self, message: str, job_id: str | None = None) -> None:
        super().__init__(message=f"Transcription failed: {message}", job_id=job_id)


class SummarizationError(ProcessingError):
    """Error during summarization."""

    def __init__(self, message: str, job_id: str | None = None) -> None:
        super().__init__(message=f"Summarization failed: {message}", job_id=job_id)


class StorageError(AppError):
    """Storage-related error."""

    def __init__(self, message: str, path: str | None = None) -> None:
        super().__init__(
            message=message,
            code="STORAGE_ERROR",
            status_code=500,
            details={"path": path} if path else {},
        )


class FileTooLargeError(ValidationError):
    """File exceeds maximum allowed size."""

    def __init__(self, size_mb: float, max_mb: int) -> None:
        super().__init__(
            message=f"File size ({size_mb:.1f}MB) exceeds maximum allowed ({max_mb}MB)",
            details={"size_mb": size_mb, "max_mb": max_mb},
        )


class UnsupportedFormatError(ValidationError):
    """Unsupported file format."""

    def __init__(self, extension: str, allowed: set[str]) -> None:
        super().__init__(
            message=f"File type '{extension}' not supported. Allowed: {', '.join(sorted(allowed))}",
            details={"extension": extension, "allowed": list(allowed)},
        )
