"""Pydantic models for job management."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job processing status."""

    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    TRANSCRIBING = "transcribing"
    POSTPROCESSING = "postprocessing"
    SUMMARIZING = "summarizing"
    DONE = "done"
    FAILED = "failed"


class TranscriptSegment(BaseModel):
    """A segment of transcript with timestamp."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")


class QuizPoint(BaseModel):
    """A quiz point for study review."""

    question: str = Field(..., description="Quiz question")
    concept: str = Field(..., description="Underlying concept being tested")
    difficulty: str = Field(..., description="Difficulty level: easy, medium, hard")


class SectionSummary(BaseModel):
    """Summary of a section with timestamp."""

    title: str = Field(..., description="Section title")
    timestamp: str = Field(..., description="Timestamp in format HH:MM:SS")
    content: str = Field(..., description="Section summary content")


class SummaryOutput(BaseModel):
    """Structured summary output."""

    tldr: str = Field(..., description="5-line summary")
    outline: list[SectionSummary] = Field(default_factory=list, description="Section summaries")
    key_points: list[str] = Field(default_factory=list, description="8-15 key bullet points")
    action_items: list[str] = Field(default_factory=list, description="Action items extracted")
    quiz_points: list[QuizPoint] = Field(default_factory=list, description="Study quiz points")


class Job(BaseModel):
    """Internal job representation."""

    id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Current job status")
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage")
    file_hash: str = Field(..., description="SHA-256 hash of uploaded file")
    original_filename: str = Field(..., description="Original uploaded filename")
    audio_path: str | None = Field(default=None, description="Path to processed audio file")
    audio_object_key: str | None = Field(
        default=None, description="Object storage key for original audio"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error: str | None = Field(default=None, description="Error message if failed")
    result: dict[str, Any] | None = Field(default=None, description="Processing result")
    metrics: dict[str, float] = Field(default_factory=dict, description="Processing metrics")


# API Request/Response Models


class JobCreateResponse(BaseModel):
    """Response for job creation."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
                "created_at": "2024-01-30T10:30:00Z",
            }
        }


class JobStatusResponse(BaseModel):
    """Response for job status query."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    error: str | None = Field(default=None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "transcribing",
                "progress": 45,
                "created_at": "2024-01-30T10:30:00Z",
                "updated_at": "2024-01-30T10:35:00Z",
                "error": None,
            }
        }


class JobResultResponse(BaseModel):
    """Response for job result."""

    job_id: str = Field(..., description="Unique job identifier")
    transcript: str = Field(..., description="Full transcript text")
    segments: list[TranscriptSegment] = Field(
        default_factory=list, description="Transcript segments with timestamps"
    )
    summary: SummaryOutput = Field(..., description="Structured summary")
    metrics: dict[str, float] = Field(default_factory=dict, description="Processing metrics")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "transcript": "This is the full transcript of the audio...",
                "segments": [
                    {"start": 0.0, "end": 5.5, "text": "This is the first segment."},
                    {"start": 5.5, "end": 12.3, "text": "This is the second segment."},
                ],
                "summary": {
                    "tldr": "Summary of the content...",
                    "outline": [
                        {
                            "title": "Introduction",
                            "timestamp": "00:00:00",
                            "content": "Overview of the topic...",
                        }
                    ],
                    "key_points": ["Point 1", "Point 2"],
                    "action_items": ["Action 1"],
                    "quiz_points": [
                        {
                            "question": "What is X?",
                            "concept": "Understanding X",
                            "difficulty": "easy",
                        }
                    ],
                },
                "metrics": {
                    "preprocessing_seconds": 2.5,
                    "transcription_seconds": 45.0,
                    "summarization_seconds": 5.0,
                },
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")
