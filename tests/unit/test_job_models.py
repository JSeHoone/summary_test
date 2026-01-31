"""Unit tests for job models."""

from datetime import datetime

import pytest

from app.jobs.job_models import (
    Job,
    JobCreateResponse,
    JobStatus,
    JobStatusResponse,
    QuizPoint,
    SectionSummary,
    SummaryOutput,
)


class TestJobModels:
    """Tests for job Pydantic models."""

    def test_job_creation(self):
        """Test Job model creation with required fields."""
        job = Job(
            id="test-job-123",
            file_hash="abc123def456",
            original_filename="test.mp3",
        )

        assert job.id == "test-job-123"
        assert job.file_hash == "abc123def456"
        assert job.original_filename == "test.mp3"
        assert job.status == JobStatus.QUEUED
        assert job.progress == 0
        assert job.error is None
        assert job.result is None

    def test_job_status_enum(self):
        """Test JobStatus enum values."""
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.PREPROCESSING.value == "preprocessing"
        assert JobStatus.TRANSCRIBING.value == "transcribing"
        assert JobStatus.POSTPROCESSING.value == "postprocessing"
        assert JobStatus.SUMMARIZING.value == "summarizing"
        assert JobStatus.DONE.value == "done"
        assert JobStatus.FAILED.value == "failed"

    def test_job_create_response(self):
        """Test JobCreateResponse model."""
        response = JobCreateResponse(
            job_id="test-123",
            status=JobStatus.QUEUED,
            created_at=datetime(2024, 1, 30, 10, 30, 0),
        )

        assert response.job_id == "test-123"
        assert response.status == JobStatus.QUEUED

    def test_job_status_response(self):
        """Test JobStatusResponse model."""
        now = datetime.utcnow()
        response = JobStatusResponse(
            job_id="test-123",
            status=JobStatus.TRANSCRIBING,
            progress=45,
            created_at=now,
            updated_at=now,
            error=None,
        )

        assert response.job_id == "test-123"
        assert response.status == JobStatus.TRANSCRIBING
        assert response.progress == 45
        assert response.error is None

    def test_summary_output(self):
        """Test SummaryOutput model."""
        summary = SummaryOutput(
            tldr="This is a test summary",
            outline=[
                SectionSummary(
                    title="Introduction",
                    timestamp="00:00:00",
                    content="Overview content",
                )
            ],
            key_points=["Point 1", "Point 2"],
            action_items=["Action 1"],
            quiz_points=[
                QuizPoint(
                    question="What is X?",
                    concept="Understanding X",
                    difficulty="easy",
                )
            ],
        )

        assert summary.tldr == "This is a test summary"
        assert len(summary.outline) == 1
        assert len(summary.key_points) == 2
        assert len(summary.action_items) == 1
        assert len(summary.quiz_points) == 1
        assert summary.quiz_points[0].difficulty == "easy"

    def test_progress_validation(self):
        """Test progress field validation."""
        # Valid progress
        job = Job(
            id="test",
            file_hash="hash",
            original_filename="test.mp3",
            progress=50,
        )
        assert job.progress == 50

        # Progress should be clamped to 0-100
        with pytest.raises(ValueError):
            Job(
                id="test",
                file_hash="hash",
                original_filename="test.mp3",
                progress=150,
            )

        with pytest.raises(ValueError):
            Job(
                id="test",
                file_hash="hash",
                original_filename="test.mp3",
                progress=-10,
            )
