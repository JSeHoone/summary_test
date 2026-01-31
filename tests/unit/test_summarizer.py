"""Unit tests for basic summarizer."""

import pytest

from app.jobs.job_models import SummaryOutput
from app.summarize.basic_summarizer import BasicSummarizer


class TestBasicSummarizer:
    """Tests for BasicSummarizer."""

    @pytest.fixture
    def summarizer(self):
        """Create a summarizer instance."""
        return BasicSummarizer()

    @pytest.fixture
    def sample_transcript(self):
        """Sample transcript for testing."""
        return """
        Today we're going to discuss the importance of machine learning in modern applications.
        Machine learning is a subset of artificial intelligence that enables systems to learn from data.
        The key benefit of machine learning is that it can identify patterns without explicit programming.
        There are three main types of machine learning: supervised, unsupervised, and reinforcement learning.
        Supervised learning uses labeled data to train models for prediction tasks.
        Unsupervised learning finds hidden patterns in data without labels.
        Reinforcement learning trains agents through rewards and penalties.
        In conclusion, machine learning is essential for building intelligent applications.
        Remember to practice implementing these concepts with real datasets.
        Your homework assignment is to build a simple classification model by next week.
        """

    @pytest.mark.asyncio
    async def test_summarize_returns_summary_output(self, summarizer, sample_transcript):
        """Test that summarize returns a SummaryOutput."""
        result = await summarizer.summarize(sample_transcript)

        assert isinstance(result, SummaryOutput)
        assert result.tldr is not None
        assert isinstance(result.key_points, list)
        assert isinstance(result.action_items, list)
        assert isinstance(result.quiz_points, list)
        assert isinstance(result.outline, list)

    @pytest.mark.asyncio
    async def test_summarize_empty_transcript(self, summarizer):
        """Test summarization with empty transcript."""
        result = await summarizer.summarize("")

        assert result.tldr == "No content available for summarization."
        assert len(result.key_points) == 0
        assert len(result.action_items) == 0

    @pytest.mark.asyncio
    async def test_extract_key_points(self, summarizer, sample_transcript):
        """Test key point extraction."""
        key_points = await summarizer.extract_key_points(sample_transcript)

        assert isinstance(key_points, list)
        assert len(key_points) >= 1
        assert all(isinstance(p, str) for p in key_points)

    @pytest.mark.asyncio
    async def test_extract_action_items(self, summarizer, sample_transcript):
        """Test action item extraction."""
        result = await summarizer.summarize(sample_transcript)

        # Should find "homework" and "remember" as action indicators
        assert len(result.action_items) >= 1

    @pytest.mark.asyncio
    async def test_generate_quiz_points(self, summarizer, sample_transcript):
        """Test quiz point generation."""
        key_points = await summarizer.extract_key_points(sample_transcript)
        quiz_points = await summarizer.generate_quiz_points(sample_transcript, key_points)

        assert isinstance(quiz_points, list)
        if quiz_points:
            assert quiz_points[0].question is not None
            assert quiz_points[0].concept is not None
            assert quiz_points[0].difficulty in ["easy", "medium", "hard"]

    def test_split_sentences(self, summarizer):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third question?"
        sentences = summarizer._split_sentences(text)

        assert len(sentences) == 3
        assert "First sentence" in sentences[0]

    def test_format_timestamp(self, summarizer):
        """Test timestamp formatting."""
        assert summarizer._format_timestamp(0) == "00:00:00"
        assert summarizer._format_timestamp(65) == "00:01:05"
        assert summarizer._format_timestamp(3661) == "01:01:01"

    def test_normalize_for_dedup(self, summarizer):
        """Test text normalization for deduplication."""
        text1 = "Hello, World!"
        text2 = "hello world"

        norm1 = summarizer._normalize_for_dedup(text1)
        norm2 = summarizer._normalize_for_dedup(text2)

        assert norm1 == norm2
