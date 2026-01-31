"""Abstract interface for summarizers."""

from abc import ABC, abstractmethod

from app.jobs.job_models import SummaryOutput


class SummarizerInterface(ABC):
    """Abstract base class for summarization engines."""

    @abstractmethod
    async def summarize(self, transcript: str) -> SummaryOutput:
        """Generate a structured summary from transcript text.

        Args:
            transcript: The full transcript text.

        Returns:
            Structured summary output.
        """
        pass

    @abstractmethod
    async def extract_key_points(self, transcript: str, max_points: int = 15) -> list[str]:
        """Extract key points from transcript.

        Args:
            transcript: The transcript text.
            max_points: Maximum number of key points to extract.

        Returns:
            List of key points.
        """
        pass

    @abstractmethod
    async def generate_quiz_points(self, transcript: str, key_points: list[str]) -> list[dict]:
        """Generate quiz points for study review.

        Args:
            transcript: The transcript text.
            key_points: Extracted key points.

        Returns:
            List of quiz point dictionaries.
        """
        pass
