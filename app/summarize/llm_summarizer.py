"""LLM-based summarizer using Hugging Face Inference API with Llama 3.3 70B."""

import json
import re

from huggingface_hub import InferenceClient

from app.core.config import settings
from app.core.logging import get_logger
from app.jobs.job_models import QuizPoint, SectionSummary, SummaryOutput
from app.summarize.summarizer_interface import SummarizerInterface

logger = get_logger(__name__)

# System prompt for summarization
SYSTEM_PROMPT = """You are an expert summarizer and study assistant. Your task is to analyze transcripts from lectures, meetings, or seminars and create structured, study-friendly summaries.

You must respond ONLY with valid JSON. Do not include any text before or after the JSON object.

IMPORTANT: All content in your response (tldr, outline, key_points, action_items, quiz_points) MUST be written in Korean (한국어)."""

SUMMARIZE_PROMPT = """Analyze the following transcript and create a comprehensive summary.

TRANSCRIPT:
{transcript}

---

Create a JSON response with this EXACT structure:
{{
  "tldr": "A concise 3-5 sentence summary of the main content",
  "outline": [
    {{
      "title": "Section title (e.g., Introduction, Main Topic, Conclusion)",
      "timestamp": "Estimated timestamp in HH:MM:SS format",
      "content": "2-3 sentence summary of this section"
    }}
  ],
  "key_points": [
    "Key point 1 - important insight or fact",
    "Key point 2 - another important point"
  ],
  "action_items": [
    "Action item 1 - tasks or follow-ups mentioned",
    "Action item 2 - homework, deadlines, etc."
  ],
  "quiz_points": [
    {{
      "question": "A study question based on the content",
      "concept": "The underlying concept being tested",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Guidelines:
- tldr: Capture the essence in 3-5 clear sentences
- outline: Create 3-5 logical sections based on content flow
- key_points: Extract 8-15 important facts, insights, or concepts
- action_items: Include any tasks, assignments, deadlines, or follow-ups mentioned (can be empty array if none)
- quiz_points: Generate 5-10 study questions of varying difficulty

CRITICAL: All text content in the JSON values MUST be written in Korean (한국어). Only the JSON keys remain in English.

Respond with ONLY the JSON object, no additional text."""


class LLMSummarizer(SummarizerInterface):
    """LLM-based summarizer using Hugging Face Inference API."""

    def __init__(
        self,
        model_id: str | None = None,
        api_token: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize the LLM summarizer.

        Args:
            model_id: Hugging Face model ID.
            api_token: Hugging Face API token.
            max_tokens: Maximum tokens for generation.
        """
        self.model_id = model_id or settings.LLM_MODEL_ID
        self.api_token = api_token or settings.HF_API_TOKEN
        self.max_tokens = max_tokens

        if not self.api_token:
            raise ValueError(
                "HF_API_TOKEN is required for LLM summarizer. "
                "Set it in .env file or environment variable."
            )

        self.client = InferenceClient(
            model=self.model_id,
            token=self.api_token,
        )

        logger.info(
            "LLM Summarizer initialized",
            model_id=self.model_id,
        )

    async def summarize(self, transcript: str) -> SummaryOutput:
        """Generate a structured summary using LLM.

        Args:
            transcript: The full transcript text.

        Returns:
            Structured summary output.
        """
        if not transcript or not transcript.strip():
            return SummaryOutput(
                tldr="No content available for summarization.",
                outline=[],
                key_points=[],
                action_items=[],
                quiz_points=[],
            )

        logger.info("Generating LLM summary", transcript_length=len(transcript))

        # Truncate transcript if too long (leave room for prompt and response)
        max_transcript_chars = 12000  # Approximate limit for context
        if len(transcript) > max_transcript_chars:
            logger.warning(
                "Transcript truncated for LLM",
                original_length=len(transcript),
                truncated_length=max_transcript_chars,
            )
            transcript = transcript[:max_transcript_chars] + "\n\n[Transcript truncated...]"

        prompt = SUMMARIZE_PROMPT.format(transcript=transcript)

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,  # Lower temperature for more consistent output
            )

            content = response.choices[0].message.content
            logger.debug("LLM response received", response_length=len(content))

            # Parse JSON response
            summary_data = self._parse_json_response(content)

            # Convert to SummaryOutput
            return self._create_summary_output(summary_data)

        except Exception as e:
            logger.error("LLM summarization failed", error=str(e))
            # Fallback to basic summary
            return SummaryOutput(
                tldr=f"Summary generation failed: {str(e)}. Original transcript length: {len(transcript)} characters.",
                outline=[],
                key_points=self._extract_basic_points(transcript),
                action_items=[],
                quiz_points=[],
            )

    async def extract_key_points(self, transcript: str, max_points: int = 15) -> list[str]:
        """Extract key points using LLM.

        Args:
            transcript: The transcript text.
            max_points: Maximum number of key points.

        Returns:
            List of key points.
        """
        # Use the full summarize method and extract key_points
        summary = await self.summarize(transcript)
        return summary.key_points[:max_points]

    async def generate_quiz_points(self, transcript: str, key_points: list[str]) -> list[QuizPoint]:
        """Generate quiz points using LLM.

        Args:
            transcript: The transcript text.
            key_points: Extracted key points.

        Returns:
            List of QuizPoint objects.
        """
        # Quiz points are generated as part of summarize
        summary = await self.summarize(transcript)
        return summary.quiz_points

    def _parse_json_response(self, content: str) -> dict:
        """Parse JSON from LLM response.

        Args:
            content: Raw LLM response content.

        Returns:
            Parsed JSON dictionary.
        """
        # Try direct JSON parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in content
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse LLM response as JSON", content=content[:500])
        return {}

    def _create_summary_output(self, data: dict) -> SummaryOutput:
        """Create SummaryOutput from parsed data.

        Args:
            data: Parsed JSON data.

        Returns:
            SummaryOutput instance.
        """
        # Parse outline
        outline = []
        for item in data.get("outline", []):
            if isinstance(item, dict):
                outline.append(
                    SectionSummary(
                        title=item.get("title", "Section"),
                        timestamp=item.get("timestamp", "00:00:00"),
                        content=item.get("content", ""),
                    )
                )

        # Parse quiz points
        quiz_points = []
        for item in data.get("quiz_points", []):
            if isinstance(item, dict):
                quiz_points.append(
                    QuizPoint(
                        question=item.get("question", ""),
                        concept=item.get("concept", ""),
                        difficulty=item.get("difficulty", "medium"),
                    )
                )

        return SummaryOutput(
            tldr=data.get("tldr", "No summary available."),
            outline=outline,
            key_points=data.get("key_points", []),
            action_items=data.get("action_items", []),
            quiz_points=quiz_points,
        )

    def _extract_basic_points(self, transcript: str, max_points: int = 5) -> list[str]:
        """Extract basic points as fallback.

        Args:
            transcript: The transcript text.
            max_points: Maximum points to extract.

        Returns:
            List of basic extracted points.
        """
        sentences = re.split(r"[.!?]+", transcript)
        points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(points) < max_points:
                points.append(sentence[:200] + ("..." if len(sentence) > 200 else ""))
        return points


def create_llm_summarizer(
    model_id: str | None = None,
    api_token: str | None = None,
) -> LLMSummarizer:
    """Factory function to create LLMSummarizer.

    Args:
        model_id: Hugging Face model ID.
        api_token: Hugging Face API token.

    Returns:
        LLMSummarizer instance.
    """
    return LLMSummarizer(model_id=model_id, api_token=api_token)
