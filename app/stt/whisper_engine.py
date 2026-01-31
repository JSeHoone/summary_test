"""Whisper transcription engine using faster-whisper."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import torch
from faster_whisper import WhisperModel

from app.core.config import settings
from app.core.errors import TranscriptionError
from app.core.logging import get_logger

logger = get_logger(__name__)


class WhisperEngine:
    """Whisper-based speech-to-text engine using faster-whisper."""

    def __init__(
        self,
        model_name: str | None = None,
        device: Literal["cpu", "cuda"] | None = None,
        compute_type: str | None = None,
    ) -> None:
        """Initialize the Whisper engine.

        Args:
            model_name: Name of the Whisper model to use.
            device: Device to run on (cpu or cuda).
            compute_type: Compute type for inference (int8, float16, float32).
        """
        self.model_name = model_name or settings.MODEL_NAME
        self.device = device or settings.DEVICE
        self.compute_type = compute_type or settings.COMPUTE_TYPE

        # Adjust compute type for device
        if self.device == "cpu" and self.compute_type == "float16":
            self.compute_type = "int8"
            logger.warning("float16 not supported on CPU, using int8")

        self._model: WhisperModel | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def model(self) -> WhisperModel:
        """Get the loaded Whisper model."""
        if self._model is None:
            raise TranscriptionError("Whisper model not loaded. Call load() first.")
        return self._model

    def load(self) -> None:
        """Load the Whisper model.

        This should be called once at application startup.
        """
        logger.info(
            "Loading Whisper model",
            model_name=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

        try:
            # Convert model name for faster-whisper
            # "openai/whisper-large-v3" -> "large-v3"
            model_size = self.model_name.split("-")[-1]
            if "large" in self.model_name:
                model_size = "large-v3" if "v3" in self.model_name else "large-v2"
            elif "medium" in self.model_name:
                model_size = "medium"
            elif "small" in self.model_name:
                model_size = "small"
            elif "base" in self.model_name:
                model_size = "base"
            elif "tiny" in self.model_name:
                model_size = "tiny"
            else:
                model_size = self.model_name

            self._model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

            logger.info("Whisper model loaded successfully")

        except Exception as e:
            logger.error("Failed to load Whisper model", error=str(e))
            raise TranscriptionError(f"Failed to load Whisper model: {e}")

    def unload(self) -> None:
        """Unload the Whisper model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Whisper model unloaded")

    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> str:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.
            language: Language code (optional, auto-detected if not provided).

        Returns:
            Transcribed text.

        Raises:
            TranscriptionError: If transcription fails.
        """
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self._transcribe_sync(audio_path, language),
            )
            return result

        except Exception as e:
            # Handle GPU OOM gracefully
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM, attempting CPU fallback")
                if self.device == "cuda":
                    return await self._transcribe_cpu_fallback(audio_path, language)

            logger.error("Transcription failed", path=str(audio_path), error=str(e))
            raise TranscriptionError(str(e))

    def _transcribe_sync(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> str:
        """Synchronous transcription (runs in thread pool).

        Args:
            audio_path: Path to the audio file.
            language: Language code.

        Returns:
            Transcribed text.
        """
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        logger.debug(
            "Transcription info",
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )

        # Collect segments
        texts: list[str] = []
        for segment in segments:
            texts.append(segment.text.strip())

        return " ".join(texts)

    async def _transcribe_cpu_fallback(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> str:
        """Fallback to CPU transcription if GPU fails.

        Args:
            audio_path: Path to the audio file.
            language: Language code.

        Returns:
            Transcribed text.
        """
        logger.info("Using CPU fallback for transcription")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create temporary CPU model
        model_size = "large-v3" if "large-v3" in self.model_name else "large-v2"
        cpu_model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

        try:
            segments, _ = cpu_model.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,
            )

            texts: list[str] = []
            for segment in segments:
                texts.append(segment.text.strip())

            return " ".join(texts)

        finally:
            del cpu_model

    async def transcribe_with_timestamps(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> list[dict]:
        """Transcribe an audio file with word-level timestamps.

        Args:
            audio_path: Path to the audio file.
            language: Language code.

        Returns:
            List of segments with timestamps.
        """
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self._transcribe_with_timestamps_sync(audio_path, language),
            )
            return result

        except Exception as e:
            logger.error(
                "Transcription with timestamps failed",
                path=str(audio_path),
                error=str(e),
            )
            raise TranscriptionError(str(e))

    def _transcribe_with_timestamps_sync(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> list[dict]:
        """Synchronous transcription with timestamps.

        Args:
            audio_path: Path to the audio file.
            language: Language code.

        Returns:
            List of segment dictionaries.
        """
        segments, _ = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            word_timestamps=True,
        )

        result: list[dict] = []
        for segment in segments:
            result.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        }
                        for word in (segment.words or [])
                    ],
                }
            )

        return result


def create_whisper_engine(
    model_name: str | None = None,
    device: Literal["cpu", "cuda"] | None = None,
    compute_type: str | None = None,
) -> WhisperEngine:
    """Factory function to create a WhisperEngine.

    Args:
        model_name: Name of the Whisper model.
        device: Device to use (cpu or cuda).
        compute_type: Compute type for inference.

    Returns:
        Configured WhisperEngine instance.
    """
    engine = WhisperEngine(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
    )
    engine.load()
    return engine
