"""Audio preprocessing utilities."""

import asyncio
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.core.errors import ProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Standard format for Whisper processing
STANDARD_SAMPLE_RATE = 16000
STANDARD_CHANNELS = 1


async def preprocess_audio(audio_path: Path, job_id: str) -> Path:
    """Preprocess audio file to standard format for transcription.

    Converts audio to:
    - WAV format
    - 16kHz sample rate
    - Mono channel

    Args:
        audio_path: Path to the input audio file.
        job_id: Job identifier for logging.

    Returns:
        Path to the preprocessed audio file.

    Raises:
        ProcessingError: If preprocessing fails.
    """
    output_path = audio_path.with_suffix(".processed.wav")

    # If already in correct format, check and potentially skip
    if audio_path.suffix.lower() == ".wav":
        info = await get_audio_info(audio_path)
        if (
            info.get("sample_rate") == STANDARD_SAMPLE_RATE
            and info.get("channels") == STANDARD_CHANNELS
        ):
            logger.info("Audio already in standard format", job_id=job_id)
            return audio_path

    logger.info(
        "Preprocessing audio",
        job_id=job_id,
        input_path=str(audio_path),
        output_path=str(output_path),
    )

    # Use ffmpeg for conversion
    cmd = [
        "ffmpeg",
        "-i",
        str(audio_path),
        "-ar",
        str(STANDARD_SAMPLE_RATE),
        "-ac",
        str(STANDARD_CHANNELS),
        "-y",  # Overwrite output
        str(output_path),
    ]

    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    check=True,
                ),
            )
    except subprocess.CalledProcessError as e:
        logger.error(
            "FFmpeg preprocessing failed",
            job_id=job_id,
            stderr=e.stderr.decode() if e.stderr else None,
        )
        raise ProcessingError(
            f"Audio preprocessing failed: {e.stderr.decode() if e.stderr else 'Unknown error'}",
            job_id=job_id,
        )
    except FileNotFoundError:
        raise ProcessingError(
            "FFmpeg not found. Please install FFmpeg.",
            job_id=job_id,
        )

    logger.info("Preprocessing complete", job_id=job_id, output_path=str(output_path))
    return output_path


async def get_audio_info(audio_path: Path) -> dict:
    """Get audio file information using ffprobe.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dictionary with audio information (duration, sample_rate, channels).
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(audio_path),
    ]

    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: subprocess.run(cmd, capture_output=True, check=True),
            )

        import json

        data = json.loads(result.stdout)

        # Extract audio stream info
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break

        if not audio_stream:
            return {}

        return {
            "duration": float(data.get("format", {}).get("duration", 0)),
            "sample_rate": int(audio_stream.get("sample_rate", 0)),
            "channels": int(audio_stream.get("channels", 0)),
            "codec": audio_stream.get("codec_name"),
        }

    except Exception as e:
        logger.warning("Failed to get audio info", path=str(audio_path), error=str(e))
        return {}


async def get_audio_duration(audio_path: Path) -> float:
    """Get the duration of an audio file in seconds.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds.
    """
    info = await get_audio_info(audio_path)
    return info.get("duration", 0.0)


def clean_transcript(transcript: str) -> str:
    """Clean and normalize transcript text.

    Args:
        transcript: Raw transcript text.

    Returns:
        Cleaned transcript text.
    """
    if not transcript:
        return ""

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", transcript)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Fix common transcription artifacts
    text = re.sub(r"\s+([.,!?])", r"\1", text)  # Remove space before punctuation
    text = re.sub(r"([.,!?])([A-Za-z])", r"\1 \2", text)  # Add space after punctuation

    # Remove repeated punctuation
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"!{2,}", "!", text)

    return text


async def validate_audio_file(audio_path: Path) -> None:
    """Validate that an audio file is readable and has audio content.

    Args:
        audio_path: Path to the audio file.

    Raises:
        ProcessingError: If validation fails.
    """
    if not audio_path.exists():
        raise ProcessingError(f"Audio file not found: {audio_path}")

    info = await get_audio_info(audio_path)

    if not info:
        raise ProcessingError(f"Could not read audio file: {audio_path}")

    if info.get("duration", 0) < 0.1:
        raise ProcessingError(f"Audio file too short or empty: {audio_path}")

    logger.debug(
        "Audio file validated",
        path=str(audio_path),
        duration=info.get("duration"),
        sample_rate=info.get("sample_rate"),
    )
