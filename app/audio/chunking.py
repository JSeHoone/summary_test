"""Audio chunking utilities for processing long audio files."""

import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from app.audio.preprocessing import get_audio_duration
from app.core.config import settings
from app.core.errors import ProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AudioChunk:
    """Represents a chunk of audio."""

    path: Path
    start_time: float  # seconds
    end_time: float  # seconds
    index: int


async def chunk_audio(
    audio_path: Path,
    chunk_seconds: int | None = None,
    overlap_seconds: int | None = None,
) -> list[AudioChunk]:
    """Split audio into fixed-duration chunks with overlap.

    Args:
        audio_path: Path to the audio file.
        chunk_seconds: Duration of each chunk in seconds.
        overlap_seconds: Overlap between chunks in seconds.

    Returns:
        List of AudioChunk objects.
    """
    chunk_seconds = chunk_seconds or settings.CHUNK_SECONDS
    overlap_seconds = overlap_seconds or settings.CHUNK_OVERLAP_SECONDS

    duration = await get_audio_duration(audio_path)
    if duration <= 0:
        raise ProcessingError(f"Invalid audio duration: {duration}")

    # If audio is shorter than chunk duration, return single chunk
    if duration <= chunk_seconds:
        logger.info(
            "Audio shorter than chunk duration, using single chunk",
            duration=duration,
            chunk_seconds=chunk_seconds,
        )
        return [
            AudioChunk(
                path=audio_path,
                start_time=0,
                end_time=duration,
                index=0,
            )
        ]

    # Calculate chunk boundaries
    chunks: list[AudioChunk] = []
    chunk_dir = audio_path.parent / f"{audio_path.stem}_chunks"
    chunk_dir.mkdir(exist_ok=True)

    step = chunk_seconds - overlap_seconds
    start_time = 0.0
    index = 0

    while start_time < duration:
        end_time = min(start_time + chunk_seconds, duration)
        chunk_path = chunk_dir / f"chunk_{index:04d}.wav"

        # Extract chunk using ffmpeg
        await extract_audio_segment(
            audio_path,
            chunk_path,
            start_time,
            end_time - start_time,
        )

        chunks.append(
            AudioChunk(
                path=chunk_path,
                start_time=start_time,
                end_time=end_time,
                index=index,
            )
        )

        start_time += step
        index += 1

        # Safety check to prevent infinite loop
        if index > 10000:
            raise ProcessingError("Too many chunks generated")

    logger.info(
        "Audio chunked",
        total_chunks=len(chunks),
        duration=duration,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )

    return chunks


async def extract_audio_segment(
    input_path: Path,
    output_path: Path,
    start_time: float,
    duration: float,
) -> None:
    """Extract a segment from an audio file.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the extracted segment.
        start_time: Start time in seconds.
        duration: Duration in seconds.
    """
    cmd = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        "-y",
        str(output_path),
    ]

    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                lambda: subprocess.run(cmd, capture_output=True, check=True),
            )
    except subprocess.CalledProcessError as e:
        logger.error(
            "Failed to extract audio segment",
            error=e.stderr.decode() if e.stderr else None,
        )
        raise ProcessingError(
            f"Failed to extract audio segment: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        )


def merge_transcripts(transcripts: list[str], overlap_handling: str = "simple") -> str:
    """Merge transcripts from multiple chunks.

    Args:
        transcripts: List of transcript texts from each chunk.
        overlap_handling: Strategy for handling overlaps ("simple" or "smart").

    Returns:
        Merged transcript text.
    """
    if not transcripts:
        return ""

    if len(transcripts) == 1:
        return transcripts[0]

    if overlap_handling == "simple":
        # Simple concatenation with space
        return " ".join(t.strip() for t in transcripts if t.strip())

    # Smart overlap handling (attempt to detect and remove duplicates)
    merged_parts: list[str] = []

    for i, transcript in enumerate(transcripts):
        if i == 0:
            merged_parts.append(transcript.strip())
            continue

        current = transcript.strip()
        if not current:
            continue

        # Try to find overlap with previous chunk's end
        prev = merged_parts[-1] if merged_parts else ""
        overlap_found = False

        # Check last N words of previous against first N words of current
        prev_words = prev.split()
        curr_words = current.split()

        for overlap_size in range(min(20, len(prev_words), len(curr_words)), 2, -1):
            prev_end = " ".join(prev_words[-overlap_size:])
            curr_start = " ".join(curr_words[:overlap_size])

            if prev_end.lower() == curr_start.lower():
                # Found overlap, skip duplicate words
                merged_parts.append(" ".join(curr_words[overlap_size:]))
                overlap_found = True
                break

        if not overlap_found:
            merged_parts.append(current)

    return " ".join(part for part in merged_parts if part)


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS timestamp.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


async def cleanup_chunks(audio_path: Path) -> None:
    """Clean up chunk files after processing.

    Args:
        audio_path: Path to the original audio file.
    """
    chunk_dir = audio_path.parent / f"{audio_path.stem}_chunks"
    if chunk_dir.exists():
        for chunk_file in chunk_dir.glob("*.wav"):
            try:
                chunk_file.unlink()
            except Exception as e:
                logger.warning("Failed to delete chunk", path=str(chunk_file), error=str(e))
        try:
            chunk_dir.rmdir()
        except Exception as e:
            logger.warning("Failed to delete chunk directory", path=str(chunk_dir), error=str(e))
