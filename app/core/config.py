"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API
    APP_NAME: str = "Long-Audio Summary Service"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Model
    MODEL_NAME: str = "openai/whisper-large-v3"
    DEVICE: Literal["cpu", "cuda"] = "cpu"
    COMPUTE_TYPE: str = "int8"  # For faster-whisper: int8, float16, float32

    # Chunking
    CHUNK_SECONDS: int = 30
    CHUNK_OVERLAP_SECONDS: int = 2

    # Limits
    MAX_UPLOAD_MB: int = 500

    # Storage
    JOB_STORAGE_DIR: Path = Path("./data/jobs")
    AUDIO_WORK_DIR: Path = Path("./data/work")

    # Object Storage
    STORAGE_BACKEND: Literal["local", "minio"] = "minio"
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_BUCKET: str = "voice-summary"
    MINIO_SECURE: bool = False

    # Database
    DATABASE_URL: str = (
        "postgresql+asyncpg://voice_summary:voice_summary_password@localhost:5432/voice_summary"
    )

    # Cleanup
    JOB_RETENTION_DAYS: int = 7

    # Allowed audio formats
    ALLOWED_EXTENSIONS: set[str] = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}

    # LLM Summarization (Hugging Face)
    HF_API_TOKEN: str | None = None
    LLM_MODEL_ID: str = "meta-llama/Llama-3.3-70B-Instruct"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
