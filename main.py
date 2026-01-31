"""Long-Audio Summary Service - FastAPI Application."""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.job_routes import router as job_router
from app.core.config import settings
from app.core.errors import AppError
from app.core.logging import get_logger, setup_logging
from app.jobs.job_service import job_service
from app.stt.whisper_engine import create_whisper_engine
from app.summarize.llm_summarizer import create_llm_summarizer

# Setup logging
setup_logging(debug=settings.DEBUG)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info(
        "Starting application",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        device=settings.DEVICE,
    )

    # Validate and create required directories
    settings.JOB_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    settings.AUDIO_WORK_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Storage directories initialized")

    # Validate CUDA availability if requested
    if settings.DEVICE == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        # Note: We can't modify settings here, but the engine will handle fallback

    # Load Whisper model
    logger.info("Loading Whisper model...")
    whisper_engine = create_whisper_engine(
        model_name=settings.MODEL_NAME,
        device=settings.DEVICE,
        compute_type=settings.COMPUTE_TYPE,
    )
    job_service.set_whisper_engine(whisper_engine)
    logger.info("Whisper model loaded successfully")

    # Initialize LLM summarizer
    if not settings.HF_API_TOKEN:
        raise RuntimeError(
            "HF_API_TOKEN is required for LLM summarizer. Please set it in your .env file."
        )
    summarizer = create_llm_summarizer(
        api_token=settings.HF_API_TOKEN, model_id=settings.LLM_MODEL_ID
    )
    logger.info("LLM summarizer initialized", model=settings.LLM_MODEL_ID)
    job_service.set_summarizer(summarizer)

    # Store references in app state
    app.state.whisper_engine = whisper_engine
    app.state.summarizer = summarizer

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application")

    # Unload Whisper model to free memory
    if hasattr(app.state, "whisper_engine"):
        app.state.whisper_engine.unload()
        logger.info("Whisper model unloaded")

    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="A FastAPI service for transcribing long-form audio and generating "
    "study-friendly summaries with key points, action items, and quiz points.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle application-specific errors."""
    logger.error(
        "Application error",
        error=exc.message,
        code=exc.code,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": {},
        },
    )


# Include routers
app.include_router(job_router)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Check application health status."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "device": settings.DEVICE,
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


def main():
    """Run the application using uvicorn."""
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
