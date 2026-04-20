"""
FastAPI application entry point for DocForge AI.

Registers all route groups, configures CORS, and manages the application
lifespan (startup / shutdown) for Redis and PostgreSQL connections.

Start the server with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Registered route prefixes:
    /api/rag/*    — RAG ingest, Q&A, evaluation, and cache management
    /api/agent/*  — Tool-calling agent: ticket management and session memory
    /api/*        — DocForge document generation pipeline
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.logger import logger, _setup_logging
from backend.services.redis_service import cache
from backend.core.config import settings
from backend.api.routes import router as docforge_router
from backend.api.rag_routes import router as rag_router
from backend.api.agent_routes import router as agent_router

try:
    from backend.services.db_service import close_pool as _close_pg_pool
except ImportError:
    _close_pg_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown sequencing.

    Startup:
        - Initializes structured logging.
        - Establishes a Redis connection for caching and deduplication.

    Shutdown:
        - Closes the PostgreSQL connection pool (if available).
        - Gracefully disconnects from Redis.

    Args:
        app: The FastAPI application instance passed by the framework.
    """
    _setup_logging()

    try:
        connected = await cache.connect(settings.REDIS_URL)
        if connected:
            logger.info("✅ Redis ready — deduplication and caching active")
        else:
            logger.warning(
                "⚠️  Redis connection failed — ticket deduplication DISABLED. "
                "Verify REDIS_URL in .env."
            )
    except Exception as e:
        logger.error(f"❌ Redis connection error: {e}")
        logger.warning("Agent operations will proceed without caching/deduplication.")

    logger.info("DocForge AI backend started — routes: /api/rag/*, /api/agent/*, /api/*")

    yield

    try:
        if _close_pg_pool:
            await _close_pg_pool()
            logger.info("🐘 PostgreSQL pool closed")
        await cache.disconnect()
        logger.info("DocForge AI backend shutting down")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


app = FastAPI(
    title="DocForge AI + CiteRAG",
    description="AI document generation (DocForge) and RAG Q&A (CiteRAG).",
    version="5.0.0",
    lifespan=lifespan,
)

_ALLOWED_ORIGINS = [o.strip() for o in settings.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(docforge_router, prefix="/api")
app.include_router(rag_router, prefix="/api")
app.include_router(agent_router, prefix="/api")


@app.get("/health", tags=["System"])
async def health():
    """Health check — confirms the service is running and accepting requests."""
    return {"status": "ok", "service": "DocForge AI + CiteRAG"}


@app.get("/", tags=["System"])
async def root():
    """Root endpoint — returns links to the interactive API docs and health check."""
    return {"docs": "/docs", "health": "/health"}
