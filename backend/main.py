"""
main.py — FastAPI entry point for DocForge AI + CiteRAG
=========================================================

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Routes registered:
  /api/rag/*    — RAG ingest, ask, status, eval, scores, cache
  /api/agent/*  — Tool-calling agent: tickets + memory
  /api/*        — DocForge document generation (routes.py)
"""

# ── Standard library ──────────────────────────────────────────────────────────
from contextlib import asynccontextmanager  # lifespan context manager

# ── Third-party ───────────────────────────────────────────────────────────────
from fastapi import FastAPI                          # Web framework
from fastapi.middleware.cors import CORSMiddleware   # Allow cross-origin requests (Streamlit → backend)

# ── Internal ──────────────────────────────────────────────────────────────────
from backend.core.logger import logger, _setup_logging  # Structured logger
from backend.services.redis_service import cache     # Redis client (dedup, caching, chat history)
from backend.core.config import settings             # App settings loaded from .env
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
    FastAPI lifespan context manager.
    
    Handles startup and shutdown logic for the application:
    - Startup: Initializes logging, connects to Redis.
    - Shutdown: Gracefully disconnects from Redis.
    
    Args:
        app: The FastAPI application instance.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
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

    yield  # ← Application is running here

    # ── Shutdown ──────────────────────────────────────────────────────────────
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

# Derive allowed origins from settings (comma-separated list in .env)
_ALLOWED_ORIGINS = [o.strip() for o in settings.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,   # C1 FIX: explicit origins required when credentials=True
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DocForge generation routes ─────────────────────────────────────────────────
app.include_router(docforge_router, prefix="/api")

# ── RAG routes (ingest, ask, status, eval, scores) ────────────────────────────
app.include_router(rag_router, prefix="/api")

# ── Agent routes (tickets, memory) ────────────────────────────────────────────
app.include_router(agent_router, prefix="/api")


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint — confirms the service is running."""
    return {"status": "ok", "service": "DocForge AI + CiteRAG"}


@app.get("/", tags=["System"])
async def root():
    """Root endpoint — returns links to API docs and health check."""
    return {"docs": "/docs", "health": "/health"}
