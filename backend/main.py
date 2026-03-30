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
from backend.core.logger import logger               # Structured logger
from backend.services.redis_service import cache     # Redis client (dedup, caching, chat history)
from backend.core.config import settings             # App settings loaded from .env


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager — runs startup logic before yield,
    shutdown logic after yield. Replaces deprecated @app.on_event.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    connected = await cache.connect(settings.REDIS_URL)
    if connected:
        logger.info("✅ Redis ready — deduplication and caching active")
    else:
        logger.warning(
            "⚠️  Redis unavailable — ticket deduplication DISABLED. "
            "Start Redis to prevent duplicate ticket creation."
        )
    logger.info("DocForge AI backend started — routes: /api/rag/*, /api/agent/*, /api/*")

    yield  # ← app is running here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await cache.disconnect()
    logger.info("DocForge AI backend shutting down")


app = FastAPI(
    title="DocForge AI + CiteRAG",
    description="AI document generation (DocForge) and RAG Q&A (CiteRAG) for Turabit",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DocForge generation routes ─────────────────────────────────────────────────
from backend.api.routes import router as docforge_router
app.include_router(docforge_router, prefix="/api")

# ── RAG routes (ingest, ask, status, eval, scores) ────────────────────────────
from backend.services.rag.rag_routes import router as rag_router
app.include_router(rag_router, prefix="/api")

# ── Agent routes (tickets, memory) ────────────────────────────────────────────
from backend.services.rag.agent_routes import router as agent_router
app.include_router(agent_router, prefix="/api")


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint — confirms the service is running."""
    return {"status": "ok", "service": "DocForge AI + CiteRAG"}


@app.get("/", tags=["System"])
async def root():
    """Root endpoint — returns links to API docs and health check."""
    return {"docs": "/docs", "health": "/health"}
