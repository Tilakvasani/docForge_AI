"""
main.py — FastAPI entry point for DocForge AI + CiteRAG
=========================================================

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Routes registered:
  /api/rag/*    — RAG ingest, ask, status, eval, scores, cache
  /api/agent/*  — LangGraph agent: tickets + memory
  /api/*        — DocForge document generation (routes.py)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.logger import logger

app = FastAPI(
    title="DocForge AI + CiteRAG",
    description="AI document generation (DocForge) and RAG Q&A (CiteRAG) for Turabit",
    version="5.0.0",
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

# ── Agent routes (tickets, memory) — THIS IS WHAT FIXES THE 404 ───────────────
from backend.services.rag.agent_routes import router as agent_router
app.include_router(agent_router, prefix="/api")


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "DocForge AI + CiteRAG"}


@app.get("/", tags=["System"])
async def root():
    return {"docs": "/docs", "health": "/health"}


@app.on_event("startup")
async def startup_event():
    logger.info("DocForge AI backend started — routes: /api/rag/*, /api/agent/*, /api/*")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("DocForge AI backend shutting down")