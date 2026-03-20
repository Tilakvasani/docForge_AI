"""
rag_routes.py — FastAPI routes for RAG system

POST /api/rag/ingest       — Notion → Chunks → Embeddings → Milvus
POST /api/rag/ask          — Query → Search → LLM → Answer + Citations
GET  /api/rag/status       — Collection stats + cache info
DELETE /api/rag/cache      — Flush retrieval cache
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

from backend.core.logger import logger
from backend.services.redis_service import cache

router = APIRouter(prefix="/rag", tags=["RAG"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    force: bool = False


class AskRequest(BaseModel):
    question:   str
    filters:    Dict[str, str] = {}   # keys: department, doc_type, version
    session_id: str = "default"
    top_k:      int = 5
    doc_a:      str = ""              # for compare tool
    doc_b:      str = ""              # for compare tool


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/ingest")
async def api_ingest(req: IngestRequest):
    """
    Fetch all Notion docs → chunk at headings → embed → store in Milvus.
    Redis prevents re-ingest within 5 min unless force=True.
    """
    try:
        from backend.services.rag.ingest_service import ingest_from_notion
        result = await ingest_from_notion(force=req.force)
        return result
    except Exception as e:
        logger.error("Ingest error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def api_ask(req: AskRequest):
    """
    Query → auto-detect tool (search/refine/compare) → retrieve → LLM → answer.
    Returns: answer, citations, chunks, tool_used, confidence
    """
    try:
        from backend.services.rag.rag_service import answer
        result = await answer(
            question=req.question,
            filters=req.filters,
            session_id=req.session_id,
            top_k=req.top_k,
            doc_a=req.doc_a,
            doc_b=req.doc_b,
        )
        return result
    except Exception as e:
        logger.error("Ask error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def api_rag_status():
    """Milvus collection stats + Redis cache info."""
    try:
        from pymilvus import MilvusClient
        from backend.core.config import settings
        from backend.services.rag.ingest_service import COLLECTION_NAME

        milvus = MilvusClient(uri=settings.MILVUS_URI)
        try:
            stats        = milvus.get_collection_stats(COLLECTION_NAME)
            total_chunks = int(stats.get("row_count", 0))
            ok           = True
        except Exception:
            total_chunks = 0
            ok           = False

        meta   = await cache.get("docforge:rag:ingest_meta") or {}
        locked = await cache.exists("docforge:rag:ingest_lock")

        return {
            "collection_ok":  ok,
            "total_chunks":   total_chunks,
            "total_docs":     meta.get("total_docs", 0),
            "ingest_locked":  locked,
            "redis_available": cache.is_available,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache")
async def api_flush_cache():
    """Flush all RAG retrieval + session cache."""
    count  = await cache.flush_pattern("docforge:rag:retrieval:*")
    count += await cache.flush_pattern("docforge:rag:session:*")
    return {"flushed": count}