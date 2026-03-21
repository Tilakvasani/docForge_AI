"""
rag_routes.py — FastAPI routes for RAG system

POST /api/rag/ingest    — Notion → Chunks → Embeddings → ChromaDB
POST /api/rag/ask       — Query → Search → LLM → Answer + Citations
GET  /api/rag/status    — Collection stats + cache info
DELETE /api/rag/cache   — Flush retrieval cache
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict

from backend.core.logger import logger
from backend.services.redis_service import cache

router = APIRouter(prefix="/rag", tags=["RAG"])


class IngestRequest(BaseModel):
    force: bool = False


class AskRequest(BaseModel):
    question:   str
    filters:    Dict[str, str] = {}
    session_id: str = "default"
    top_k:      int = 5
    doc_a:      str = ""
    doc_b:      str = ""


@router.post("/ingest")
async def api_ingest(req: IngestRequest):
    try:
        from backend.services.rag.ingest_service import ingest_from_notion
        result = await ingest_from_notion(force=req.force)
        return result
    except Exception as e:
        logger.error("Ingest error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def api_ask(req: AskRequest):
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
    try:
        from backend.services.rag.ingest_service import COLLECTION_NAME
        from backend.core.config import settings
        import chromadb

        client     = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)
        total_chunks = collection.count()

        meta   = await cache.get("docforge:rag:ingest_meta") or {}
        locked = await cache.exists("docforge:rag:ingest_lock")

        return {
            "collection_ok":   True,
            "total_chunks":    total_chunks,
            "total_docs":      meta.get("total_docs", 0),
            "ingest_locked":   locked,
            "redis_available": cache.is_available,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache")
async def api_flush_cache():
    count  = await cache.flush_pattern("docforge:rag:retrieval:*")
    count += await cache.flush_pattern("docforge:rag:session:*")
    return {"flushed": count}