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
        # Safety rule 2: flush answer cache so stale answers never served
        flushed = await cache.flush_pattern("docforge:rag:answer:*")
        logger.info("Answer cache flushed after ingest (%d keys)", flushed)
        return result
    except Exception as e:
        logger.error("Ingest error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def api_ask(req: AskRequest):
    try:
        from backend.services.rag.agent_graph import run_agent, _classify_intent

        # ── 1. Classify intent ──────────────────────────────────────────────
        intent = _classify_intent(req.question)

        # ── 2. Standard RAG execution ───────────────────────────────────────
        from backend.services.rag.rag_service import answer

        rag_result = await answer(
            question=req.question,
            filters=req.filters,
            session_id=req.session_id,
            top_k=req.top_k,
            doc_a=req.doc_a,
            doc_b=req.doc_b,
        )

        # ── 3. Agent post-processing: auto-ticket + memory update ────────────
        result = await run_agent(
            question=req.question,
            rag_result=rag_result,
            session_id=req.session_id,
        )
        result["intent"] = "rag"
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
    count += await cache.flush_pattern("docforge:rag:answer:*")
    return {"flushed": count}


@router.get("/scores")
async def api_get_scores(key: str):
    """
    Poll for RAGAS scores by ragas_key returned from /ask.
    Returns scores dict if ready, null if still computing, error string if failed.
    """
    if not key or not key.startswith("ragas:"):
        raise HTTPException(status_code=400, detail="Invalid ragas_key format")
    try:
        scores = await cache.get(key)
        return {"key": key, "scores": scores, "ready": scores is not None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EvalRequest(BaseModel):
    question:     str
    ground_truth: str = ""   # optional — enables context_recall scoring
    top_k:        int = 15


@router.post("/eval")
async def api_eval(req: EvalRequest):
    """
    Manual RAGAS evaluation endpoint.
    Runs RAG then awaits real RAGAS scoring synchronously — scores are
    guaranteed in the response (no background task, no polling needed).
    Use this from the RAGAS tab manual eval panel, NOT from the chat UI.
    """
    try:
        from backend.services.rag.rag_service import answer
        from backend.services.rag.ragas_scorer import score as ragas_score

        # Step 1: get RAG answer + retrieved chunks
        rag_result = await answer(
            question=req.question,
            filters={},
            session_id="ragas_eval",
            top_k=req.top_k,
        )

        chunks      = rag_result.get("chunks", [])
        rag_answer  = rag_result.get("answer", "")

        # Step 2: run real RAGAS synchronously (await — blocks until done)
        ragas_scores = None
        ragas_error  = None
        if chunks and rag_answer:
            try:
                ragas_scores = await ragas_score(
                    question=req.question,
                    answer=rag_answer,
                    chunks=chunks,
                    ground_truth=req.ground_truth.strip() or None,
                )
                logger.info("Eval RAGAS scores: %s", ragas_scores)
            except Exception as e:
                ragas_error = str(e)
                logger.error("RAGAS scoring failed in /eval: %s", e, exc_info=True)
                ragas_scores = None

        return {
            **rag_result,
            "ragas_scores": ragas_scores,
            "ragas_error":  ragas_error,
        }

    except Exception as e:
        logger.error("Eval error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))