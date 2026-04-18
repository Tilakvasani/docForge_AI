"""
rag_routes.py — FastAPI routes for RAG + Single Router LLM
============================================================

POST /api/rag/ingest    — Notion → Chunks → Embeddings → ChromaDB
POST /api/rag/ask       — User message → Single Router LLM (agent_graph.run_agent)
GET  /api/rag/status    — Collection stats
DELETE /api/rag/cache   — Flush retrieval cache
GET  /api/rag/scores    — Poll RAGAS scores by key
POST /api/rag/eval      — Manual RAGAS evaluation

Flow for /ask:
  1. sanitized_question() — fast string-match injection guard (HTTP 422 on obvious patterns)
  2. run_agent() — ONE LLM call with full history; picks the right tool; executes it; returns answer
  No pre-RAG run. No separate classify step. No rewrite step.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import asyncio
import datetime
import re
import uuid
from typing import Dict

# ── Third-party ───────────────────────────────────────────────────────────────
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Internal ──────────────────────────────────────────────────────────────────
from backend.core.logger import logger
from backend.services.redis_service import cache
from backend.rag.rag_service import tool_search, _save_turn, _answer_key
from backend.rag.ragas_scorer import score as ragas_score
from backend.rag.ingest_service import COLLECTION_NAME, ingest_from_notion, _get_collection
from backend.agents.agent_graph import run_agent

router = APIRouter(prefix="/rag", tags=["RAG"])


from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# M2 FIX: Single definition of safe (idempotent) tools — only these are cached.
_SAFE_TOOLS = frozenset({"search", "compare", "multi_compare", "multi_query", "full_doc", "analysis", "refine"})


class IngestRequest(BaseModel):
    """Configuration for the Notion ingestion process."""
    force: bool = False  # If True, re-indexes even if chunks exist.


class AskRequest(BaseModel):
    """
    Schema for a CiteRAG question request.
    
    Attributes:
        question: The user's query or instruction.
        filters: Optional department or doc_type filters.
        session_id: Unique ID for conversation history tracking.
        top_k: Number of chunks to retrieve for single tasks.
        doc_a: Name of first document (for comparisons).
        doc_b: Name of second document (for comparisons).
        doc_list: List of 3+ documents for multi-comparisons.
        stream: If True, returns a token stream.
        skip_cache: If True, bypasses the Redis answer cache.
    """
    # H5 FIX: Constrained fields to prevent oversized prompts and Redis key injection
    question:       str = Field(..., max_length=2000)
    filters:        Dict[str, str] = Field(default_factory=dict)
    session_id:     str = Field("default", max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    top_k:          int = 5
    doc_a:          str = ""
    doc_b:          str = ""
    doc_list:       list[str] = Field(default_factory=list)
    stream:         bool = False
    skip_cache:     bool = False

    def sanitized_question(self) -> str:
        """
        Cleans and truncates the user question for safety.
        
        Collapses multiple whitespaces and caps length to prevent 
        DoS-style oversized query attacks.
        
        Returns:
            The normalized question string.
        """
        # ── Fast normalisation ──
        q = " ".join(self.question.strip().split())
        q = q[:2000]
        return q


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/ingest")
async def api_ingest(req: IngestRequest):
    """Trigger Notion → ChromaDB ingest pipeline.
    
    Smart auto-detect:
      - If ChromaDB has 0 chunks → auto-force ingest (first time)
      - If chunks exist → respect the force flag (default=False skips, True re-ingests)
    """
    try:
        # Auto-detect: if no chunks exist, force ingest automatically
        collection = _get_collection()
        chunk_count = collection.count()
        auto_force = req.force
        if chunk_count == 0:
            logger.info("📦 [Ingest] No chunks found in ChromaDB — auto-forcing ingest")
            auto_force = True
        else:
            logger.info("📦 [Ingest] %d chunks already in ChromaDB (force=%s)", chunk_count, req.force)
        
        result  = await ingest_from_notion(force=auto_force)
        flushed = await cache.flush_pattern("docforge:rag:answer:*")
        logger.info("🧹 [Cache] Flushed after ingest (%d keys)", flushed)
        result["existing_chunks"] = chunk_count
        return result
    except Exception as e:
        logger.error("❌ [Ingest] Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def api_ask(req: AskRequest):
    """
    Primary Q&A endpoint for CiteRAG.
    
    Routes questions to the Agent Graph for intent detection and tool
    execution (search, compare, ticket creation, etc.). Supports 
    streaming for low-latency feedback.
    
    Args:
        req: AskRequest containing question, session, and filters.
        
    Returns:
        The dictionary response or a StreamingResponse if req.stream is True.
    """
    request_id = str(uuid.uuid4())[:8]

    # ── Injection guard ───────────────────────────────────────────────────────
    try:
        question = req.sanitized_question()
    except HTTPException as e:
        if e.status_code == 422:
            block_msg = (
                "I could not find information about this in the available documents. "
                "[Note: Request restricted by security policy 🛡️]"
            )
            logger.warning(
                "🛡️ [%s] Injection blocked, returning safe response | session=%s",
                request_id, req.session_id,
            )
            await _save_turn(req.session_id, req.question[:2000], block_msg)
            return {
                "answer":     block_msg,
                "citations":  [],
                "chunks":     [],
                "tool_used":  "chat",
                "confidence": "low",
            }
        raise

    logger.info("🚀 [%s] /ask | session=%s | q=%r", 
                request_id, req.session_id, question[:500])

    try:
        # ── Cache Check ───────────────────────────────────────────────────────
        a_key = _answer_key(question, req.filters)
        if not req.skip_cache:
            hit = await cache.get(a_key)
            if hit:
                logger.info("⚡ [%s] Cache HIT", request_id)
                await _save_turn(req.session_id, question, hit.get("answer", ""))
                
                if req.stream:
                    async def cache_streamer():
                        yield json.dumps({"type": "token", "content": hit.get("answer", "")}) + "\n"
                        yield json.dumps({"type": "done", "result": hit}) + "\n"
                    return StreamingResponse(cache_streamer(), media_type="application/x-ndjson")
                return hit

        # ── Run Agent Graph ───────────────────────────────────────────────────
        if req.stream:
            stream_queue = asyncio.Queue()
            
            async def streaming_generator():
                # H2 FIX: sentinel-based drain — task signals completion by putting None on the queue,
                # eliminating the 0.1s poll loop and its per-token tail latency.
                async def _run_and_sentinel():
                    try:
                        return await run_agent(
                            question=question,
                            session_id=req.session_id,
                            doc_a=req.doc_a,
                            doc_b=req.doc_b,
                            doc_list=req.doc_list,
                            stream_queue=stream_queue,
                        )
                    finally:
                        await stream_queue.put(None)  # sentinel — always sent, even on error

                task = asyncio.create_task(_run_and_sentinel())

                # Drain queue until sentinel
                while True:
                    item = await stream_queue.get()
                    if item is None:
                        break
                    yield json.dumps(item) + "\n"

                # H3 FIX: Catch exceptions from run_agent and yield an error event
                # instead of silently closing the stream.
                try:
                    result = task.result()
                except Exception as agent_err:
                    err_msg = str(agent_err)
                    logger.error("❌ [%s] Streaming agent error: %s", request_id, agent_err)
                    yield json.dumps({"type": "error", "message": err_msg}) + "\n"
                    return

                tool_used = result.get("tool_used", "chat")
                info_found = "could not find" not in result.get("answer", "").lower()

                if tool_used in _SAFE_TOOLS and info_found:
                    await cache.set(a_key, result, ttl=3600)

                yield json.dumps({"type": "done", "result": result}) + "\n"

            return StreamingResponse(streaming_generator(), media_type="application/x-ndjson")
            
        # Pass the full question to the agent graph, which will split it if needed.
        result = await run_agent(
            question=question,
            session_id=req.session_id,
            doc_a=req.doc_a,
            doc_b=req.doc_b,
            doc_list=req.doc_list
        )

        logger.info("✅ [%s] Done | tool=%s", request_id, result.get("tool_used", "?"))
        
        # ── Global Result Cache Save ──────────────────────────────────────────
        # Uses module-level _SAFE_TOOLS frozenset (defined at top of file).
        # NOTE: Do NOT re-define _SAFE_TOOLS as a local variable here — doing so
        # would poison the streaming_generator closure and cause a NameError.
        tool_used = result.get("tool_used", "chat")
        info_found = "could not find" not in result.get("answer", "").lower()
        if tool_used in _SAFE_TOOLS and info_found:
            await cache.set(a_key, result, ttl=3600)
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        err_str = str(e)
        if "content_filter" in err_str or "ResponsibleAIPolicyViolation" in err_str:
            block_msg = (
                "I could not find information about this in the available documents. "
                "[Note: Request restricted by security policy 🛡️]"
            )
            await _save_turn(req.session_id, question, block_msg)
            return {
                "answer":     block_msg,
                "citations":  [],
                "chunks":     [],
                "tool_used":  "chat",
                "confidence": "low",
            }
        logger.error("❌ [%s] Ask error: %s", request_id, err_str)
        raise HTTPException(status_code=500, detail=err_str)


@router.get("/status")
async def api_rag_status():
    """Return ChromaDB collection stats (chunk count, doc count, ingest lock status)."""
    try:
        # FIX: reuse singleton — two PersistentClients on same path = file-lock crash
        collection   = _get_collection()
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
    """Flush all RAG retrieval, session, and answer caches in Redis."""
    count  = await cache.flush_pattern("docforge:rag:retrieval:*")
    count += await cache.flush_pattern("docforge:rag:session:*")
    count += await cache.flush_pattern("docforge:rag:answer:*")
    return {"flushed": count}


@router.get("/scores")
async def api_get_scores(key: str):
    """Poll for RAGAS evaluation scores by a ragas_key returned from /eval."""
    if not key or not key.startswith("ragas:"):
        raise HTTPException(status_code=400, detail="Invalid ragas_key format")
    try:
        scores = await cache.get(key)
        return {"key": key, "scores": scores, "ready": scores is not None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EvalRequest(BaseModel):
    question:     str
    ground_truth: str = ""
    top_k:        int = 15


@router.post("/eval")
async def api_eval(req: EvalRequest):
    """Manual RAGAS evaluation. Stores run snapshot in Redis for reproducibility."""
    request_id = str(uuid.uuid4())[:8]
    logger.info("🧪 [%s] /eval | q=%r", request_id, req.question[:60])
    try:
        # Eval calls search directly — no agent overhead
        rag_result = await tool_search(
            question=req.question, filters={}, session_id="ragas_eval",
        )
        chunks     = rag_result.get("chunks", [])
        rag_answer = rag_result.get("answer", "")

        ragas_scores = None
        ragas_error  = None
        if chunks and rag_answer:
            try:
                ragas_scores = await ragas_score(
                    question=req.question, answer=rag_answer,
                    chunks=chunks, ground_truth=req.ground_truth.strip() or None,
                )
            except Exception as e:
                ragas_error = str(e)
                logger.error("❌ [%s] RAGAS scoring failed: %s", request_id, e)

        run_snapshot = {
            "run_id":    request_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "config":    {"top_k": req.top_k, "collection": "rag_store"},
            "input":     {"question": req.question, "ground_truth": req.ground_truth},
            "output":    {"answer": rag_answer, "chunk_count": len(chunks)},
            "scores":    ragas_scores,
            "error":     ragas_error,
        }
        if not await cache.set(f"ragas:runs:{request_id}", run_snapshot, ttl=604800):
            logger.warning(f"Cache write failed for ragas:runs:{request_id}")
        all_runs = await cache.get("ragas:run_index") or []
        all_runs.insert(0, {"run_id": request_id, "timestamp": run_snapshot["timestamp"], "question": req.question})
        if not await cache.set("ragas:run_index", all_runs[:50], ttl=604800):
            logger.warning("Cache write failed for ragas:run_index")
        logger.info("💾 [%s] Eval run stored (scores=%s)", request_id, ragas_scores is not None)

        return {**rag_result, "ragas_scores": ragas_scores, "ragas_error": ragas_error, "run_id": request_id}
    except Exception as e:
        logger.error("❌ [%s] Eval error: %s", request_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eval/runs")
async def api_eval_runs():
    """Browse the last 50 stored RAGAS evaluation runs."""
    try:
        runs = await cache.get("ragas:run_index") or []
        return {"total": len(runs), "runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eval/runs/{run_id}")
async def api_eval_run_detail(run_id: str):
    """Retrieve the full snapshot of a specific RAGAS evaluation run."""
    try:
        snapshot = await cache.get(f"ragas:runs:{run_id}")
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        return snapshot
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
