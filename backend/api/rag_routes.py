"""
FastAPI routes for RAG ingestion, Q&A, evaluation, and cache management.

Provides the HTTP interface to the CiteRAG pipeline. The `/ask` endpoint
is the primary entry point and routes all questions through the LangGraph
agent graph for intent detection and tool execution.

Route prefix: /api/rag/

Endpoints:
    POST   /ingest           — Notion → ChromaDB ingestion pipeline
    POST   /ask              — CiteRAG Q&A with optional streaming
    GET    /status           — ChromaDB collection stats
    DELETE /cache            — Flush all RAG Redis caches
    GET    /scores           — Poll RAGAS scores by key
    POST   /eval             — Manual RAGAS evaluation run
    GET    /eval/runs        — List last 50 RAGAS evaluation runs
    GET    /eval/runs/{id}   — Retrieve a specific RAGAS run snapshot
"""

import asyncio
import datetime
import re
import uuid
from typing import Dict, List, Optional

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.core.logger import logger
from backend.services.redis_service import cache
from backend.rag.rag_service import tool_search, _save_turn, _answer_key
from backend.rag.ragas_scorer import score as ragas_score
from backend.rag.ingest_service import COLLECTION_NAME, ingest_from_notion, _get_collection
from backend.agents.agent_graph import run_agent

router = APIRouter(prefix="/rag", tags=["RAG"])


_SAFE_TOOLS = frozenset({"search", "compare", "multi_compare", "multi_query", "full_doc", "analysis", "refine"})


class IngestRequest(BaseModel):
    """Configuration for the Notion-to-ChromaDB ingestion process."""

    force: bool = False


class AskRequest(BaseModel):
    """
    Request body for the CiteRAG Q&A endpoint.

    Attributes:
        question:   The user's query or instruction (max 2000 characters).
        filters:    Optional key-value filters (e.g. department, doc_type).
        session_id: Unique identifier for conversation history tracking.
        top_k:      Number of chunks to retrieve for single-question tasks.
        doc_a:      First document name for pairwise comparisons.
        doc_b:      Second document name for pairwise comparisons.
        doc_list:   Three or more document names for multi-comparisons.
        stream:     If True, returns tokens as a newline-delimited JSON stream.
        skip_cache: If True, bypasses the Redis answer cache.
    """

    question:   str = Field(..., max_length=2000)
    filters:    Dict[str, str] = Field(default_factory=dict)
    session_id: str = Field("default", max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    top_k:      int = 5
    doc_a:      str = ""
    doc_b:      str = ""
    doc_list:   list[str] = Field(default_factory=list)
    stream:     bool = False
    skip_cache: bool = False

    def sanitized_question(self) -> str:
        """
        Normalize and truncate the user question for safe downstream use.

        Collapses multiple whitespace characters and caps the string at
        2000 characters to prevent oversized prompt injection.

        Returns:
            The normalized question string.
        """
        q = " ".join(self.question.strip().split())
        return q[:2000]


@router.post("/ingest")
async def api_ingest(req: IngestRequest):
    """
    Trigger the Notion → ChromaDB ingestion pipeline.

    If ChromaDB contains zero chunks, ingest is forced automatically
    regardless of the `force` flag (first-run bootstrap). Otherwise,
    the `force` flag controls whether existing chunks are re-indexed.
    """
    try:
        collection  = _get_collection()
        chunk_count = collection.count()
        auto_force  = req.force

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

    Routes questions through the LangGraph agent graph for intent
    detection and tool execution (search, compare, ticket creation, etc.).
    Supports optional streaming for low-latency token delivery.

    Security: If `sanitized_question()` raises HTTP 422, the request is
    treated as a potential injection and a safe deflection response is
    returned without invoking the agent. Azure content filter violations
    are handled similarly.

    Args:
        req: `AskRequest` with question, session, and optional filters.

    Returns:
        A dict response, or a `StreamingResponse` (NDJSON) if `req.stream` is True.
    """
    request_id = str(uuid.uuid4())[:8]

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

    logger.info(
        "📥 [%s] /ask RECEIVED | session=%s | stream=%s | top_k=%d | skip_cache=%s",
        request_id, req.session_id, req.stream, req.top_k, req.skip_cache,
    )
    logger.info("   ↳ User question: %r", question)
    if req.filters:
        logger.info("   ↳ Filters: %s", req.filters)
    if req.doc_a or req.doc_b:
        logger.info("   ↳ Compare docs: %r vs %r", req.doc_a, req.doc_b)
    if req.doc_list:
        logger.info("   ↳ Multi-compare docs: %s", req.doc_list)

    try:
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

        if req.stream:
            stream_queue = asyncio.Queue()

            async def streaming_generator():
                """
                Drain agent output tokens from the queue until the sentinel None is received.

                The agent task always puts None on completion (even on error), so
                the generator is guaranteed to terminate. Errors from the agent task
                are surfaced as a JSON error event rather than silently closing the stream.
                """
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
                        await stream_queue.put(None)

                task = asyncio.create_task(_run_and_sentinel())

                while True:
                    item = await stream_queue.get()
                    if item is None:
                        break
                    yield json.dumps(item) + "\n"

                try:
                    result = task.result()
                except Exception as agent_err:
                    err_msg = str(agent_err)
                    logger.error("❌ [%s] Streaming agent error: %s", request_id, agent_err)
                    yield json.dumps({"type": "error", "message": err_msg}) + "\n"
                    return

                tool_used  = result.get("tool_used", "chat")
                info_found = "could not find" not in result.get("answer", "").lower()

                if tool_used in _SAFE_TOOLS and info_found:
                    await cache.set(a_key, result, ttl=3600)

                yield json.dumps({"type": "done", "result": result}) + "\n"

            return StreamingResponse(streaming_generator(), media_type="application/x-ndjson")

        result = await run_agent(
            question=question,
            session_id=req.session_id,
            doc_a=req.doc_a,
            doc_b=req.doc_b,
            doc_list=req.doc_list
        )

        logger.info("✅ [%s] Done | tool=%s", request_id, result.get("tool_used", "?"))

        tool_used  = result.get("tool_used", "chat")
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
    """Return ChromaDB collection stats including chunk count, doc count, and ingest lock status."""
    try:
        collection   = _get_collection()
        total_chunks = collection.count()
        meta         = await cache.get("docforge:rag:ingest_meta") or {}
        locked       = await cache.exists("docforge:rag:ingest_lock")
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
    """Flush all RAG retrieval, session, and answer caches from Redis."""
    count  = await cache.flush_pattern("docforge:rag:retrieval:*")
    count += await cache.flush_pattern("docforge:rag:session:*")
    count += await cache.flush_pattern("docforge:rag:answer:*")
    return {"flushed": count}


@router.get("/scores")
async def api_get_scores(key: str):
    """Poll for RAGAS evaluation scores by the `ragas_key` returned from `/eval`."""
    if not key or not key.startswith("ragas:"):
        raise HTTPException(status_code=400, detail="Invalid ragas_key format")
    try:
        scores = await cache.get(key)
        return {"key": key, "scores": scores, "ready": scores is not None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EvalRequest(BaseModel):
    """Request body for a manual RAGAS evaluation run."""

    question:     str
    ground_truth: str = ""
    top_k:        int = 15


@router.post("/eval")
async def api_eval(req: EvalRequest):
    """
    Run a manual RAGAS evaluation against the live RAG pipeline.

    Calls `tool_search` directly (bypassing the agent graph) to capture
    the raw retrieval result, then scores it with RAGAS. The full run
    snapshot is stored in Redis for up to 7 days for reproducibility.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info("🧪 [%s] /eval | q=%r", request_id, req.question[:60])
    try:
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
    """Browse the index of the last 50 stored RAGAS evaluation runs."""
    try:
        runs = await cache.get("ragas:run_index") or []
        return {"total": len(runs), "runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eval/runs/{run_id}")
async def api_eval_run_detail(run_id: str):
    """Retrieve the full snapshot of a specific RAGAS evaluation run by its ID."""
    try:
        snapshot = await cache.get(f"ragas:runs:{run_id}")
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        return snapshot
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
