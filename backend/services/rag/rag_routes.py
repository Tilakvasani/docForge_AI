"""
rag_routes.py — FastAPI routes for RAG + Tool-Calling Agent
============================================================

POST /api/rag/ingest    — Notion → Chunks → Embeddings → ChromaDB
POST /api/rag/ask       — User message → RAG (if needed) → Tool-calling Agent
GET  /api/rag/status    — Collection stats
DELETE /api/rag/cache   — Flush retrieval cache
GET  /api/rag/scores    — Poll RAGAS scores by key
POST /api/rag/eval      — Manual RAGAS evaluation

Flow for /ask:
  1. Run RAG to get document search result (always, for search tool to use)
  2. Pass everything to run_agent()
  3. Agent's LLM sees full chat history + picks the right tool
  4. Tool executes → response returned

The agent handles ALL routing decisions. This file just feeds it data.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import asyncio    # parallel RAG calls for multi-question inputs
import datetime   # eval run timestamps
import re         # question splitting regex
import uuid       # request trace IDs
from typing import Dict

# ── Third-party ───────────────────────────────────────────────────────────────
import chromadb                                     # vector store client
from fastapi import APIRouter, HTTPException        # FastAPI routing + error responses
from pydantic import BaseModel                      # Request schema validation

# ── Internal ──────────────────────────────────────────────────────────────────
from backend.core.logger import logger                              # Structured logger
from backend.core.config import settings                            # App settings (.env)
from backend.services.redis_service import cache                    # Redis caching layer
from backend.services.rag.rag_service import answer, _save_turn           # Core RAG pipeline
from backend.services.rag.ragas_scorer import score as ragas_score  # RAGAS evaluation scorer
from backend.services.rag.ingest_service import COLLECTION_NAME, ingest_from_notion  # Notion ingest
from backend.services.rag.agent_graph import run_agent              # Tool-calling agent

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

    def sanitized_question(self) -> str:
        """Strip whitespace, collapse internal whitespace, cap at 2000 chars.
        (Prompt injection detection temporarily disabled for LLM testing)
        """
        q = " ".join(self.question.strip().split())
        q = q[:2000]

        # ── Prompt injection detection (DISABLED FOR TESTING) ─────────────────
        # _injection_patterns = [ ... ]
        # q_lower = q.lower()
        # if any(pattern in q_lower for pattern in _injection_patterns):
        #     raise HTTPException(...)

        return q


# ── Multi-question splitting ───────────────────────────────────────────────────

def _split_questions(text: str) -> list[str]:
    """
    Split user input into individual questions (max 5).
    "who is tilak? who is gujar?" → ["who is tilak?", "who is gujar?"]
    "1. who is tilak 2. who is gujar" → ["who is tilak", "who is gujar"]
    """
    text = text.strip()

    numbered = re.findall(r'\d+[\.\\)]\s*(.+?)(?=\s*\d+[\.\\)]|$)', text, re.DOTALL)
    if len(numbered) > 1:
        return [q.strip() for q in numbered if len(q.strip()) > 3][:5]

    if text.count("?") > 1:
        parts = re.split(r'(?<=\?)', text)
        questions = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
        if len(questions) > 1:
            return questions[:5]

    return [text]


async def _process_multi_question(questions: list[str], req: AskRequest) -> dict:
    """Run multiple questions through RAG in parallel."""
    logger.info("🔀 [Multi] Split into %d questions (parallel)", len(questions))

    tasks = [
        answer(
            question=q, filters=req.filters,
            session_id=req.session_id, top_k=req.top_k,
            doc_a=req.doc_a, doc_b=req.doc_b,
        )
        for q in questions
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    found_parts      = []
    unanswered_parts = []
    all_citations    = []

    for q, r in zip(questions, results):
        if isinstance(r, Exception):
            logger.warning("⚠️ [Sub-RAG] error for '%s': %s", q[:40], r)
            unanswered_parts.append({"question": q, "raw_chunks": []})
            continue

        conf      = r.get("confidence", "high")
        ans       = r.get("answer", "")
        not_found = conf == "low" or "could not find" in ans.lower()

        if not_found:
            unanswered_parts.append({
                "question":   q,
                "raw_chunks": r.get("_raw_chunks") or r.get("chunks") or [],
            })
        else:
            found_parts.append({"question": q, "answer": ans, "citations": r.get("citations", [])})
            all_citations.extend(r.get("citations", []))

    sections = [f"**Q: {fp['question']}**\n\n{fp['answer']}" for fp in found_parts]
    combined = "\n\n---\n\n".join(sections)

    return {
        "answer":                combined,
        "confidence":            "low" if unanswered_parts else "high",
        "chunks":                [],
        "citations":             all_citations,
        "tool_used":             "search",
        "_unanswered_questions": unanswered_parts,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/ingest")
async def api_ingest(req: IngestRequest):
    """Trigger Notion → ChromaDB ingest pipeline. Pass force=True to re-ingest all pages."""
    try:
        result  = await ingest_from_notion(force=req.force)
        flushed = await cache.flush_pattern("docforge:rag:answer:*")
        logger.info("🧹 [Cache] Flushed after ingest (%d keys)", flushed)
        return result
    except Exception as e:
        logger.error("❌ [Ingest] Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def api_ask(req: AskRequest):
    """
    Main entry point. Always runs RAG first (gives agent context for search tool),
    then passes everything to the tool-calling agent which decides what to do.
    """
    request_id = str(uuid.uuid4())[:8]
    question   = req.sanitized_question()
    logger.info("🚀 [%s] /ask | session=%s | q=%r", request_id, req.session_id, question[:80])

    try:
        # ── Run RAG (gives search context to the agent's search tool) ─────────
        questions = _split_questions(question)

        if len(questions) == 1:
            rag_result = await answer(
                question=question,
                filters=req.filters,
                session_id=req.session_id,
                top_k=req.top_k,
                doc_a=req.doc_a,
                doc_b=req.doc_b,
            )
        else:
            logger.info("🔀 [%s] Multi-question split into %d parts", request_id, len(questions))
            rag_result = await _process_multi_question(questions, req)

        # ── Agent: one LLM call, picks the right tool ──────────────────────────
        result = await run_agent(
            question=question,
            rag_result=rag_result,
            session_id=req.session_id,
        )
        logger.info("✅ [%s] Finished | tool_used=%s", request_id, result.get("tool_used", "?"))
        return result

    except Exception as e:
        err_str = str(e)
        if "content_filter" in err_str or "ResponsibleAIPolicyViolation" in err_str:
            
            block_msg = "Classified: I am not authorized to disclose internal system configurations, secrets, or execute override commands."
            await _save_turn(req.session_id, question, block_msg)
            
            return {
                "answer": block_msg,
                "citations": [],
                "chunks": [],
                "tool_used": "search",
                "confidence": "low"
            }
        logger.error("❌ [%s] Ask error: %s", request_id, err_str)
        raise HTTPException(status_code=500, detail=err_str)


@router.get("/status")
async def api_rag_status():
    """Return ChromaDB collection stats (chunk count, doc count, ingest lock status)."""
    try:
        client       = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        collection   = client.get_or_create_collection(COLLECTION_NAME)
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
    """Poll for RAGAS evaluation scores by a ragas_key returned from /eval. Returns null until ready."""
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
        rag_result = await answer(
            question=req.question, filters={},
            session_id="ragas_eval", top_k=req.top_k,
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

        # ── Store eval run snapshot for reproducibility ──────────────────────────
        run_snapshot = {
            "run_id":       request_id,
            "timestamp":    datetime.datetime.utcnow().isoformat(),
            "config":       {"top_k": req.top_k, "collection": "rag_store"},
            "input":        {"question": req.question, "ground_truth": req.ground_truth},
            "output":       {"answer": rag_answer, "chunk_count": len(chunks)},
            "scores":       ragas_scores,
            "error":        ragas_error,
        }
        await cache.set(f"ragas:runs:{request_id}", run_snapshot, ttl=604800)  # 7 days
        # Append run_id to index list for browsing all runs
        all_runs = await cache.get("ragas:run_index") or []
        all_runs.insert(0, {"run_id": request_id, "timestamp": run_snapshot["timestamp"], "question": req.question})
        await cache.set("ragas:run_index", all_runs[:50], ttl=604800)  # keep last 50
        logger.info("💾 [%s] Eval run stored (scores=%s)", request_id, ragas_scores is not None)

        return {**rag_result, "ragas_scores": ragas_scores, "ragas_error": ragas_error, "run_id": request_id}
    except Exception as e:
        logger.error("❌ [%s] Eval error: %s", request_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eval/runs")
async def api_eval_runs():
    """Browse the last 50 stored RAGAS evaluation runs (index only, no full output)."""
    try:
        runs = await cache.get("ragas:run_index") or []
        return {"total": len(runs), "runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eval/runs/{run_id}")
async def api_eval_run_detail(run_id: str):
    """Retrieve the full snapshot of a specific RAGAS evaluation run by its run_id."""
    try:
        snapshot = await cache.get(f"ragas:runs:{run_id}")
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found (may have expired)")
        return snapshot
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))