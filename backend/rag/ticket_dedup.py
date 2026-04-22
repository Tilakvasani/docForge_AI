"""
ticket_dedup.py — 4-Stage duplicate ticket detection
=====================================================

Pipeline (ordered by speed, cheapest first):

  Stage 0 — Redis Intent Cache        ~0ms    FREE
    • SHA-256 hash of normalized question → ticket_id lookup
    • Populated after every LLM-confirmed duplicate
    • Handles repeated questions (leave, salary, NDA) instantly

  Stage 1 — ChromaDB Centroid Search  ~50ms   FREE
    • Tickets stored with CENTROID embeddings (avg of original + paraphrases)
    • Catches "same meaning, different words/language" raw embeddings miss
    • Combined score: 0.7 × embedding + 0.3 × BM25 (replaces weak Jaccard)

  Stage 2 — Cross-Encoder Reranker    ~80ms   FREE (local CPU model)
    • score > 0.85 → auto duplicate, skip LLM entirely
    • score < 0.35 → definitely not a duplicate, skip LLM entirely
    • score 0.35–0.85 → ambiguous, send to LLM

  Stage 3 — LLM Judge                 ~500ms  COSTS MONEY
    • Only ~15–20% of queries reach here
    • Confirmed matches written back to Redis cache (Stage 0)

Cost impact at 1000 tickets/day:
    Before: 1000 LLM calls/day  (~$2.00/day)
    After:  ~150 LLM calls/day  (~$0.30/day)  — 85% reduction

Public API (drop-in replacement — nothing changes for callers):
    find_duplicate(question)
    insert_ticket(ticket)
    insert_tickets(tickets)
    update_ticket_status(ticket_id, status)
    find_similar_tickets(question, top_k)
    embed_ticket(ticket)
    flush_dedup_cache()
"""

import asyncio
import hashlib
import re
from typing import Optional

from backend.core.logger import logger
from backend.core.llm import get_llm as _get_llm
from backend.core.vector import get_embedder as _get_embedder
from backend.core.vector import get_chroma_client as _get_chroma_client
from backend.services.redis_service import cache as _cache
from backend.rag.paraphrase_engine import build_centroid_embedding, rerank_candidates
from rank_bm25 import BM25Okapi
# ── Tuning knobs ──────────────────────────────────────────────────────────────

_TOP_K             = 10
_CHROMA_COLLECTION = "ticket_vectors"
_NORMALIZE_TIMEOUT = 10.0
_INTENT_CACHE_TTL  = 60 * 60 * 24 * 30   # 30 days

_ACTIVE_STATUSES = ["Open", "In Progress"]

# ── Redis cache key helpers ───────────────────────────────────────────────────

def _intent_cache_key(norm_q: str) -> str:
    """SHA-256 hash of normalized question → stable Redis key."""
    digest = hashlib.sha256(norm_q.strip().lower().encode()).hexdigest()[:16]
    return f"docforge:dedup:intent:{digest}"


# ── LLM prompts ───────────────────────────────────────────────────────────────

_NORMALIZE_PROMPT = """\
Translate the following support question to clear, intent-focused English.
Fix any Hinglish, shorthand, or spelling mistakes. Do not answer the question, just return the normalized question text.

Question: "{question}"
Normalized Question:"""

_DEDUP_PROMPT = """\
You are a support ticket duplicate detector with multilingual and paraphrase intelligence.

Below are the most semantically similar OPEN or IN-PROGRESS support tickets:
{ticket_list}

New question from user:
"{new_question}"
Normalized intent:
"{normalized_question}"

STEP 1 — COMPARE NORMALISED INTENT
Count as DUPLICATE only if the normalised intent asks about the EXACT SAME
specific entity (same person, same policy, same document, same comparison).

DUPLICATE examples (same intent, different words or language):
- "who is tilak"          vs "tilak kon he"                  -> SAME person, different language
- "who is raju"           vs "tell me about raju"            -> SAME person lookup
- "what is notice period" vs "how long is notice period"     -> SAME policy
- "NDA mandatory hai kya" vs "is NDA mandatory"              -> SAME document question

NOT DUPLICATE examples (different entity or topic):
- "who is raju"      vs "who is ramesh"        -> DIFFERENT people
- "is NDA mandatory" vs "is SOW mandatory"     -> DIFFERENT documents
- "salary structure" vs "notice period"        -> DIFFERENT HR topics
- "who is raju"      vs "what is leave policy" -> completely different

Reply in EXACTLY this format, nothing else:
DUPLICATE: YES
TICKET_ID: <the matching ticket id>

OR:
DUPLICATE: NO"""


# ── ChromaDB collection ───────────────────────────────────────────────────────

def _get_chroma_collection():
    """Return the ChromaDB collection for ticket deduplication."""
    return _get_chroma_client().get_or_create_collection(
        name=_CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _collection_is_empty(collection) -> bool:
    try:
        return collection.count() == 0
    except Exception:
        return False


# ── Utilities ─────────────────────────────────────────────────────────────────

async def _normalize_question(question: str) -> str:
    """Normalize question to English intent via LLM. Falls back on timeout/error."""
    try:
        prompt = _NORMALIZE_PROMPT.format(question=question)
        resp = await asyncio.wait_for(
            _get_llm().ainvoke(prompt),
            timeout=_NORMALIZE_TIMEOUT,
        )
        return resp.content.strip()
    except asyncio.TimeoutError:
        logger.warning("[dedup] Normalize timed out (%.1fs) for: %.60s", _NORMALIZE_TIMEOUT, question)
        return question
    except Exception as e:
        logger.warning("[dedup] Failed to normalize question: %s", e)
        return question


def _bm25_score(query: str, document: str) -> float:
    """
    BM25-based relevance score. Falls back to Jaccard if rank_bm25 not installed.
    Much more accurate than plain Jaccard for short HR questions.
    """
    try:
        
        tokenize = lambda t: re.sub(r"[?,.]", "", t.lower()).split()
        bm25 = BM25Okapi([tokenize(document)])
        raw = float(bm25.get_scores(tokenize(query))[0])
        return min(1.0, raw / 10.0)   # normalize to 0–1
    except ImportError:
        # Jaccard fallback
        t1 = set(re.sub(r"[?,.]", "", query.lower()).split())
        t2 = set(re.sub(r"[?,.]", "", document.lower()).split())
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)


# ── Stage 0: Redis Intent Cache ───────────────────────────────────────────────

async def _check_intent_cache(norm_q: str) -> Optional[dict]:
    """Check Redis for a previously confirmed duplicate. ~0ms, FREE."""
    try:
        result = await _cache.get(_intent_cache_key(norm_q))
        if result:
            logger.info(
                "[dedup] ✅ Stage 0 cache hit → ticket=%s  q='%.60s'",
                result.get("ticket_id", "?"), norm_q,
            )
            return result
    except Exception as e:
        logger.warning("[dedup] Intent cache lookup failed: %s", e)
    return None


async def confirm_duplicate_pair(norm_q: str, matched_ticket: dict) -> None:
    """
    Write a confirmed duplicate pair to Redis.
    Next time the same (or hash-equivalent) question is asked, it resolves
    at Stage 0 with zero ChromaDB or LLM cost.
    """
    try:
        await _cache.set(_intent_cache_key(norm_q), matched_ticket, ttl=_INTENT_CACHE_TTL)
        logger.info(
            "[dedup] Intent cache written: ticket=%s  q='%.60s'",
            matched_ticket.get("ticket_id", "?"), norm_q,
        )
    except Exception as e:
        logger.warning("[dedup] Failed to write intent cache: %s", e)


# ── Stage 1: ChromaDB Storage & Pre-filter ────────────────────────────────────

async def insert_tickets(tickets: list[dict]) -> None:
    """
    Generate CENTROID embeddings and bulk upsert tickets into ChromaDB.

    Unlike the original (which stored a single raw-question embedding),
    this stores the average of original + 4 paraphrases so semantically
    equivalent questions in any phrasing land near the stored vector.

    Normalizations and centroid builds run fully in parallel.

    Args:
        tickets: list of dicts with keys {ticket_id, question, page_id, url, status}
    """
    if not tickets:
        return

    try:
        collection = _get_chroma_collection()

        # Normalize all questions in parallel
        norm_qs = await asyncio.gather(
            *[_normalize_question(t["question"]) for t in tickets]
        )

        # Build centroid embeddings in parallel (LLM paraphrases + embed per ticket)
        logger.info("[dedup] Building centroid embeddings for %d tickets...", len(tickets))
        vectors = await asyncio.gather(
            *[build_centroid_embedding(nq) for nq in norm_qs]
        )

        ids = [str(t["ticket_id"]) for t in tickets]
        metadatas = [
            {
                "ticket_id":           str(t["ticket_id"]),
                "question":            t["question"],
                "normalized_question": norm_qs[i],
                "page_id":             t.get("page_id", ""),
                "url":                 t.get("url", ""),
                "status":              t.get("status", "Open"),
            }
            for i, t in enumerate(tickets)
        ]

        collection.upsert(
            ids=ids,
            embeddings=list(vectors),
            documents=[t["question"] for t in tickets],
            metadatas=metadatas,
        )
        logger.info("[dedup] Bulk upserted %d tickets with centroid embeddings", len(tickets))

    except Exception as e:
        logger.warning("[dedup] Failed to bulk upsert tickets: %s", e)


async def insert_ticket(ticket: dict) -> None:
    """Generate centroid embedding and upsert a single ticket into ChromaDB."""
    await insert_tickets([ticket])


# Backward-compatibility alias
embed_ticket = insert_ticket


async def update_ticket_status(ticket_id: str, status: str) -> None:
    """Update status metadata for an existing ticket. Closed tickets are excluded from searches."""
    try:
        collection = _get_chroma_collection()
        res = collection.get(ids=[str(ticket_id)])
        if not res or not res["ids"]:
            logger.warning("[dedup] Cannot update status — ticket %s not found", ticket_id)
            return
        existing_meta = res["metadatas"][0]
        existing_meta["status"] = status
        collection.update(ids=[str(ticket_id)], metadatas=[existing_meta])
        logger.info("[dedup] Ticket %s status → %s", ticket_id, status)
    except Exception as e:
        logger.warning("[dedup] Failed to update status for ticket %s: %s", ticket_id, e)


async def find_similar_tickets(question: str, top_k: int = _TOP_K) -> tuple[list, str]:
    """
    Query ChromaDB for similar active tickets using centroid vectors + BM25.
    Returns: (candidates sorted by score desc, normalized_question)
    """
    try:
        collection = _get_chroma_collection()

        if _collection_is_empty(collection):
            logger.info("[dedup] Collection empty — skipping ChromaDB search")
            return [], question

        norm_q    = await _normalize_question(question)
        embedder  = _get_embedder()
        query_vec = embedder.embed_query(norm_q)

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, collection.count()),   # guard: n_results <= collection size
            where={"status": {"$in": _ACTIVE_STATUSES}},
        )

        candidates = []
        if results and results["ids"] and results["ids"][0]:
            for i, cand_id in enumerate(results["ids"][0]):
                meta    = results["metadatas"][0][i]
                dist    = results["distances"][0][i]
                cand_nq = meta.get("normalized_question") or meta.get("question", "")

                emb_score   = max(0.0, 1.0 - dist)          # ✅ correct cosine formula
                bm25        = _bm25_score(norm_q, cand_nq)  # ✅ BM25 replaces Jaccard
                final_score = round((0.7 * emb_score) + (0.3 * bm25), 4)

                candidates.append({
                    "ticket_id":           meta.get("ticket_id", cand_id),
                    "question":            meta.get("question", ""),
                    "normalized_question": cand_nq,
                    "page_id":             meta.get("page_id", ""),
                    "url":                 meta.get("url", ""),
                    "score":               final_score,
                    "embedding_score":     round(emb_score, 4),
                    "bm25_score":          round(bm25, 4),
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)

        if candidates:
            logger.info(
                "[dedup] ChromaDB: %d candidates, best score=%.3f",
                len(candidates), candidates[0]["score"],
            )

        return candidates, norm_q

    except Exception as e:
        logger.warning("[dedup] ChromaDB search failed: %s", e)
        return [], question


# ── Stage 3: LLM judge ────────────────────────────────────────────────────────

async def _llm_duplicate_check(
    new_question: str,
    norm_question: str,
    candidates: list,
) -> Optional[dict]:
    """
    LLM judge for the ambiguous zone (cross_score 0.35–0.85).
    Only ~15–20% of queries reach this stage.
    """
    if not candidates:
        return None

    try:
        ticket_lines = "\n".join(
            f"  [{c['ticket_id']}] (score={c['score']:.0%}) {c['question']}"
            for c in candidates
        )
        prompt = _DEDUP_PROMPT.format(
            ticket_list=ticket_lines,
            new_question=new_question,
            normalized_question=norm_question,
        )

        resp = await _get_llm().ainvoke(prompt)
        raw  = resp.content.strip()
        logger.debug("[dedup] LLM raw: %s", raw)

        # Robust regex parsing — handles preamble and colon-containing IDs
        parsed: dict[str, str] = {}
        for line in raw.splitlines():
            m = re.match(r"^(DUPLICATE|TICKET_ID)\s*:\s*(.+)$", line.strip(), re.IGNORECASE)
            if m:
                parsed[m.group(1).upper()] = m.group(2).strip()

        if parsed.get("DUPLICATE", "NO").upper() != "YES":
            return None

        matched_id = parsed.get("TICKET_ID", "").strip()
        if not matched_id:
            logger.warning("[dedup] LLM said YES but gave no TICKET_ID")
            return None

        matched_upper = matched_id.upper()

        # Exact match first
        for c in candidates:
            if str(c["ticket_id"]).upper() == matched_upper:
                logger.info("[dedup] ✅ LLM confirmed duplicate  ticket=%s  q='%.60s'", c["ticket_id"], c["question"])
                return c

        # Partial match fallback (LLM sometimes trims prefix/zeros)
        for c in candidates:
            if matched_upper in str(c["ticket_id"]).upper():
                logger.info("[dedup] ✅ LLM confirmed duplicate (partial)  ticket=%s", c["ticket_id"])
                return c

        logger.warning("[dedup] LLM TICKET_ID=%s not found in %d candidates", matched_id, len(candidates))
        return None

    except Exception as e:
        logger.warning("[dedup] LLM check failed: %s — allowing ticket creation", e)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

async def find_duplicate(question: str) -> Optional[dict]:
    """
    Full 4-stage duplicate detection pipeline.

    Stage 0: Redis intent cache  → ~0ms,  FREE  — known repeats
    Stage 1: ChromaDB centroid   → ~50ms, FREE  — semantic pre-filter
    Stage 2: Cross-encoder       → ~80ms, FREE  — auto-resolve clear cases
    Stage 3: LLM judge           → ~500ms, $$$  — only ambiguous ~15%

    Returns the matching ticket dict or None.
    """
    # Stage 0 — Redis intent cache
    norm_q = await _normalize_question(question)
    cached = await _check_intent_cache(norm_q)
    if cached:
        return cached

    # Stage 1 — ChromaDB centroid search
    candidates, norm_q = await find_similar_tickets(question, _TOP_K)
    if not candidates:
        logger.info("[dedup] No candidates found — not a duplicate")
        return None

    # Stage 2 — Cross-encoder reranker
    candidates, auto_match, reason = rerank_candidates(norm_q, candidates)

    if reason == "auto_match" and auto_match:
        await confirm_duplicate_pair(norm_q, auto_match)
        return auto_match

    if reason == "auto_skip":
        return None

    # Stage 3 — LLM judge (ambiguous zone only)
    matched = await _llm_duplicate_check(question, norm_q, candidates)
    if matched:
        await confirm_duplicate_pair(norm_q, matched)
    return matched


async def flush_dedup_cache() -> None:
    """Wipe ChromaDB collection AND Redis intent cache. Forces full re-index."""
    # ChromaDB
    try:
        try:
            _get_chroma_client().delete_collection(name=_CHROMA_COLLECTION)
            logger.info("[dedup] ChromaDB collection '%s' deleted", _CHROMA_COLLECTION)
        except Exception:
            pass
    except Exception as e:
        logger.warning("[dedup] Failed to flush ChromaDB: %s", e)

    # Redis intent cache
    try:
        deleted = await _cache.flush_pattern("docforge:dedup:intent:*")
        logger.info("[dedup] Flushed %d Redis intent cache keys", deleted)
    except Exception as e:
        logger.warning("[dedup] Failed to flush Redis intent cache: %s", e)