"""
ticket_dedup.py — Embedding + LLM duplicate ticket detection
=============================================================

Scalable duplicate detection using ChromaDB and LLM.

Stage 1 — Vector Pre-filter (fast, cheap)
  • Tickets questions are normalized (Hinglish -> English intent)
  • Tickets are embedded once and stored in ChromaDB
  • New question is normalized and embedded
  • ChromaDB similarity finds Top-K candidates fast
  • Combined scoring (0.7 embedding + 0.3 keyword)
  • Top-K (default 10) candidates ALWAYS go to the LLM to avoid missing matches

Stage 2 — LLM Judge (accurate, focused)
  • LLM receives the Top-K candidates
  • Small, focused prompt → fast + cheap
  • Returns YES/NO + matched ticket ID

Public API:
    find_duplicate(question)           — returns a matching ticket dict or None.
    insert_ticket(ticket)              — embed + insert a single new ticket into ChromaDB.
    insert_tickets(tickets)            — batched version for efficient bulk inserts.
    update_ticket_status(ticket_id, status) — update status of a ticket in ChromaDB.
    find_similar_tickets(question, top_k)   — raw query to ChromaDB for similar open/in-progress tickets.
    embed_ticket(ticket)               — alias for insert_ticket for backward compatibility.
    flush_dedup_cache()                — wipe the vector collection (force full re-index).
"""

from typing import Optional
import chromadb
from chromadb.api.types import QueryResult

from backend.core.logger import logger
from backend.core.config import settings
from backend.core.llm import get_llm as _get_llm

# ── Tuning knobs ──────────────────────────────────────────────────────────────

_TOP_K = 10
_CHROMA_COLLECTION = "ticket_vectors"

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

from backend.core.vector import get_embedder as _get_embedder
from backend.core.vector import get_chroma_client as _get_chroma_client


def _get_chroma_collection():
    """Return the ChromaDB collection for ticket deduplication."""
    return _get_chroma_client().get_or_create_collection(
        name=_CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


# ── Utilities ─────────────────────────────────────────────────────────────────

async def _normalize_question(question: str) -> str:
    """Use LLM to normalize a question to English intent."""
    try:
        prompt = _NORMALIZE_PROMPT.format(question=question)
        resp = await _get_llm().ainvoke(prompt)
        return resp.content.strip()
    except Exception as e:
        logger.warning("[dedup] Failed to normalize question: %s", e)
        return question

def _keyword_overlap(text1: str, text2: str) -> float:
    """Compute Jaccard similarity for keyword overlap."""
    t1 = set(text1.lower().replace("?", "").replace(",", "").split())
    t2 = set(text2.lower().replace("?", "").replace(",", "").split())
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)


# ── Stage 1: ChromaDB Storage & Pre-filter ────────────────────────────────────

async def insert_tickets(tickets: list[dict]) -> None:
    """
    Generate embeddings and bulk upsert multiple tickets into ChromaDB.
    
    Args:
        tickets: list of dicts with keys {ticket_id, question, page_id, url, status}
    """
    if not tickets:
        return
        
    try:
        collection = _get_chroma_collection()
        embedder = _get_embedder()
        
        ids = []
        questions = []
        norm_qs = []
        
        for t in tickets:
            ids.append(str(t["ticket_id"]))
            questions.append(t["question"])
            norm = await _normalize_question(t["question"])
            norm_qs.append(norm)
        
        vectors = embedder.embed_documents(norm_qs)
        
        metadatas = []
        for i, t in enumerate(tickets):
            metadatas.append({
                "ticket_id": t["ticket_id"],
                "question": t["question"],
                "normalized_question": norm_qs[i],
                "page_id": t.get("page_id", ""),
                "url": t.get("url", ""),
                "status": t.get("status", "Open"),
            })
            
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=questions,
            metadatas=metadatas
        )
        logger.info("[dedup] Bulk upserted %d tickets into ChromaDB", len(tickets))

    except Exception as e:
        logger.warning("[dedup] Failed to bulk upsert tickets: %s", e)


async def insert_ticket(ticket: dict) -> None:
    """
    Generate embedding and upsert a single ticket into ChromaDB.
    """
    await insert_tickets([ticket])


async def update_ticket_status(ticket_id: str, status: str) -> None:
    """
    Update the status metadata for an existing ticket in ChromaDB.
    Closed tickets will be automatically excluded from future searches.
    """
    try:
        collection = _get_chroma_collection()
        
        res = collection.get(ids=[ticket_id])
        if not res or not res["ids"]:
            logger.warning("[dedup] Cannot update status, ticket %s not found in ChromaDB", ticket_id)
            return
            
        existing_meta = res["metadatas"][0]
        existing_meta["status"] = status
        
        collection.update(
            ids=[ticket_id],
            metadatas=[existing_meta]
        )
        logger.info("[dedup] Updated status for ticket %s to %s", ticket_id, status)
    except Exception as e:
        logger.warning("[dedup] Failed to update status for ticket %s: %s", ticket_id, e)


async def find_similar_tickets(question: str, top_k: int = _TOP_K) -> tuple[list, str]:
    """
    Query ChromaDB for similar tickets, filtering by Status = Open or In Progress.
    Combines embedding and keyword scores to ALWAYS return top_k candidates to LLM.
    
    Returns: (list of candidates, normalized_question)
    """
    try:
        norm_q = await _normalize_question(question)
        collection = _get_chroma_collection()
        embedder = _get_embedder()
        
        query_vec = embedder.embed_query(norm_q)
        
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where={"status": {"$in": ["Open", "In Progress"]}}
        )
        
        # Calculate dynamic threshold for logging purposes only
        word_count = len(norm_q.split())
        dyn_threshold = 0.6 if word_count < 5 else 0.75
        
        candidates = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            
            for i in range(len(ids)):
                meta = metadatas[i]
                cand_norm_q = meta.get("normalized_question", meta.get("question", ""))
                
                emb_score = max(0.0, 1.0 - distances[i])
                kw_score = _keyword_overlap(norm_q, cand_norm_q)
                
                final_score = (0.7 * emb_score) + (0.3 * kw_score)
                
                candidates.append({
                    "ticket_id": meta.get("ticket_id", ids[i]),
                    "question": meta.get("question", ""),
                    "normalized_question": cand_norm_q,
                    "page_id": meta.get("page_id", ""),
                    "url": meta.get("url", ""),
                    "score": round(final_score, 4),
                    "embedding_score": round(emb_score, 4),
                    "keyword_score": round(kw_score, 4),
                })
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        if candidates:
            best = candidates[0]["score"]
            if best < dyn_threshold:
                logger.info("[dedup] Best score %.2f < threshold %.2f, but passing to LLM anyway.", best, dyn_threshold)
            else:
                logger.info("[dedup] Best score %.2f >= threshold %.2f. Passing to LLM.", best, dyn_threshold)
        
        return candidates, norm_q
        
    except Exception as e:
        logger.warning("[dedup] Failed to search similar tickets in ChromaDB: %s", e)
        return [], question


# ── Stage 2: LLM judge ────────────────────────────────────────────────────────

async def _llm_duplicate_check(new_question: str, norm_question: str, candidates: list) -> Optional[dict]:
    """
    Ask the LLM whether `new_question` duplicates any of the Top-K candidates.
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
        logger.debug("[dedup] LLM response: %s", raw)

        lines = {
            line.split(":", 1)[0].strip().upper(): line.split(":", 1)[1].strip()
            for line in raw.splitlines()
            if ":" in line
        }

        if lines.get("DUPLICATE", "NO").upper() != "YES":
            return None

        matched_id = lines.get("TICKET_ID", "").strip().upper()
        if not matched_id:
            return None

        for c in candidates:
            if c["ticket_id"] == matched_id:
                logger.info(
                    "[dedup] ✅ LLM confirmed duplicate  ticket=%s  q='%s'",
                    matched_id, c["question"][:60],
                )
                return c

        logger.warning("[dedup] LLM returned ticket_id=%s but not in candidate list", matched_id)
        return None

    except Exception as e:
        logger.warning("[dedup] LLM check failed: %s — allowing ticket creation", e)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

async def find_duplicate(question: str) -> Optional[dict]:
    """
    Check whether an Open/In-Progress ticket already exists for this question.
    """
    candidates, norm_q = await find_similar_tickets(question, _TOP_K)
    if not candidates:
        logger.info("[dedup] No candidates found in ChromaDB — no duplicate")
        return None

    return await _llm_duplicate_check(question, norm_q, candidates)


async def flush_dedup_cache() -> None:
    """
    Wipe the ticket vector collection from ChromaDB.
    """
    try:
        client = _get_chroma_client()
        try:
            client.delete_collection(name=_CHROMA_COLLECTION)
            logger.info("[dedup] ChromaDB collection %s deleted.", _CHROMA_COLLECTION)
        except Exception:
            pass # Collection doesn't exist
            
    except Exception as e:
        logger.warning("[dedup] Failed to flush ChromaDB collection: %s", e)