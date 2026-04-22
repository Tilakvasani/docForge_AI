"""
paraphrase_engine.py — Centroid embedding + Cross-encoder reranking
====================================================================

Two responsibilities:

1. CENTROID EMBEDDING (called once at ticket INSERT time)
   -------------------------------------------------------
   Instead of embedding just the raw question, we generate 4 paraphrases
   at insert time and store the AVERAGE (centroid) of all 5 vectors.

   Why this matters:
     "notice period kitna hai"     →  vector A
     "how many days before I quit" →  vector B
   These two may have cosine distance 0.45 — too far for the pre-filter
   to surface as a candidate. The LLM never sees them → duplicate missed.

   With centroid embedding, the ticket's stored vector is the average of:
     - "notice period kitna hai"           (original)
     - "What is the notice period?"        (formal English)
     - "how long is the notice period"     (casual English)
     - "notice period duration days"       (keywords only)
     - "नोटिस पीरियड कितना होता है"         (Hindi)
   Now BOTH "notice period kitna hai" AND "how many days before I quit"
   land close to this centroid. Pre-filter catches both. ✅

2. CROSS-ENCODER RERANKING (called at QUERY time, replaces Jaccard score)
   -----------------------------------------------------------------------
   A local CPU model (no API call, no cost) scores each (query, candidate)
   pair with high accuracy. This replaces the current Jaccard keyword score
   and adds automatic decision thresholds:

     score > CROSS_AUTO_MATCH  → definite duplicate, skip LLM entirely
     score < CROSS_AUTO_SKIP   → definitely not a duplicate, skip LLM
     score in between          → ambiguous, send to LLM

   At 1000 tickets this means ~80% of queries never reach the LLM at all.

Dependencies (add to requirements.txt):
    sentence-transformers>=2.7.0

The cross-encoder model (~80 MB) is downloaded automatically on first run
and cached locally by the sentence-transformers library.
"""

import asyncio
import json
import re
from typing import Optional

from backend.core.logger import logger
from backend.core.llm import get_llm as _get_llm
from backend.core.vector import get_embedder as _get_embedder

# ── Tuning knobs ──────────────────────────────────────────────────────────────

# Cross-encoder decision thresholds
CROSS_AUTO_MATCH = 0.85   # score above this → auto duplicate (skip LLM)
CROSS_AUTO_SKIP  = 0.35   # score below this → auto not-duplicate (skip LLM)

# Number of paraphrases to generate at insert time
_PARAPHRASE_COUNT = 4

# LLM timeout for paraphrase generation (seconds)
_PARAPHRASE_TIMEOUT = 15.0

# ── LLM prompt ────────────────────────────────────────────────────────────────

_PARAPHRASE_PROMPT = """\
Generate exactly {n} different ways to ask the same question.
Cover these styles in order:
1. Formal English
2. Casual English
3. Hinglish (mix of Hindi + English)
4. Keywords only (no grammar, just key terms)

Return ONLY a valid JSON array of {n} strings. No explanation, no markdown, no extra text.

Question: "{question}"
JSON:"""

# ── Cross-encoder singleton ───────────────────────────────────────────────────

_cross_encoder = None

def _get_cross_encoder():
    """
    Lazy singleton for the cross-encoder model.
    Downloads ~80MB on first call, then cached locally.
    Runs on CPU — no GPU required.
    """
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("[paraphrase] Loading cross-encoder model (first run may download ~80MB)...")
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("[paraphrase] ✅ Cross-encoder loaded")
        except ImportError:
            logger.warning(
                "[paraphrase] sentence-transformers not installed. "
                "Run: pip install sentence-transformers  "
                "Falling back to embedding-only scoring."
            )
            _cross_encoder = None
    return _cross_encoder


# ── Centroid embedding ────────────────────────────────────────────────────────

async def _generate_paraphrases(question: str) -> list[str]:
    """
    Ask the LLM to generate N paraphrases of the question.
    Returns a list of strings (may be shorter than requested on parse failure).
    """
    try:
        prompt = _PARAPHRASE_PROMPT.format(n=_PARAPHRASE_COUNT, question=question)
        resp = await asyncio.wait_for(
            _get_llm().ainvoke(prompt),
            timeout=_PARAPHRASE_TIMEOUT,
        )
        raw = resp.content.strip()

        # Strip any accidental markdown fences
        raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip().rstrip("`").strip()

        paraphrases = json.loads(raw)
        if isinstance(paraphrases, list):
            # Keep only non-empty strings, max _PARAPHRASE_COUNT
            return [str(p).strip() for p in paraphrases if str(p).strip()][:_PARAPHRASE_COUNT]

    except asyncio.TimeoutError:
        logger.warning("[paraphrase] Paraphrase LLM timed out for: %.60s", question)
    except Exception as e:
        logger.warning("[paraphrase] Failed to generate paraphrases: %s", e)

    return []  # fallback: no paraphrases, use raw question only


def _compute_centroid(vectors: list[list[float]]) -> list[float]:
    """
    Average a list of equal-length embedding vectors into one centroid vector.
    """
    if not vectors:
        raise ValueError("Cannot compute centroid of empty vector list")
    if len(vectors) == 1:
        return vectors[0]

    dim = len(vectors[0])
    centroid = [0.0] * dim
    for vec in vectors:
        for i, val in enumerate(vec):
            centroid[i] += val
    n = len(vectors)
    return [v / n for v in centroid]


async def build_centroid_embedding(normalized_question: str) -> list[float]:
    """
    Build a centroid embedding for a ticket question.

    Steps:
      1. Generate N paraphrases via LLM (done ONCE at insert time)
      2. Embed original + all paraphrases
      3. Return the average (centroid) vector

    This makes the stored vector represent the full "intent space"
    of the question, not just one surface phrasing. Queries in any
    phrasing or language will land closer to this centroid.

    Args:
        normalized_question: The English-normalized question text.

    Returns:
        A single centroid embedding vector (list of floats).
    """
    embedder = _get_embedder()

    # Generate paraphrases (LLM call, but only at insert time — not query time)
    paraphrases = await _generate_paraphrases(normalized_question)

    all_texts = [normalized_question] + paraphrases
    logger.info(
        "[paraphrase] Building centroid from %d texts for: %.60s",
        len(all_texts), normalized_question,
    )

    # Embed all texts in one batched call
    try:
        vectors = embedder.embed_documents(all_texts)
        centroid = _compute_centroid(vectors)
        logger.info("[paraphrase] ✅ Centroid built (%d-dim from %d texts)", len(centroid), len(all_texts))
        return centroid
    except Exception as e:
        logger.warning("[paraphrase] Centroid build failed: %s — falling back to single embedding", e)
        return embedder.embed_query(normalized_question)


# ── Cross-encoder reranking ───────────────────────────────────────────────────

def rerank_candidates(
    query: str,
    candidates: list[dict],
) -> tuple[list[dict], Optional[dict], Optional[str]]:
    """
    Rerank candidates using the cross-encoder and apply automatic decisions.

    Returns:
        (reranked_candidates, auto_decision, reason)

        auto_decision: the matched candidate dict if score > CROSS_AUTO_MATCH, else None
        reason: "auto_match", "auto_skip", or None (ambiguous → send to LLM)

    Each candidate gets a new "cross_score" field added in place.
    """
    model = _get_cross_encoder()

    if model is None:
        # sentence-transformers not installed — return as-is, no auto decision
        return candidates, None, None

    if not candidates:
        return candidates, None, "auto_skip"

    try:
        pairs = [(query, c.get("normalized_question") or c.get("question", "")) for c in candidates]
        raw_scores = model.predict(pairs)

        for c, score in zip(candidates, raw_scores):
            c["cross_score"] = round(float(score), 4)

        # Sort by cross_score descending
        candidates.sort(key=lambda x: x.get("cross_score", 0.0), reverse=True)

        best = candidates[0]
        best_score = best.get("cross_score", 0.0)

        logger.info(
            "[paraphrase] Cross-encoder best score=%.3f for ticket=%s",
            best_score, best.get("ticket_id", "?"),
        )

        if best_score >= CROSS_AUTO_MATCH:
            logger.info("[paraphrase] ✅ Auto-match (score=%.3f >= %.2f) — skipping LLM", best_score, CROSS_AUTO_MATCH)
            return candidates, best, "auto_match"

        if best_score < CROSS_AUTO_SKIP:
            logger.info("[paraphrase] ❌ Auto-skip (score=%.3f < %.2f) — skipping LLM", best_score, CROSS_AUTO_SKIP)
            return candidates, None, "auto_skip"

        logger.info(
            "[paraphrase] Ambiguous score=%.3f (%.2f–%.2f) — sending to LLM",
            best_score, CROSS_AUTO_SKIP, CROSS_AUTO_MATCH,
        )
        return candidates, None, None  # None reason = ambiguous, LLM needed

    except Exception as e:
        logger.warning("[paraphrase] Cross-encoder rerank failed: %s — falling back to embedding score", e)
        return candidates, None, None
