"""
rag_service.py — RAG Tool Library (CiteRAG)
============================================
Pure tool functions — no routing, no intent classification.
All routing is done by the Single Router LLM in agent_graph.py.

Tools:
  tool_search()        — vector search + LLM answer  [Redis answer cache ✅]
  tool_compare()       — 2-document side-by-side comparison
  tool_multi_compare() — 3+ document cross-comparison
  tool_analysis()      — gap/contradiction/audit analysis
  tool_refine()        — HyDE-based summary generation
  tool_full_doc()      — full document retrieval

Redis fix (was broken):
  _answer_key() was defined but never used.  tool_search() now does:
    1. cache.get(answer_key)  → return cached answer immediately
    2. on miss: run LLM, then cache.set(answer_key, result, TTL_ANSWER)
  Retrieval results are still cached via _retrieval_key (unchanged).
"""


import hashlib
import json
import asyncio
from backend.core.config import settings
from backend.core.logger import logger
from backend.services.redis_service import cache


COLLECTION_NAME = "rag_chunks"
MIN_SCORE       = 0.20
TTL_RETRIEVAL   = 600
TTL_SESSION     = 86400
TTL_ANSWER      = 3600


# ── Prompts ───────────────────────────────────────────────────────────────────

ANSWER_PROMPT = """\
You are CiteRAG — a business document assistant for turabit.
Your answers are based strictly on the context provided below.
If the information is not present in the context, say:
"I could not find information about this in the available documents."

{history}

Context:
{context}

Question: {question}

Rules:
- Answer only what is asked — skip unrequested analysis
- Begin directly with the fact or answer — no intro phrases like "Based on the context..."
- Always include specific values from the documents: numbers, names, dates, percentages
- After each key fact, cite the source: [Employee Handbook § Leave Policy]
- Never omit specific values that exist in the context
- Never repeat the question back in your answer
- If the answer requires steps or items, use a numbered list — never inline as prose
- If the context partially answers the question, answer what IS there and state what is missing
- If something is not in the context, say so — never fill gaps with generic knowledge

Output format — match to question type:

Single fact → 1-2 sentences with the exact value + [Document § Section]

What / How / Why → 2-4 sentences with key facts and source citations

List → bullet points, one citation per item

Yes / No → open with YES or NO, then 1-2 supporting sentences from the context

Analysis / Audit → structured sections (Contradictions / Gaps / Ambiguities) only when asked

Answer:"""

COMPARE_PROMPT = """\
You are CiteRAG — a document analyst for turabit.
Compare the two documents on the specific question below.

Question: {question}

Content from {doc_a}:
{content_a}

Content from {doc_b}:
{content_b}

Rules:
- If a document or clause is missing, state it once in one sentence — do not repeat it
- Never say "Document B mirrors Document A" — state each finding with its actual value
- Each document section: 3-5 bullets of specific facts only (numbers, dates, clause wording)
- Comparison table cells must contain specific values — never just "Yes/No" or "Same"
- If a clause is identical in both documents, write the actual shared value in the cell
- GAP IDENTIFIED: skip entirely if there is no real gap — do not invent one
- KEY DIFFERENCE: name the specific clause or value that differs, not a vague category
- SUMMARY: concrete recommendation only — not a restatement of findings already listed

Respond in this format:

FINAL ANSWER
[1-2 sentences. Direct answer to the question. If a document is missing, say so and stop padding.]

DOCUMENT A -- {doc_a}
[3-5 bullets: exact facts, numbers, durations, clause wording. "Clause not present" if missing.]

DOCUMENT B -- {doc_b}
[3-5 bullets: exact facts, numbers, durations, clause wording. "Clause not present" if missing.]

COMPARISON TABLE
| Aspect | {doc_a} | {doc_b} |
|---|---|---|
| [aspect] | [exact finding or "Not present"] | [exact finding or "Not present"] |

KEY DIFFERENCE:
[One sentence: the single most important difference, or "No substantive difference found."]

GAP IDENTIFIED:
What: [specific missing clause or risk — skip if no real gap]
Risk: [one concrete legal/operational impact]
Severity: [🔴 HIGH / 🟡 MEDIUM / 🟢 LOW]

COMPARISON INSIGHT:
Expected: [best practice standard]
Actual: [what was found]
Fix: [one specific action]

SUMMARY: [2 sentences max. Main finding and recommended action.]"""


MULTI_COMPARE_PROMPT = """\
You are CiteRAG — a document analyst for turabit.
Compare ALL the listed documents on the specific question below.

Question: {question}

Documents provided:
{contents}

Rules:
- For each document, give 3-5 bullets of specific facts only (numbers, dates, exact clause wording)
- Never summarise by saying "same as above" — always state the actual value from that document
- Comparison table cells must contain the real value found, never just "Same" or "Yes"
- If a clause is identical across all documents, write the actual shared value in every cell
- GAP IDENTIFIED: skip entirely if there is no real gap — do not invent one
- KEY DIFFERENCE: name the specific clause or value that differs, if any
- FINAL ANSWER: a direct 1-2 sentence yes/no verdict answering the question

Respond in this format:

FINAL ANSWER
[1-2 sentences. Direct answer to the question. Include the specific value (e.g. "30 days") if uniform.]

{doc_sections}

COMPARISON TABLE
| Aspect | {doc_headers} |
|{separator}|
| [aspect] | {doc_cells} |

KEY DIFFERENCE:
[One sentence naming the single most important difference, or "No substantive difference found."]

GAP IDENTIFIED:
[State "None." if there is no real gap, otherwise: What / Risk / Severity]

SUMMARY: [2 sentences max. Main finding and recommended action.]"""



HYDE_PROMPT = """\
Write a brief factual description (2-3 sentences) as if it were a passage from a corporate HR policy,
legal contract, or finance document at a software company. Cover: {question}
Use specific language: include plausible numbers, durations, or conditions where relevant.
Return ONLY the description. No preamble, no conversational filler, no bullet points."""

SUMMARY_PROMPT = """\
You are CiteRAG — a professional document analyst for turabit.
Write a structured, scannable summary. Use ONLY the context below.

Context:
{context}

Topic/Question: {question}

Output format — follow EXACTLY:

SUMMARY
[One sentence: what this document/policy covers and its purpose.]

KEY FUNCTIONS

**1. [Function Name]**
[1-2 sentences. Real facts: names, numbers, conditions, timelines. No vague labels.]

**2. [Function Name]**
[1-2 sentences. Real facts only.]

**3. [Function Name]**
[1-2 sentences. Real facts only.]

(Continue up to 8 functions maximum)

CONCLUSION
[1 sentence. What this document/policy achieves overall.]

RULES:
- Under 220 words total
- No bullet points inside sections
- No intro or outro phrases
- Every section must contain real content — skip if not in context
- If the context is sparse, output a short summary. DO NOT pad with generic industry information
- Short, structured, scannable — not a paragraph essay
- Every KEY FUNCTION must include at least one specific value: a number, name, date, duration, or condition
- Do not repeat the same fact across multiple KEY FUNCTION sections
- CONCLUSION must state the practical outcome for an employee, not just restate the document purpose

Summary:"""


ANALYSIS_PROMPT = """\
You are CiteRAG — a senior legal and business document analyst for turabit.
Analyze the provided documents and answer the question precisely.

CRITICAL DEFINITIONS — apply strictly:

CONTRADICTION: Two statements that CANNOT both be true simultaneously.
  Real example: Doc A says 30-day notice period AND Doc B says 60 days.
  NOT a contradiction: vague wording, different terminology, missing info.

INCONSISTENCY: Same concept, different wording — not logically conflicting.
GAP: A standard clause or section that is completely missing.
AMBIGUITY: Wording that is unclear or interpretable in multiple ways.

Document content:
{context}

Question: {question}

FORMAT — include ONLY sections with actual findings:

FINAL ANSWER
[1-2 sentences. Direct YES/NO or overall verdict answering the question.]

## CONTRADICTIONS
[If none: **No true contradictions found.**]

## INCONSISTENCIES
[Skip if none]

## GAPS
[Skip if none]

## AMBIGUITIES
[Skip if none]

For EACH finding:
- **What:** [specific issue — quote exact wording from document]
  **Where:** [document name] > [section name]
  **Risk:** [concrete legal or operational impact]
  **Severity:** 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
  **Severity Reason:** [1 sentence explaining why this severity level]
  **Fix:** [concrete, actionable recommendation]

## CONCLUSION
[2-3 sentences. Overall assessment: how serious? What is the priority action?]

RULES:
- Always write FINAL ANSWER first before any section headers
- FINAL ANSWER: YES/NO for yes/no questions, or a direct verdict
- Only report what is in the documents — no hallucination
- Cite the exact [Document > Section] for every single finding
- Do not invent or hypothesize risks. State standard operational risks only if clearly linked to a gap
- Cross-check ALL documents, not just the first match
- Flag undefined terms (reasonable, promptly, material breach) as AMBIGUITIES
- Flag missing standard clauses (indemnity, liability cap, force majeure) as GAPS
- CONCLUSION must name the single highest-priority action, not summarise all findings again
- If FINAL ANSWER is YES or NO, the first word of the response must literally be YES or NO
- Each finding must quote the exact clause wording — paraphrasing is not acceptable
- Severity must be justified by a real legal or operational consequence, not assigned by gut feel

Analysis:"""

EXPAND_PROMPT = """\
Generate 3 alternative search queries for the following question.
Each query should use different vocabulary but find the same information.
Return ONLY the 3 queries, one per line, no numbering, no explanation.

Question: {question}"""


# ── Singleton clients (created once, reused) ────────────────────────────────────

_llm_instance = None
_embedder_instance = None
_collection_instance = None


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        from langchain_openai import AzureChatOpenAI
        _llm_instance = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_LLM_ENDPOINT,
            api_key=settings.AZURE_OPENAI_LLM_KEY,
            azure_deployment=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
            api_version="2024-12-01-preview",
            temperature=0.2,
            max_tokens=3000,
        )
    return _llm_instance


def _get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        from langchain_openai import AzureOpenAIEmbeddings
        _embedder_instance = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_EMB_ENDPOINT,
            api_key=settings.AZURE_OPENAI_EMB_KEY,
            azure_deployment=settings.AZURE_EMB_DEPLOYMENT,
            api_version=settings.AZURE_EMB_API_VERSION,
        )
    return _embedder_instance


def _get_collection():
    global _collection_instance
    if _collection_instance is None:
        import chromadb
        client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        _collection_instance = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection_instance


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _retrieval_key(query: str, filters: dict, top_k: int) -> str:
    raw = json.dumps({"q": query, "f": filters, "k": top_k}, sort_keys=True)
    return f"docforge:rag:retrieval:{hashlib.md5(raw.encode()).hexdigest()}"


def _answer_key(question: str, filters: dict) -> str:
    raw = json.dumps({"q": question.strip().lower(), "f": filters}, sort_keys=True)
    return f"docforge:rag:answer:{hashlib.md5(raw.encode()).hexdigest()}"


async def _get_history(session_id: str) -> str:
    """
    Read chat history from the SAME key the agent uses.
    Converts [{role, content}] → formatted string for LLM prompts inside tools.
    """
    AGENT_HISTORY_KEY = "docforge:agent:history:{session_id}"
    data = await cache.get(AGENT_HISTORY_KEY.format(session_id=session_id)) or []
    if not data:
        return ""
    lines = ["Previous conversation:"]
    # data is [{role, content}] — pick last 4 user+assistant pairs
    for msg in data[-8:]:
        role = msg.get("role", "")
        content = (msg.get("content") or "")[:200]
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}...")
    return "\n".join(lines) + "\n"


async def _save_turn(session_id: str, q: str, a: str):
    """
    Write a turn to the SAME key the agent uses (OpenAI message format).
    This keeps tools and the agent in the same conversation store.
    """
    AGENT_HISTORY_KEY = "docforge:agent:history:{session_id}"
    key  = AGENT_HISTORY_KEY.format(session_id=session_id)
    data = await cache.get(key) or []
    data.append({"role": "user",      "content": q})
    data.append({"role": "assistant", "content": a})
    await cache.set(key, data[-40:], ttl=TTL_SESSION)   # keep last 20 turns


# ── Retriever ─────────────────────────────────────────────────────────────────

async def _retrieve_single(query: str, filters: dict, top_k: int,
                            embedder, collection) -> list:
    """Single query retrieval."""
    count = collection.count()
    if count == 0:
        return []

    query_emb = embedder.embed_query(query)

    where = {}
    if filters.get("department"):
        where["department"] = filters["department"]
    if filters.get("doc_type"):
        where["doc_type"] = filters["doc_type"]
    if filters.get("version"):
        where["version"] = filters["version"]

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, count),
        where=where if where else None,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        score = round(1 - dist / 2, 4)
        if score < MIN_SCORE:
            continue
        chunks.append({
            "score":          score,
            "notion_page_id": meta.get("notion_page_id", ""),
            "doc_title":      meta.get("doc_title", ""),
            "doc_type":       meta.get("doc_type", ""),
            "department":     meta.get("department", ""),
            "version":        meta.get("version", ""),
            "heading":        meta.get("heading", ""),
            "content":        doc,
            "citation":       meta.get("citation", ""),
        })
    return chunks


async def _retrieve(query: str, filters: dict, top_k: int = 8) -> list:
    """
    Smart retrieval with query expansion.
    Searches with original + expanded queries, merges and deduplicates results.
    """
    key    = _retrieval_key(query, filters, top_k)
    cached = await cache.get(key)
    if cached is not None:
        return cached

    embedder   = _get_embedder()
    collection = _get_collection()

    # Step 1: search with original query
    all_chunks = await _retrieve_single(query, filters, top_k, embedder, collection)

    # Step 2: expand query with synonyms (only if original returns < 5 results)
    if len(all_chunks) < 5:
        try:
            expanded = _get_llm().invoke(
                EXPAND_PROMPT.format(question=query)
            ).content.strip()
            variants = [v.strip() for v in expanded.splitlines() if v.strip()][:3]
            logger.info("🌿 [Expand] Query expanded to: %s", variants)

            seen_ids = {c["notion_page_id"] + c["heading"] for c in all_chunks}
            # PERF: run variant retrievals in parallel instead of sequential
            variant_results = await asyncio.gather(
                *[_retrieve_single(v, filters, 4, embedder, collection) for v in variants],
                return_exceptions=True,
            )
            for extras in variant_results:
                if isinstance(extras, Exception):
                    continue
                for c in extras:
                    uid = c["notion_page_id"] + c["heading"]
                    if uid not in seen_ids:
                        seen_ids.add(uid)
                        all_chunks.append(c)
        except Exception as e:
            logger.warning("⚠️ [Expand] Failed: %s", e)

    # Step 3: deduplicate and sort by score
    seen, final = set(), []
    for c in sorted(all_chunks, key=lambda x: x["score"], reverse=True):
        uid = c["notion_page_id"] + c["heading"]
        if uid not in seen:
            seen.add(uid)
            final.append(c)

    final = final[:top_k]

    # Diversity check — trigger if ANY single doc contributes > 50% of chunks
    if final:
        doc_counts = {}
        for c in final:
            doc_counts[c["doc_title"]] = doc_counts.get(c["doc_title"], 0) + 1
        dominant_doc = max(doc_counts, key=doc_counts.get)
        dominant_pct = doc_counts[dominant_doc] / len(final)

        if dominant_pct > 0.5:
            unique_titles = {c["doc_title"] for c in final}
            logger.info("🔍 [Retrieve] Low diversity: '%s' = %.0f%% of chunks. Expanding...",
                        dominant_doc, dominant_pct * 100)
            diverse_queries = [
                f"{query} employee handbook policy",
                f"{query} employment contract terms",
                f"{query} vendor agreement conditions",
                f"{query} HR policy procedure",
            ]
            seen = {c["notion_page_id"] + c["heading"] for c in final}
            # PERF: run diversity queries in parallel
            diverse_results = await asyncio.gather(
                *[_retrieve_single(dq, {}, 3, embedder, collection) for dq in diverse_queries],
                return_exceptions=True,
            )
            for extras in diverse_results:
                if isinstance(extras, Exception):
                    continue
                for c in extras:
                    uid = c["notion_page_id"] + c["heading"]
                    if uid not in seen and c["doc_title"] != dominant_doc:
                        seen.add(uid)
                        unique_titles.add(c["doc_title"])
                        final.append(c)
            final = sorted(final, key=lambda x: x["score"], reverse=True)[:top_k]
            logger.info("⚖️ [Retrieve] After diversity filter: %d chunks from %d docs",
                        len(final), len({c["doc_title"] for c in final}))

    # Table recovery
    _table_ref_phrases = [
        "as per the table", "refer to the table", "table below",
        "outlined below", "as follows", "see the table", "per the schedule",
        "listed below", "as detailed below", "entitlement table",
        "schedule below", "the following table",
    ]
    _table_refs_found = any(
        phrase in c["content"].lower()
        for c in final
        for phrase in _table_ref_phrases
    )
    if _table_refs_found:
        logger.info("📊 [Retrieve] Table reference detected — running targeted table recovery")
        _recovery_q = f"{query} table entitlement schedule days carry forward list"
        seen_ids = {c["notion_page_id"] + c["heading"] for c in final}
        extra = await _retrieve_single(_recovery_q, filters, 5, embedder, collection)
        for c in extra:
            uid = c["notion_page_id"] + c["heading"]
            if uid not in seen_ids:
                seen_ids.add(uid)
                final.append(c)
        final = sorted(final, key=lambda x: x["score"], reverse=True)[:top_k + 3]
        logger.info("📊 [Retrieve] After table recovery: %d chunks", len(final))

    logger.info("✅ [Retrieve] Final: %d chunks found for %r", len(final), query[:80])

    # Cache the retrieval result
    await cache.set(key, final, ttl=TTL_RETRIEVAL)
    return final


def _build_context(chunks: list) -> str:
    if not chunks:
        return "No relevant documents found."
    quality = [c for c in chunks if c.get("score", 0) >= 0.20]
    if not quality:
        quality = chunks[:5]
    return "\n\n---\n\n".join(
        f"Source: {c['citation']}\n{c['content']}"
        for c in quality)


def _citations(chunks: list) -> list:
    seen, out = set(), []
    top_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)[:8]
    for c in top_chunks:
        cit     = c.get("citation", "")
        page_id = c.get("notion_page_id", "")
        url     = f"https://www.notion.so/{page_id.replace('-', '')}" if page_id else ""
        dedup_key = c.get("doc_title", "") + "§" + c.get("heading", "")
        if cit and dedup_key not in seen:
            seen.add(dedup_key)
            out.append({"text": cit, "url": url})
    return out


def _confidence(chunks: list) -> str:
    if not chunks:
        return "low"
    avg = sum(c["score"] for c in chunks) / len(chunks)
    return "high" if avg >= 0.60 else "medium" if avg >= 0.40 else "low"


# ── Tools ─────────────────────────────────────────────────────────────────────

async def tool_search(question: str, filters: dict,
                      session_id: str, top_k: int = 8) -> dict:
    # ── FIX: Check Redis answer cache before doing any work ───────────────────
    a_key  = _answer_key(question, filters)
    cached = await cache.get(a_key)
    if cached is not None:
        logger.info("⚡ [Cache HIT] answer for %r", question[:60])
        # node_save_history() in agent_graph handles history persistence
        return cached

    chunks = await _retrieve(question, filters, top_k)

    _hr_policy_signals = [
        "leave", "policy", "handbook", "employee", "hr ", "salary",
        "working hours", "holiday", "benefit", "attendance", "overtime",
        "probation", "notice", "resignation", "reimbursement", "appraisal",
    ]
    q_lower = question.lower()
    _is_hr_policy = any(sig in q_lower for sig in _hr_policy_signals)

    if _is_hr_policy:
        _hr_fallback_queries = [
            f"{question} employee handbook turabit policy",
            f"{question} HR policy leave entitlement days",
            f"{question} employment contract terms conditions",
        ]
        seen       = {c["notion_page_id"] + c["heading"] for c in chunks}
        embedder   = _get_embedder()
        collection = _get_collection()
        # PERF: run HR fallback queries in parallel
        hr_results = await asyncio.gather(
            *[_retrieve_single(fq, {}, 4, embedder, collection) for fq in _hr_fallback_queries],
            return_exceptions=True,
        )
        for extras in hr_results:
            if isinstance(extras, Exception):
                continue
            for c in extras:
                uid = c["notion_page_id"] + c["heading"]
                if uid not in seen:
                    seen.add(uid)
                    chunks.append(c)
        chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)[:top_k + 4]
        logger.info("HR second-pass: now %d chunks from %d docs",
                    len(chunks), len({c["doc_title"] for c in chunks}))

    # Build context (sync, fast) and fetch history in parallel
    context = _build_context(chunks)
    history = await _get_history(session_id)

    # ── Early not-found guard ─────────────────────────────────────────────────
    # If no chunks pass the quality threshold, skip the LLM entirely and return
    # a clean one-liner instead of a padded "not found" essay.
    quality_chunks = [c for c in chunks if c.get("score", 0) >= MIN_SCORE]
    if not quality_chunks:
        not_found_answer = "I could not find information about this in the available documents."
        # node_save_history() in agent_graph handles history persistence
        result = {
            "answer":     not_found_answer,
            "citations":  _citations(chunks),
            "chunks":     chunks,
            "tool_used":  "search",
            "confidence": "low",
        }
        # ── FIX: Caching chunks even for not-found responses ──────────────────
        # This ensures the "Show Sources" button appears even when we 
        # fail to find a confident answer.
        await cache.set(a_key, result, ttl=600)
        return result

    response = await _get_llm().ainvoke(
        ANSWER_PROMPT.format(history=history, context=context, question=question)
    )
    answer = response.content.strip()
    not_found = "could not find" in answer.lower()
    if not_found:
        # LLM said not found but we had chunks — return clean message
        answer = "I could not find information about this in the available documents."
    # node_save_history() in agent_graph handles history persistence

    result = {
        "answer":     answer,
        "citations":  _citations(chunks),
        "chunks":     chunks,
        "tool_used":  "search",
        "confidence": "low" if not_found else _confidence(chunks),
    }

    # ── FIX: Write to Redis answer cache (include chunks for 'Show Sources') ─
    # The 'Show Sources' button in the UI depends on having the 'chunks' key.
    # We now cache the full result so repeated hits still show citations.
    ttl = 600 if not_found else TTL_ANSWER        # shorter TTL for not-found
    await cache.set(a_key, result, ttl=ttl)
    logger.info("💾 [Cache SET] answer for %r (ttl=%ds)", question[:60], ttl)

    return result


async def tool_full_doc(question: str, filters: dict,
                        session_id: str) -> dict:
    """For full document requests — retrieve more chunks with higher top_k."""
    # PERF: run retrieval and history in parallel
    chunks_coro = _retrieve(question, filters, top_k=15)
    history_coro = _get_history(session_id)
    chunks, history = await asyncio.gather(chunks_coro, history_coro)

    context = _build_context(chunks)
    prompt  = ANSWER_PROMPT.format(
        history=history, context=context, question=question)
    response = await _get_llm().ainvoke(prompt)
    answer = response.content.strip()
    not_found = "could not find" in answer.lower()
    # node_save_history() in agent_graph handles history persistence
    return {
        "answer":     answer,
        "citations":  _citations(chunks),
        "chunks":     chunks,
        "tool_used":  "full_doc",
        "confidence": "low" if not_found else _confidence(chunks),
    }


async def tool_refine(question: str, filters: dict,
                      session_id: str, top_k: int = 15) -> dict:
    """HyDE for summaries — generate hypothetical answer first for better retrieval."""
    # PERF: run HyDE generation and history fetch in parallel
    hyp_coro     = asyncio.get_running_loop().run_in_executor(
        None, lambda: _get_llm().invoke(HYDE_PROMPT.format(question=question)).content.strip()
    )
    history_coro = _get_history(session_id)
    hyp, history = await asyncio.gather(hyp_coro, history_coro)

    chunks  = await _retrieve(hyp, filters, top_k)
    context = _build_context(chunks)
    answer  = _get_llm().invoke(
        SUMMARY_PROMPT.format(context=context, question=question)
    ).content.strip()
    not_found = "could not find" in answer.lower()
    # node_save_history() in agent_graph handles history persistence
    return {
        "answer":     answer,
        "citations":  _citations(chunks),
        "chunks":     chunks,
        "tool_used":  "refine",
        "confidence": "low" if not_found else _confidence(chunks),
    }


async def tool_compare(question: str, doc_a: str, doc_b: str,
                       filters: dict, session_id: str, top_k: int = 6) -> dict:

    _boost = "contract agreement clause legal terms obligations"
    query_a = f"{question} {_boost} {doc_a}"
    query_b = f"{question} {_boost} {doc_b}"

    # PERF: already parallel — also fetch history at the same time
    chunks_a, chunks_b, history = await asyncio.gather(
        _retrieve(query_a, filters, top_k * 3),
        _retrieve(query_b, filters, top_k * 3),
        _get_history(session_id),  # PERF: was fetched separately after retrieval
    )

    def filter_doc(chunks, target_title, other_title):
        """Return chunks that match target_title, explicitly excluding other_title."""
        # Step 1: exact match on target
        exact = [c for c in chunks
                 if target_title.lower() in c["doc_title"].lower()
                 and other_title.lower() not in c["doc_title"].lower()]
        if exact:
            return exact[:top_k]
        # Step 2: partial word match on target keywords, still excluding other
        words = [w for w in target_title.lower().split() if len(w) > 3]
        partial = [c for c in chunks
                   if any(w in c["doc_title"].lower() for w in words)
                   and other_title.lower() not in c["doc_title"].lower()]
        if partial:
            return partial[:top_k]
        # Step 3: fallback — best scoring chunks, still excluding the other doc
        excluded = [c for c in chunks
                    if other_title.lower() not in c["doc_title"].lower()]
        return sorted(excluded or chunks, key=lambda x: x["score"], reverse=True)[:top_k]

    chunks_a = filter_doc(chunks_a, doc_a, doc_b)
    chunks_b = filter_doc(chunks_b, doc_b, doc_a)
    content_a = _build_context(chunks_a)
    content_b = _build_context(chunks_b)

    response = await _get_llm().ainvoke(
        COMPARE_PROMPT.format(
            question=question, doc_a=doc_a, doc_b=doc_b,
            content_a=content_a, content_b=content_b)
    )
    raw = response.content.strip()

    def _extract(text, start_tag, end_tags):
        if start_tag not in text:
            return ""
        part = text.split(start_tag, 1)[1]
        for tag in end_tags:
            if tag in part:
                part = part.split(tag, 1)[0]
        return part.strip()

    doc_a_tag = f"DOCUMENT A -- {doc_a}"
    doc_b_tag = f"DOCUMENT B -- {doc_b}"
    if doc_a_tag in raw:
        side_a     = _extract(raw, doc_a_tag, [doc_b_tag, "COMPARISON TABLE", "GAP IDENTIFIED:"])
        side_b     = _extract(raw, doc_b_tag, ["COMPARISON TABLE", "GAP IDENTIFIED:", "KEY DIFFERENCE:", "SYSTEMIC ISSUE", "COMPARISON INSIGHT:", "SUMMARY:"])
        comp_table = _extract(raw, "COMPARISON TABLE", ["GAP IDENTIFIED:", "KEY DIFFERENCE:", "SYSTEMIC ISSUE", "COMPARISON INSIGHT:"])
    else:
        side_a     = _extract(raw, "DOCUMENT_A:", ["DOCUMENT_B:"])
        side_b     = _extract(raw, "DOCUMENT_B:", ["GAP IDENTIFIED:", "KEY DIFFERENCE:", "COMPARISON INSIGHT:", "SUMMARY:"])
        comp_table = ""
    summary = _extract(raw, "SUMMARY:", [])

    if not side_a:
        side_a, side_b, comp_table = content_a[:600], content_b[:600], ""

    all_chunks = chunks_a + chunks_b
    # node_save_history() in agent_graph handles history persistence
    return {
        "answer":      raw,
        "side_a":      side_a,
        "side_b":      side_b,
        "comp_table":  comp_table,
        "summary":     summary,
        "doc_a":      doc_a,
        "doc_b":      doc_b,
        "citations":  _citations(all_chunks),
        "chunks":     all_chunks,
        "tool_used":  "compare",
    }
    


async def tool_multi_compare(
    question: str,
    doc_names: list,
    filters: dict,
    session_id: str,
    top_k: int = 6,
) -> dict:
    """
    Compare N documents against a single question.
    Retrieves chunks per-doc in parallel, then sends one structured prompt.
    """
    _boost = "contract agreement clause legal terms obligations"

    # ── 1. Parallel retrieval for all docs ───────────────────────────────────
    async def _retrieve_for_doc(doc_name: str):
        q = f"{question} {_boost} {doc_name}"
        chunks = await _retrieve(q, filters, top_k * 3)
        # Filter to only chunks from that document
        exact = [c for c in chunks if doc_name.lower() in c["doc_title"].lower()]
        if exact:
            return doc_name, exact[:top_k]
        # Fallback: best-scoring chunks that are not from OTHER named docs
        other_docs = [d.lower() for d in doc_names if d != doc_name]
        excluded = [c for c in chunks
                    if not any(od in c["doc_title"].lower() for od in other_docs)]
        return doc_name, sorted(excluded or chunks,
                                key=lambda x: x["score"], reverse=True)[:top_k]

    results = await asyncio.gather(
        *[_retrieve_for_doc(d) for d in doc_names],
        _get_history(session_id),
        return_exceptions=True,
    )

    history = ""
    doc_chunks_map: dict[str, list] = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        if isinstance(r, str):          # history comes back as str
            history = r
            continue
        doc_name, chunks = r
        doc_chunks_map[doc_name] = chunks

    # ── 2. Build per-doc content blocks ─────────────────────────────────────
    contents_block = []
    all_chunks = []
    for doc_name in doc_names:
        chunks = doc_chunks_map.get(doc_name, [])
        all_chunks.extend(chunks)
        content = _build_context(chunks) if chunks else "No relevant content found for this document."
        contents_block.append(
            f"--- {doc_name} ---\n{content}"
        )

    # ── 3. Build prompt slots ────────────────────────────────────────────────
    doc_headers = " | ".join(doc_names)
    separator   = "|".join(["---"] * (len(doc_names) + 1))
    doc_cells   = " | ".join([f"[{d} value]" for d in doc_names])
    doc_sections = "\n\n".join(
        f"DOCUMENT — {d}\n[3-5 bullets: exact facts, clause wording, numbers from {d} only]"
        for d in doc_names
    )

    prompt = MULTI_COMPARE_PROMPT.format(
        question=question,
        contents="\n\n".join(contents_block),
        doc_sections=doc_sections,
        doc_headers=doc_headers,
        separator=separator,
        doc_cells=doc_cells,
    )

    response = await _get_llm().ainvoke(prompt)
    raw = response.content.strip()

    summary = ""
    if "SUMMARY:" in raw:
        summary = raw.split("SUMMARY:", 1)[1].strip()

    # node_save_history() in agent_graph handles history persistence
    return {
        "answer":      raw,
        "summary":     summary,
        "doc_names":   doc_names,
        "citations":   _citations(all_chunks),
        "chunks":      all_chunks,
        "tool_used":   "multi_compare",
        "confidence":  _confidence(all_chunks),
    }


async def tool_analysis(question: str, filters: dict,
                        session_id: str) -> dict:
    """
    For analysis questions — retrieve broad context then reason over it.
    """
    q_lower = question.lower()

    legal_keywords = [
        "notice period", "termination", "liability", "indemnity", "clause",
        "contract", "agreement", "confidential", "penalty", "enforce",
        "jurisdiction", "governing law", "dispute", "breach", "exit",
        "payment", "fee", "tax", "ip", "intellectual property", "compliance",
        "definition", "term", "defined", "enforceable", "void", "fair",
        "one-sided", "clause", "obligation", "role", "responsibility",
        "hierarchy", "precedence", "structure", "best practice", "scale",
    ]
    is_legal = any(kw in q_lower for kw in legal_keywords)

    if is_legal:
        contract_boost = "contract agreement clause legal terms obligations termination"
        primary_query = f"{question} {contract_boost}"
    else:
        primary_query = question

    # PERF: run primary retrieval and history fetch in parallel
    chunks, history = await asyncio.gather(
        _retrieve(primary_query, filters, top_k=15),
        _get_history(session_id),  # PERF: fetched in parallel instead of after
    )

    if is_legal and len(chunks) < 10:
        legal_queries = [
            f"termination notice period contract agreement {question}",
            f"clause obligation legal contract {question}",
            f"employment vendor sales NDA agreement {question}",
        ]
        seen = {c["notion_page_id"] + c["heading"] for c in chunks}
        # PERF: run legal fallback queries in parallel
        legal_results = await asyncio.gather(
            *[_retrieve(lq, {}, top_k=5) for lq in legal_queries[:2]],
            return_exceptions=True,
        )
        for extras in legal_results:
            if isinstance(extras, Exception):
                continue
            for c in extras:
                uid = c["notion_page_id"] + c["heading"]
                if uid not in seen:
                    seen.add(uid)
                    chunks.append(c)

    # Bug 7 fix: only run broad contract sweep for legal questions
    if is_legal:
        legal_contract_queries = [
            "employment contract termination confidentiality obligations",
            "vendor contract payment liability dispute resolution",
            "sales agreement NDA governing law jurisdiction",
            "service agreement indemnity force majeure clause",
        ]
        seen = {c["notion_page_id"] + c["heading"] for c in chunks}
        # PERF: run all contract queries in parallel
        contract_results = await asyncio.gather(
            *[_retrieve(lq, {}, top_k=4) for lq in legal_contract_queries],
            return_exceptions=True,
        )
        for extras in contract_results:
            if isinstance(extras, Exception):
                continue
            for c in extras:
                uid = c["notion_page_id"] + c["heading"]
                if uid not in seen:
                    seen.add(uid)
                    chunks.append(c)

    if not chunks:
        return {
            "answer":     "No documents found in the knowledge base to analyze. Please run ingest first.",
            "citations":  [],
            "chunks":     [],
            "tool_used":  "analysis",
            "confidence": "low",
        }

    chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
    context = _build_context(chunks[:20])

    response = await _get_llm().ainvoke(
        ANALYSIS_PROMPT.format(context=context, question=question)
    )
    answer = response.content.strip()

    # node_save_history() in agent_graph handles history persistence
    return {
        "answer":     answer,
        "citations":  _citations(chunks),
        "chunks":     chunks,
        "tool_used":  "analysis",
        "confidence": "high",
    } 