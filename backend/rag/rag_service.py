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
"""

import hashlib
import json
import re
import asyncio

import chromadb
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from backend.core.config import settings
from backend.core.logger import logger
from backend.services.redis_service import cache


COLLECTION_NAME = "rag_chunks"
MIN_SCORE       = 0.20
TTL_RETRIEVAL   = 600
TTL_SESSION     = 86400
TTL_ANSWER      = 3600


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

ANSWER_PROMPT = """\
You are CiteRAG — Turabit's internal document assistant.
Answer the question using the context documents provided below.

{history}

Context:
{context}

Question: {question}

ANSWER RULES — follow every rule strictly:

1. USE WHAT IS THERE
   - Answer from the context as completely as possible.
   - If the context PARTIALLY answers the question, give the partial answer and
     state what is missing in one sentence: e.g. "The document covers X but does
     not specify Y."
   - Do NOT say "I could not find" if ANY relevant information is present in the
     context, even if it is incomplete or indirect.

2. FORMAT BY QUESTION TYPE
   Single fact  → 1-2 sentences with the exact value
   What/How/Why → 2-4 sentences with key facts
   List / Steps → numbered list — never as inline prose
   Yes / No     → start with YES or NO, then 1-2 supporting sentences
   Person lookup→ state their name, role, department, and any details present

3. TABLES — REPRODUCE EXACTLY
   - If the context contains a markdown table (rows with | pipes |), you MUST
     reproduce the FULL table in your answer — do not summarize or omit rows.
   - Keep all columns, headers, and data exactly as they appear.
   - If the user asks for a table, section, or schedule, prioritize showing
     the raw table data from the context.

4. EXACT VALUES
   - Always include exact numbers, dates, names, percentages, durations from context.
   - Never paraphrase or approximate a number — use the exact value as written.

5. WHEN TRULY NOT IN CONTEXT
   - Say "I could not find information about this in the available documents."
     ONLY if the context contains absolutely nothing relevant to the question.
   - Never fill gaps with general knowledge, assumptions, or industry standards.

6. STYLE
   - Begin directly with the fact or answer.
   - No intro phrases: "Based on the context...", "According to...", "Certainly!".
   - Never repeat the question back in the answer.
   - Keep answers concise and factual.

Answer:"""


COMPARE_PROMPT = """\
You are CiteRAG — a document analyst for Turabit.
Compare the two documents on the specific question below.

Question: {question}

Content from {doc_a}:
{content_a}

Content from {doc_b}:
{content_b}

COMPARISON RULES:
- Use ALL content provided — do not discard partial or indirect evidence.
- If one document's content is limited or missing, use what is available and note it is partial.
- If BOTH documents are completely silent on the topic, state that there is no document match and REFUSE to answer. Do NOT use general knowledge.
- Each document section: 3-5 bullets of specific facts (numbers, dates, clause wording).
- If a clause is identical in both, write the ACTUAL shared value — never just "Same".
- Never say "Document B mirrors Document A" — state each finding with its own actual value.
- COMPARISON TABLE cells must contain specific values — never "Yes/No" or "Same".
- GAP IDENTIFIED: skip this entire block if there is no real gap — never invent one.
- KEY DIFFERENCE: name the specific clause or value, not a vague category.
- SUMMARY: one concrete recommendation — not a restatement of findings above.

Respond in EXACTLY this format (no extra sections, no reordering):

FINAL ANSWER
[1-2 sentences. Direct answer. If a document is missing a clause, say so and stop padding.]

DOCUMENT A -- {doc_a}
[3-5 bullets: exact facts, numbers, durations, clause wording. Write "Clause not present" if missing.]

DOCUMENT B -- {doc_b}
[3-5 bullets: exact facts, numbers, durations, clause wording. Write "Clause not present" if missing.]

COMPARISON TABLE
| Aspect | {doc_a} | {doc_b} |
|---|---|---|
| [aspect] | [exact finding or "Not present"] | [exact finding or "Not present"] |

KEY DIFFERENCE:
[One sentence: the single most important difference, or "No substantive difference found."]

GAP IDENTIFIED:
What: [specific missing clause or risk — OMIT this entire block if no real gap exists]
Risk: [one concrete legal/operational impact]
Severity: [🔴 HIGH / 🟡 MEDIUM / 🟢 LOW]

COMPARISON INSIGHT:
Expected: [best practice standard]
Actual: [what was found]
Fix: [one specific action]

SUMMARY: [2 sentences max. Main finding and recommended action.]"""


MULTI_COMPARE_PROMPT = """\
You are CiteRAG — a document analyst for Turabit.
Compare ALL the listed documents on the specific question below.

Question: {question}

Documents provided:
{contents}

COMPARISON RULES:
- Use ALL content provided — do not discard partial or indirect evidence.
- For each document: 3-5 bullets of specific facts (numbers, dates, exact clause wording).
- If one document's content is limited, note it is partial and use what is available.
- Never say "same as above" — always write the actual value for that document.
- If a clause is identical across all documents, write the actual shared value in every cell.
- COMPARISON TABLE cells must contain real values — never "Same" or "Yes".
- GAP IDENTIFIED: write "None." if there is no real gap — never invent one.
- KEY DIFFERENCE: name the specific clause or value that differs, if any.
- FINAL ANSWER: a direct 1-2 sentence verdict answering the question.

Respond in EXACTLY this format:

FINAL ANSWER
[1-2 sentences. Direct answer. Include the specific value (e.g. "30 days") if uniform.]

{doc_sections}

COMPARISON TABLE
| Aspect | {doc_headers} |
|{separator}|
| [aspect] | {doc_cells} |

KEY DIFFERENCE:
[One sentence naming the most important difference, or "No substantive difference found."]

GAP IDENTIFIED:
[Write "None." if no real gap. Otherwise: What / Risk / Severity]

SUMMARY: [2 sentences max. Main finding and recommended action.]"""


HYDE_PROMPT = """\
Write a brief factual description (2-4 sentences) as if it were a passage from a
corporate HR policy, legal contract, or finance document at a software company.

Topic: {question}

Include where relevant:
- Specific numbers, durations, percentages, or conditions
- Policy names, clause names, or document section titles
- Employee roles, departments, or hierarchy levels
- Turabit-specific HR terms: notice period, probation, appraisal, reimbursement,
  leave entitlement, carry-forward, salary structure, working hours, overtime,
  resignation, contract terms, confidentiality, non-disclosure, indemnity

Return ONLY the description. No preamble, no filler, no bullet points."""


SUMMARY_PROMPT = """\
You are CiteRAG — a professional document analyst for Turabit.
Write a structured, scannable summary using ONLY the context below.

Context:
{context}

Topic/Question: {question}

SUMMARY RULES:
- If the context is partial, summarise what IS there — do not pad with generic content.
- Every KEY FUNCTION must include at least one specific value: number, name, date, duration, or condition.
- Do not repeat the same fact across multiple KEY FUNCTION sections.
- CONCLUSION must state the practical outcome for an employee — not just the document's purpose.
- Under 300 words total.
- No bullet points inside sections — write in prose sentences.
- No intro or outro phrases ("Here is a summary of...", "In conclusion...").
- Skip any section that has no real content in the context.

Output format — follow EXACTLY:

SUMMARY
[One sentence: what this document/policy covers and its purpose.]

KEY FUNCTIONS

**1. [Function Name]**
[1-2 sentences. Real facts: names, numbers, conditions, timelines.]

**2. [Function Name]**
[1-2 sentences. Real facts only.]

**3. [Function Name]**
[1-2 sentences. Real facts only.]

(Continue up to 8 functions maximum — only include functions with actual content from context)

CONCLUSION
[1 sentence. Practical outcome for an employee.]

Summary:"""


ANALYSIS_PROMPT = """\
You are CiteRAG — a senior legal and business document analyst for Turabit.
Analyze the provided documents and answer the question precisely.

CRITICAL DEFINITIONS — apply strictly:

CONTRADICTION: Two statements that CANNOT both be true simultaneously.
  Real example: Doc A says 30-day notice period AND Doc B says 60 days.
  NOT a contradiction: vague wording, different terminology, missing info.

INCONSISTENCY: Same concept, different wording — not logically conflicting.
GAP: A standard clause or section that is completely absent.
AMBIGUITY: Wording that is unclear or interpretable in multiple ways.

Document content:
{context}

Question: {question}

ANALYSIS RULES:
- Use ALL available evidence — do not refuse to analyze if context is partial.
- If context is limited, analyze what IS there and note specifically what is missing.
- Only report findings that are actually supported by the documents.
- Cite exact [Document > Section] for every single finding.
- Do not invent or hypothesize risks beyond what the documents clearly imply.
- Flag undefined terms (reasonable, promptly, material breach) as AMBIGUITIES.
- Flag absent standard clauses (indemnity, liability cap, force majeure) as GAPS.
- Cross-check ALL documents, not just the first match.
- Each finding must quote or closely paraphrase the actual clause wording.
- Severity must be justified by a real legal or operational consequence.
- CONCLUSION must name the single highest-priority action — do not summarise all findings again.
- If FINAL ANSWER is YES or NO, the very first word of the response must be YES or NO.

FORMAT — include ONLY sections with actual findings:

FINAL ANSWER
[1-2 sentences. Direct YES/NO or overall verdict answering the question.]

## CONTRADICTIONS
[If none found: **No true contradictions found.**]

## INCONSISTENCIES
[Skip this section entirely if none found]

## GAPS
[Skip this section entirely if none found]

## AMBIGUITIES
[Skip this section entirely if none found]

For EACH finding, use this structure:
- **What:** [specific issue — quote or closely paraphrase exact wording]
  **Where:** [document name] > [section name]
  **Risk:** [concrete legal or operational impact]
  **Severity:** 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
  **Severity Reason:** [1 sentence explaining why]
  **Fix:** [concrete, actionable recommendation]

## CONCLUSION
[2-3 sentences. How serious overall? What is the single priority action?]

Analysis:"""


EXPAND_PROMPT = """\
Generate 4 alternative search queries for the following question.
Use different vocabulary and phrasing in each — the goal is to find the same
information using different words that might appear in corporate documents.
Include at least one simpler or more informal phrasing (e.g. how a user might
type it casually, or a Hinglish/transliterated version if the question involves
HR or policy topics common in Indian companies).
Return ONLY the 4 queries, one per line. No numbering, no explanation.

Question: {question}"""


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLETON CLIENTS  (created once, reused)
# ══════════════════════════════════════════════════════════════════════════════

_llm_instance        = None
_embedder_instance   = None
_collection_instance = None


def _get_llm(temperature: float = 0.2, max_tokens: int = 3000):
    global _llm_instance
    
    # If using defaults, try to return singleton
    if temperature == 0.2 and max_tokens == 3000:
        if _llm_instance is None:
            _llm_instance = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_LLM_ENDPOINT,
                api_key=settings.AZURE_OPENAI_LLM_KEY,
                azure_deployment=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
                api_version="2024-12-01-preview",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return _llm_instance
    
    # Otherwise return a fresh instance for specialized tasks (e.g. history compression)
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_LLM_ENDPOINT,
        api_key=settings.AZURE_OPENAI_LLM_KEY,
        azure_deployment=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
        api_version="2024-12-01-preview",
        temperature=temperature,
        max_tokens=max_tokens,
    )   


def _get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
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
        client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        _collection_instance = client.get_or_create_collection(name="rag_chunks")
    return _collection_instance
# ══════════════════════════════════════════════════════════════════════════════
#  LANGUAGE + CACHE HELPERS
# ══════════════════════════════════════════════════════════════════════════════


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
    for msg in data[-8:]:
        role    = msg.get("role", "")
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


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

async def _retrieve_single(query: str, filters: dict, top_k: int,
                            embedder, collection) -> list:
    """Single query retrieval against ChromaDB."""
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



    # Step 0.5: Section/clause number detection — boost retrieval for numeric references
    # Queries like "section 27", "clause 5.2", "article 3" don't embed well,
    # so we also do a direct text search in chunk content for the exact reference.
    _section_pattern = re.compile(
        r'\b(section|clause|article|para|paragraph|schedule|annexure|appendix)'
        r'\s*[\-–—]?\s*(\d+(?:\.\d+)*)\b',
        re.IGNORECASE,
    )
    _section_match = _section_pattern.search(query)
    section_boost_chunks: list = []
    if _section_match:
        sec_label = _section_match.group(1)  # e.g. "section"
        sec_num   = _section_match.group(2)  # e.g. "27"
        sec_ref   = f"{sec_label} {sec_num}"  # e.g. "section 27"
        logger.info("📑 [Section Boost] Detected reference: '%s'", sec_ref)
        # Search ChromaDB with a larger pool and filter by content containing the reference
        try:
            count = collection.count()
            if count > 0:
                broad_results = collection.get(
                    include=["documents", "metadatas"],
                    limit=min(count, 200),
                )
                docs = broad_results.get("documents", [])
                metas = broad_results.get("metadatas", [])
                for doc_text, meta in zip(docs, metas):
                    # Check if this chunk's content or heading contains the section reference
                    text_lower = doc_text.lower()
                    heading_lower = (meta.get("heading", "") or "").lower()
                    if sec_ref.lower() in text_lower or sec_ref.lower() in heading_lower:
                        section_boost_chunks.append({
                            "score":          0.85,  # High synthetic score — exact match
                            "notion_page_id": meta.get("notion_page_id", ""),
                            "doc_title":      meta.get("doc_title", ""),
                            "doc_type":       meta.get("doc_type", ""),
                            "department":     meta.get("department", ""),
                            "version":        meta.get("version", ""),
                            "heading":        meta.get("heading", ""),
                            "content":        doc_text,
                            "citation":       meta.get("citation", ""),
                        })
                if section_boost_chunks:
                    logger.info("📑 [Section Boost] Found %d chunks containing '%s'",
                                len(section_boost_chunks), sec_ref)
        except Exception as e:
            logger.warning("📑 [Section Boost] Failed: %s", e)

    # Step 1: search with original query
    all_chunks = await _retrieve_single(query, filters, top_k, embedder, collection)

    # Merge section-boost chunks (prioritized) with embedding results
    if section_boost_chunks:
        seen_ids = {c["notion_page_id"] + c["heading"] for c in all_chunks}
        for c in section_boost_chunks:
            uid = c["notion_page_id"] + c["heading"]
            if uid not in seen_ids:
                seen_ids.add(uid)
                all_chunks.insert(0, c)  # Insert at front for priority

    # Step 2: expand query with synonyms (only if original returns < 5 results)
    if len(all_chunks) < 5:
        try:
            # Fixed: use async instead of blocking sync .invoke()
            expand_resp = await _get_llm().ainvoke(
                EXPAND_PROMPT.format(question=query)
            )
            expanded = expand_resp.content.strip()
            variants = [v.strip() for v in expanded.splitlines() if v.strip()][:4]
            logger.info("🌿 [Expand] Query expanded to: %s", variants)

            seen_ids = {c["notion_page_id"] + c["heading"] for c in all_chunks}
            # PERF: run variant retrievals in parallel
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

    # Table recovery — if chunks reference a table/schedule, fetch the table chunk too
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
    quality = [c for c in chunks if c.get("score", 0) >= 0.35]
    if not quality:
        quality = chunks[:3]  # fewer bad chunks = less noise
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


def _confidence(chunks: list, answer: str = "") -> str:
    """Answer-aware confidence scoring using top + avg chunk scores."""
    if not chunks:
        return "low"
    if answer and "could not find" in answer.lower():
        return "low"
    avg = sum(c["score"] for c in chunks) / len(chunks)
    top = max(c["score"] for c in chunks)
    if top >= 0.70 and avg >= 0.50:
        return "high"
    elif top >= 0.50 or avg >= 0.35:
        return "medium"
    return "low"


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-VERIFY + RERANK  (Phase 1 & 2 accuracy improvements)
# ══════════════════════════════════════════════════════════════════════════════

VERIFY_PROMPT = """\
Question: {question}
Answer: {answer}

Does this answer directly address the question using specific facts?
Reply ONLY: GOOD, PARTIAL, or BAD
- GOOD: Answer has specific facts that address the question
- PARTIAL: Answer is related but missing key details
- BAD: Answer is vague, generic, or doesn't address the question"""

RERANK_PROMPT = """\
Question: {question}

Rank these passages by relevance to the question (most relevant first).
Return ONLY the numbers in order, comma-separated. Example: 3,1,2

{passages}"""


async def _self_verify(question: str, answer: str) -> str:
    """Quick LLM self-check (~0.5s). Returns GOOD/PARTIAL/BAD."""
    try:
        resp = await _get_llm().ainvoke(
            VERIFY_PROMPT.format(question=question, answer=answer)
        )
        verdict = resp.content.strip().upper().split()[0]
        if verdict in ("GOOD", "PARTIAL", "BAD"):
            return verdict
    except Exception:
        pass
    return "GOOD"  # fail-open


async def _rerank(question: str, chunks: list, top_n: int = 8) -> list:
    """LLM-based reranking of retrieved chunks. Always-on."""
    if len(chunks) <= 3:
        return chunks
    passages = "\n\n".join(
        f"[{i+1}] {c['content'][:200]}" for i, c in enumerate(chunks[:12])
    )
    try:
        resp = await _get_llm().ainvoke(
            RERANK_PROMPT.format(question=question, passages=passages)
        )
        order = [int(x.strip()) - 1 for x in resp.content.strip().split(",")]
        reranked = [chunks[i] for i in order if 0 <= i < len(chunks)]
        return reranked[:top_n] if reranked else chunks[:top_n]
    except Exception:
        return chunks[:top_n]


async def _hyde_retry(question, filters, history, stream_queue=None):
    """HyDE-based retry: generate hypothetical answer, re-retrieve, re-answer."""
    hyp_resp = await _get_llm().ainvoke(HYDE_PROMPT.format(question=question))
    hyp = hyp_resp.content.strip()
    new_chunks = await _retrieve(hyp, filters, top_k=10)
    new_context = _build_context(new_chunks)
    prompt_str = ANSWER_PROMPT.format(history=history, context=new_context, question=question)

    if stream_queue:
        answer_chunks = []
        async for chunk in _get_llm().astream(prompt_str):
            if chunk.content:
                answer_chunks.append(chunk.content)
                await stream_queue.put({"type": "token", "content": chunk.content})
        return "".join(answer_chunks).strip(), new_chunks
    else:
        resp = await _get_llm().ainvoke(prompt_str)
        return resp.content.strip(), new_chunks


# ══════════════════════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════════════════════

async def tool_search(question: str, filters: dict,
                      session_id: str, top_k: int = 8,
                      stream_queue: asyncio.Queue = None) -> dict:
    """
    Standard RAG search tool (Retrieve -> Rerank -> Generate).
    
    Performs a vector search, reranks the results for relevance, and uses 
    the LLM to generate a cited response. Includes self-healing logic 
    (HyDE) if no chunks are found initially.
    
    Args:
        question: The user's query or specific question.
        filters: Dictionary of filters (e.g., department, doc_type).
        session_id: Client session ID for history retrieval.
        top_k: Number of chunks to retrieve initially.
        stream_queue: Optional queue for streaming LLM tokens.
        
    Returns:
        A dictionary containing the answer, citations, chunks, and confidence.
    """
    # ── Redis answer cache ────────────────────────────────────────────────────
    a_key  = _answer_key(question, filters)
    cached = await cache.get(a_key)
    if cached is not None:
        logger.info("⚡ [Cache HIT] answer for %r", question[:60])
        return cached

    chunks = await _retrieve(question, filters, top_k)


    # ── Rerank chunks for better context ────────────────────────────────────────
    chunks = await _rerank(question, chunks)

    # ── Build context + fetch history in parallel ─────────────────────────────
    context = _build_context(chunks)
    history = await _get_history(session_id)

    # ── Early not-found guard WITH SELF-HEALING ────────────────────────────────
    quality_chunks = [c for c in chunks if c.get("score", 0) >= MIN_SCORE]
    if not quality_chunks:
        logger.info("🔄 First-pass empty → triggering HyDE Self-Healing for: %s", question[:60])
        answer, chunks = await _hyde_retry(question, filters, history, stream_queue)
        if "could not find" in answer.lower():
            # Truly not found even after HyDE
            not_found_answer = "I could not find information about this in the available documents."
            result = {
                "answer":     not_found_answer,
                "citations":  [],
                "chunks":     [],
                "tool_used":  "search",
                "confidence": "low",
            }
            await cache.set(a_key, result, ttl=600)
            return result
        # If HyDE found something, we continue to the rest of the logic
    
    # Re-check quality chunks after potential HyDE
    quality_chunks = [c for c in chunks if c.get("score", 0) >= MIN_SCORE]

    # ── LLM generates the answer from retrieved context ───────────────────────
    prompt_str = ANSWER_PROMPT.format(history=history, context=context, question=question)
    
    if stream_queue:
        answer_chunks = []
        async for chunk in _get_llm().astream(prompt_str):
            if chunk.content:
                answer_chunks.append(chunk.content)
                await stream_queue.put({"type": "token", "content": chunk.content})
        answer = "".join(answer_chunks).strip()
    else:
        response = await _get_llm().ainvoke(prompt_str)
        answer = response.content.strip()

    # ── Self-verify: catch bad answers before returning ────────────────────────
    verdict = await _self_verify(question, answer)
    if verdict in ("BAD", "PARTIAL") and quality_chunks:
        logger.info("🔄 Self-verify=%s → retrying with HyDE for: %s", verdict, question[:60])
        answer, chunks = await _hyde_retry(question, filters, history, stream_queue)

    not_found = "could not find" in answer.lower()

    result = {
        "answer":     answer,
        "citations":  _citations(chunks),
        "chunks":     chunks,
        "tool_used":  "search",
        "confidence": "low" if not_found else _confidence(chunks, answer),
    }

    ttl = 600 if not_found else TTL_ANSWER
    await cache.set(a_key, result, ttl=ttl)
    logger.info("💾 [Cache SET] answer for %r (ttl=%ds)", question[:60], ttl)

    return result


async def tool_full_doc(question: str, filters: dict,
                         session_id: str, stream_queue: asyncio.Queue = None) -> dict:
    """
    Retrieves the complete content or a broad set of chunks for a document.
    
    Used when the user asks for the 'full' or 'entire' version of a policy.
    
    Args:
        question: The document name or request.
        filters: Target filters (doc_title, etc.).
        session_id: Client session ID.
        stream_queue: Optional token stream queue.
        
    Returns:
        A dictionary with the complete answer and relevant citations.
    """
    # PERF: run retrieval and history in parallel
    chunks_coro  = _retrieve(question, filters, top_k=15)
    history_coro = _get_history(session_id)
    chunks, history = await asyncio.gather(chunks_coro, history_coro)

    context  = _build_context(chunks)
    prompt   = ANSWER_PROMPT.format(
        history=history, context=context, question=question)
    if stream_queue:
        answer_chunks = []
        async for chunk in _get_llm().astream(prompt):
            if chunk.content:
                answer_chunks.append(chunk.content)
                await stream_queue.put({"type": "token", "content": chunk.content})
        answer = "".join(answer_chunks).strip()
    else:
        response = await _get_llm().ainvoke(prompt)
        answer   = response.content.strip()

    # ── Self-verify ───────────────────────────────────────────────────────────
    verdict = await _self_verify(question, answer)
    if verdict == "BAD" and chunks:
        logger.info("🔄 Full-doc self-verify=BAD → retrying with HyDE")
        answer, chunks = await _hyde_retry(question, filters, history, stream_queue)

    not_found = "could not find" in answer.lower()
    return {
        "answer":     answer,
        "citations":  _citations(chunks),
        "chunks":     chunks,
        "tool_used":  "full_doc",
        "confidence": "low" if not_found else _confidence(chunks, answer),
    }


async def tool_refine(question: str, filters: dict,
                      session_id: str, top_k: int = 15, stream_queue: asyncio.Queue = None) -> dict:
    """HyDE for summaries — generate hypothetical answer first for better retrieval."""
    # Fixed: use async instead of blocking run_in_executor
    hyp_resp = await _get_llm().ainvoke(HYDE_PROMPT.format(question=question))
    hyp = hyp_resp.content.strip()
    history = await _get_history(session_id)

    chunks  = await _retrieve(hyp, filters, top_k)
    context = _build_context(chunks)
    prompt_str = SUMMARY_PROMPT.format(context=context, question=question)
    if stream_queue:
        answer_chunks = []
        async for chunk in _get_llm().astream(prompt_str):
            if chunk.content:
                answer_chunks.append(chunk.content)
                await stream_queue.put({"type": "token", "content": chunk.content})
        answer = "".join(answer_chunks).strip()
    else:
        answer = await _get_llm().ainvoke(prompt_str)
        answer = answer.content.strip()
    not_found = "could not find" in answer.lower()
    return {
        "answer":     answer,
        "citations":  _citations(chunks),
        "chunks":     chunks,
        "tool_used":  "refine",
        "confidence": "low" if not_found else _confidence(chunks, answer),
    }


async def tool_compare(question: str, doc_a: str, doc_b: str,
                       filters: dict, session_id: str, top_k: int = 6, 
                       stream_queue: asyncio.Queue = None) -> dict:
    """
    Compares two specific documents against a common question.
    
    Retrieves prioritized chunks for both documents and uses a structured
    comparison prompt to generate a side-by-side analysis and a verdict.
    
    Args:
        question: The aspect or question to compare (e.g., 'notice period').
        doc_a: Title of the first document.
        doc_b: Title of the second document.
        filters: Shared filters.
        session_id: Client session ID.
        top_k: Number of chunks to retrieve per document.
        stream_queue: Optional token stream queue.
        
    Returns:
        A dictionary containing the parsed comparison results and citations.
    """

    # Dynamic boost: only use legal terms if the question isn't specific
    _specialized = any(w in question.lower() for w in ["policy", "payment", "date", "term", "clause", "notice", "leave"])
    _boost = "" if _specialized else "contract agreement clause legal terms obligations"
    
    query_a = f"{question} {_boost} {doc_a}"
    query_b = f"{question} {_boost} {doc_b}"

    # PERF: already parallel — also fetch history at the same time
    chunks_a, chunks_b, history = await asyncio.gather(
        _retrieve(query_a, filters, top_k * 3),
        _retrieve(query_b, filters, top_k * 3),
        _get_history(session_id),
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
        words   = [w for w in target_title.lower().split() if len(w) > 3]
        partial = [c for c in chunks
                   if any(w in c["doc_title"].lower() for w in words)
                   and other_title.lower() not in c["doc_title"].lower()]
        if partial:
            return partial[:top_k]
        # Step 3: fallback — best scoring chunks, still excluding the other doc
        excluded = [c for c in chunks
                    if other_title.lower() not in c["doc_title"].lower()]
        return sorted(excluded or chunks, key=lambda x: x["score"], reverse=True)[:top_k]

    chunks_a  = filter_doc(chunks_a, doc_a, doc_b)
    chunks_b  = filter_doc(chunks_b, doc_b, doc_a)
    content_a = _build_context(chunks_a)
    content_b = _build_context(chunks_b)

    prompt_str = COMPARE_PROMPT.format(
            question=question, doc_a=doc_a, doc_b=doc_b,
            content_a=content_a, content_b=content_b)
    
    if stream_queue:
        answer_chunks = []
        async for chunk in _get_llm().astream(prompt_str):
            if chunk.content:
                answer_chunks.append(chunk.content)
                await stream_queue.put({"type": "token", "content": chunk.content})
        raw = "".join(answer_chunks).strip()
    else:
        response = await _get_llm().ainvoke(prompt_str)
        raw = response.content.strip()

    def _extract(text, start_tag, end_tags):
        """Case-insensitive tag extraction — handles LLM formatting variations."""
        text_lower = text.lower()
        start_lower = start_tag.lower()
        if start_lower not in text_lower:
            return ""
        idx = text_lower.index(start_lower)
        part = text[idx + len(start_tag):]
        for tag in end_tags:
            tag_lower = tag.lower()
            if tag_lower in part.lower():
                end_idx = part.lower().index(tag_lower)
                part = part[:end_idx]
        return part.strip()

    # Try multiple tag formats the LLM might produce
    doc_a_tag = None
    doc_b_tag = None
    for fmt_a in [f"DOCUMENT A -- {doc_a}", f"DOCUMENT A: {doc_a}",
                  f"DOCUMENT A — {doc_a}", f"**DOCUMENT A** -- {doc_a}",
                  f"## {doc_a}", f"DOCUMENT A - {doc_a}"]:
        if fmt_a.lower() in raw.lower():
            doc_a_tag = fmt_a
            break
    for fmt_b in [f"DOCUMENT B -- {doc_b}", f"DOCUMENT B: {doc_b}",
                  f"DOCUMENT B — {doc_b}", f"**DOCUMENT B** -- {doc_b}",
                  f"## {doc_b}", f"DOCUMENT B - {doc_b}"]:
        if fmt_b.lower() in raw.lower():
            doc_b_tag = fmt_b
            break

    if doc_a_tag:
        end_tags_a = [doc_b_tag or "DOCUMENT B", "COMPARISON TABLE", "GAP IDENTIFIED:"]
        side_a     = _extract(raw, doc_a_tag, end_tags_a)
        end_tags_b = ["COMPARISON TABLE", "GAP IDENTIFIED:", "KEY DIFFERENCE:", "SYSTEMIC ISSUE", "COMPARISON INSIGHT:", "SUMMARY:"]
        side_b     = _extract(raw, doc_b_tag or f"DOCUMENT B -- {doc_b}", end_tags_b)
        comp_table = _extract(raw, "COMPARISON TABLE", ["GAP IDENTIFIED:", "KEY DIFFERENCE:", "SYSTEMIC ISSUE", "COMPARISON INSIGHT:"])
    else:
        side_a     = _extract(raw, "DOCUMENT_A:", ["DOCUMENT_B:"])
        side_b     = _extract(raw, "DOCUMENT_B:", ["GAP IDENTIFIED:", "KEY DIFFERENCE:", "COMPARISON INSIGHT:", "SUMMARY:"])
        comp_table = ""
    summary = _extract(raw, "SUMMARY:", [])

    if not side_a:
        side_a, side_b, comp_table = content_a[:600], content_b[:600], ""

    all_chunks = chunks_a + chunks_b
    return {
        "answer":      raw,
        "side_a":      side_a,
        "side_b":      side_b,
        "comp_table":  comp_table,
        "summary":     summary,
        "doc_a":       doc_a,
        "doc_b":       doc_b,
        "citations":   _citations(all_chunks),
        "chunks":      all_chunks,
        "tool_used":   "compare",
    }


async def tool_multi_compare(
    question: str,
    doc_names: list,
    filters: dict,
    session_id: str,
    top_k: int = 6,
    stream_queue: asyncio.Queue = None,
) -> dict:
    """
    Compare N documents against a single question.
    Retrieves chunks per-doc in parallel, then sends one structured prompt.
    """
    # Dynamic boost: only use legal terms if the question isn't specific
    _specialized = any(w in question.lower() for w in ["policy", "payment", "date", "term", "clause", "notice", "leave"])
    _boost = "" if _specialized else "contract agreement clause legal terms obligations"

    # ── 1. Parallel retrieval for all docs ───────────────────────────────────
    async def _retrieve_for_doc(doc_name: str):
        q      = f"{question} {_boost} {doc_name}"
        chunks = await _retrieve(q, filters, top_k * 3)
        # Filter to only chunks from that document
        exact  = [c for c in chunks if doc_name.lower() in c["doc_title"].lower()]
        if exact:
            return doc_name, exact[:top_k]
        # Fallback: best-scoring chunks that are not from OTHER named docs
        other_docs = [d.lower() for d in doc_names if d != doc_name]
        excluded   = [c for c in chunks
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

    # ── 2. Build per-doc content blocks ──────────────────────────────────────
    contents_block = []
    all_chunks     = []
    for doc_name in doc_names:
        chunks  = doc_chunks_map.get(doc_name, [])
        all_chunks.extend(chunks)
        content = _build_context(chunks) if chunks else "No relevant content found for this document."
        contents_block.append(f"--- {doc_name} ---\n{content}")

    # ── 3. Build prompt slots ─────────────────────────────────────────────────
    doc_headers  = " | ".join(doc_names)
    separator    = "|".join(["---"] * (len(doc_names) + 1))
    doc_cells    = " | ".join([f"[{d} value]" for d in doc_names])
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

    if stream_queue:
        answer_chunks = []
        async for chunk in _get_llm().astream(prompt):
            if chunk.content:
                answer_chunks.append(chunk.content)
                await stream_queue.put({"type": "token", "content": chunk.content})
        raw = "".join(answer_chunks).strip()
    else:
        response = await _get_llm().ainvoke(prompt)
        raw      = response.content.strip()

    summary = ""
    if "SUMMARY:" in raw:
        summary = raw.split("SUMMARY:", 1)[1].strip()

    # node_save_history() in agent_graph handles history persistence
    return {
        "answer":     raw,
        "summary":    summary,
        "doc_names":  doc_names,
        "citations":  _citations(all_chunks),
        "chunks":     all_chunks,
        "tool_used":  "multi_compare",
        "confidence": _confidence(all_chunks),
    }


async def tool_analysis(question: str, filters: dict,
                        session_id: str, stream_queue: asyncio.Queue = None) -> dict:
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
        primary_query  = f"{question} {contract_boost}"
    else:
        primary_query = question

    # PERF: run primary retrieval and history fetch in parallel
    chunks, history = await asyncio.gather(
        _retrieve(primary_query, filters, top_k=15),
        _get_history(session_id),
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

    chunks  = sorted(chunks, key=lambda x: x["score"], reverse=True)
    context = _build_context(chunks[:20])

    prompt_str = ANALYSIS_PROMPT.format(context=context, question=question)

    if stream_queue:
        answer_chunks = []
        async for chunk in _get_llm().astream(prompt_str):
            if chunk.content:
                answer_chunks.append(chunk.content)
                await stream_queue.put({"type": "token", "content": chunk.content})
        answer = "".join(answer_chunks).strip()
    else:
        response = await _get_llm().ainvoke(prompt_str)
        answer = response.content.strip()

    # ── Self-verify ───────────────────────────────────────────────────────────
    verdict = await _self_verify(question, answer)
    if verdict == "BAD" and chunks:
        logger.info("🔄 Analysis self-verify=BAD → retrying with HyDE")
        answer, chunks = await _hyde_retry(question, {}, history, stream_queue)

    return {
        "answer":     answer,
        "citations":  _citations(chunks),
        "chunks":     chunks,
        "tool_used":  "analysis",
        "confidence": _confidence(chunks, answer),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FOLLOW-UP GENERATION
# ══════════════════════════════════════════════════════════════════════════════

async def generate_followups(question: str, answer: str) -> list[str]:
    """Generate 3 likely follow-up questions based on the answer provided."""
    if not answer or len(answer) < 20 or "Could not find information" in answer:
        return []
        
    prompt = (
        "You are a helpful assistant. Based on the user's question and your answer, "
        "predict 3 highly relevant and specific follow-up questions the user is likely to ask next.\n\n"
        f"User Question: {question}\n\n"
        f"Your Answer: {answer}\n\n"
        "Return ONLY a JSON array of 3 strings. Example: "
        '["What is the penalty for late payment?", "Who do I contact for an extension?", "Can I see the SLA terms?"]'
    )
    
    try:
        import json
        llm = _get_llm()
        # Fixed: use async instead of blocking run_in_executor
        resp = await llm.ainvoke(prompt)
        res = resp.content
        
        # Strip markdown json blocks if present
        res = res.strip()
        if res.startswith("```json"):
            res = res[7:-3].strip()
        elif res.startswith("```"):
            res = res[3:-3].strip()
            
        followups = json.loads(res)
        if isinstance(followups, list):
            return followups[:3]
    except Exception as e:
        logger.warning(f"Failed to generate followups: {e}")
    return []