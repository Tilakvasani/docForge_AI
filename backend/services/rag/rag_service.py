"""
rag_service.py — Smart RAG Service
Query → Understand intent → Right tool → Retrieve → LLM → Clean answer
"""

import hashlib
import json
from typing import Optional
from backend.core.config import settings
from backend.core.logger import logger
from backend.services.redis_service import cache

COLLECTION_NAME = "rag_chunks"
MIN_SCORE       = 0.10
TTL_RETRIEVAL   = 600
TTL_SESSION     = 1800


# ── Prompts ───────────────────────────────────────────────────────────────────

ANSWER_PROMPT = """\
You are a smart document assistant for turabit. Use ONLY the context below to answer.
If the answer is not in the context, say: "I could not find information about this in the available documents."

{history}

Context:
{context}

Question: {question}

Rules:
1. NO intro phrases — start the answer immediately
2. NO outro phrases — stop when the answer is complete
3. NO vague labels like "authorization policy" or "approval policy" — give the ACTUAL content
4. ALWAYS include real details: specific numbers, names, dates, amounts, conditions, steps
5. NEVER summarize a section heading — explain what that section actually says

Output format based on question type:
- Yes/No question → 1 sentence
- Single fact (notice period, salary, date) → state it directly with the exact value
- What/How/Why question → 2-5 sentences with real specifics
- List question (what are, list all) → clean bullet points with actual details per item
- Summary request → 150-250 words, each bullet point has real content not just a label
- Full document request → all sections with complete content
- Comparison → per-document breakdown with actual differences

Do NOT write things like:
- "Discount approval policy" ← BAD (just a label)
- "Network access policy for technology" ← BAD (restating the title)
- "Authorized signature policy" ← BAD (meaningless)

DO write things like:
- "Discounts above 15% require written approval from the Sales Manager" ← GOOD
- "Network access requires VPN, 2FA, and manager approval within 24 hours" ← GOOD
- "Signatures required from HR Manager, Chief People Officer, and Employee" ← GOOD

Answer:"""

COMPARE_PROMPT = """\
You are comparing two documents. Answer the question for each document separately using ONLY their content.
Be specific, detailed and complete. Include all relevant facts, numbers, dates from each document.

Question: {question}

Content from {doc_a}:
{content_a}

Content from {doc_b}:
{content_b}

Respond in this EXACT format (no extra text before or after):
DOCUMENT_A: [Complete detailed answer based ONLY on {doc_a} content. Use bullet points for multiple items. Include specific numbers, dates, names.]
DOCUMENT_B: [Complete detailed answer based ONLY on {doc_b} content. Use bullet points for multiple items. Include specific numbers, dates, names.]
SUMMARY: [3-5 sentences covering the main similarities and key differences between the two documents on this topic]"""

HYDE_PROMPT = """\
Write a brief factual description (2-3 sentences) about this business topic: {question}"""

EXPAND_PROMPT = """\
Rewrite this question in 3 different ways using different words and synonyms that mean the same thing.
Keep each version short (under 15 words). Return only the 3 versions, one per line, no numbering.

Question: {question}"""

REWRITE_PROMPT = """\
You are a query understanding assistant for a company document system at turabit.
The user asked: "{question}"

Your job:
1. Understand what the user really wants
2. Rewrite it as a clear, precise question that will find the right document content
3. Identify the intent type

Rules:
- Fix typos and informal language
- Expand abbreviations (e.g. "HR" → "Human Resources")
- Make vague questions specific (e.g. "tell me about pay" → "What is the salary and compensation structure?")
- Keep domain context (this is a company document system)

Reply in this exact format:
REWRITTEN: [the clear precise question]
INTENT: [one of: GREETING, GENERAL, COMPARE, FULL_DOC, SUMMARY, LIST, YESNO, SPECIFIC, EXPLAIN, ANALYSIS, SEARCH]"""

ANALYSIS_PROMPT = """\
You are a senior legal and business document analyst for turabit.
Analyze the provided documents and answer the question precisely.

CRITICAL DEFINITIONS — apply strictly:

CONTRADICTION: Two statements that CANNOT both be true simultaneously.
  Real: Doc A says notice period 30 days AND Doc B says 60 days.
  NOT contradiction: vague wording, different terminology, missing info.

INCONSISTENCY: Same concept, different wording, not logically conflicting.
GAP: Standard clause completely missing.
AMBIGUITY: Wording unclear or interpretable multiple ways.

Document content:
{context}

Question: {question}

Format rules:
- Use ## CONTRADICTIONS, ## INCONSISTENCIES, ## GAPS, ## AMBIGUITIES as section headers
- Only include sections with actual findings
- If no true contradictions write: **No true contradictions found.**
- Start immediately, no intro phrases
- For each finding use this exact format:

  - **What:** [specific issue with exact wording from document]
    **Where:** [document name] > [section name]
    **Risk:** [legal or operational impact]
    **Severity:** 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
    **Fix:** [concrete recommendation]

Analysis:"""





# ── Clients ───────────────────────────────────────────────────────────────────

def _get_llm():
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_LLM_ENDPOINT,
        api_key=settings.AZURE_OPENAI_LLM_KEY,
        azure_deployment=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
        api_version="2024-12-01-preview",
        temperature=0.2,
        max_tokens=3000,
    )


def _get_embedder():
    from langchain_openai import AzureOpenAIEmbeddings
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.AZURE_EMB_ENDPOINT,
        api_key=settings.AZURE_OPENAI_EMB_KEY,
        azure_deployment=settings.AZURE_EMB_DEPLOYMENT,
        api_version=settings.AZURE_EMB_API_VERSION,
    )


def _get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _retrieval_key(query: str, filters: dict, top_k: int) -> str:
    raw = json.dumps({"q": query, "f": filters, "k": top_k}, sort_keys=True)
    return f"docforge:rag:retrieval:{hashlib.md5(raw.encode()).hexdigest()}"


async def _get_history(session_id: str) -> str:
    data = await cache.get(f"docforge:rag:session:{session_id}") or []
    if not data:
        return ""
    lines = ["Previous conversation:"]
    for turn in data[-4:]:
        lines.append(f"User: {turn['q']}")
        lines.append(f"Assistant: {turn['a'][:200]}...")
    return "\n".join(lines) + "\n"


async def _save_turn(session_id: str, q: str, a: str):
    key  = f"docforge:rag:session:{session_id}"
    data = await cache.get(key) or []
    data.append({"q": q, "a": a})
    await cache.set(key, data[-10:], ttl=TTL_SESSION)


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


async def _retrieve(query: str, filters: dict, top_k: int = 15) -> list:
    """
    Smart retrieval with query expansion.
    Searches with original + expanded queries, merges and deduplicates results.
    Handles semantic mismatch: user says "pay" → finds "compensation",
    user says "fire" → finds "termination/dismissal", etc.
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
            logger.info("Query expanded to: %s", variants)

            # Search with each variant
            seen_ids = {c["notion_page_id"] + c["heading"] for c in all_chunks}
            for variant in variants:
                extra = await _retrieve_single(
                    variant, filters, top_k // 2, embedder, collection)
                for c in extra:
                    uid = c["notion_page_id"] + c["heading"]
                    if uid not in seen_ids:
                        seen_ids.add(uid)
                        all_chunks.append(c)
        except Exception as e:
            logger.warning("Query expansion failed: %s", e)

    # Step 3: deduplicate and sort by score
    seen, final = set(), []
    for c in sorted(all_chunks, key=lambda x: x["score"], reverse=True):
        uid = c["notion_page_id"] + c["heading"]
        if uid not in seen:
            seen.add(uid)
            final.append(c)

    final = final[:top_k]
    await cache.set(key, final, ttl=TTL_RETRIEVAL)
    logger.info("Retrieved %d chunks (with expansion) for: %s", len(final), query[:50])
    return final


def _build_context(chunks: list) -> str:
    if not chunks:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"Source: {c['citation']}\n{c['content']}"
        for c in chunks)


def _citations(chunks: list) -> list:
    seen, out = set(), []
    for c in chunks:
        cit     = c.get("citation", "")
        page_id = c.get("notion_page_id", "")
        url     = f"https://www.notion.so/{page_id}" if page_id else ""
        if cit and cit not in seen:
            seen.add(cit)
            out.append({"text": cit, "url": url})
    return out


def _confidence(chunks: list) -> str:
    if not chunks:
        return "low"
    avg = sum(c["score"] for c in chunks) / len(chunks)
    return "high" if avg >= 0.60 else "medium" if avg >= 0.40 else "low"


# ── Intent detection ──────────────────────────────────────────────────────────

def _classify_intent(question: str) -> str:
    """Smart rule-based intent detection for all question types."""
    q = question.lower().strip().rstrip("!?.")

    # ── Greeting ──────────────────────────────────────────────────────────────
    if q in ["hey", "hi", "hello", "hiya", "yo", "sup", "howdy",
             "thanks", "thank you", "thx", "bye", "goodbye", "ok", "okay", "cool"]:
        return "GREETING"
    if q in ["how are you", "how r u", "how are u"]:
        return "GREETING"
    if any(q.startswith(w + " ") for w in ["hey", "hi", "hello"]):
        rest = q.split(" ", 1)[1]
        doc_words = ["policy", "letter", "contract", "document", "offer",
                     "leave", "salary", "employee", "hr", "notice", "clause"]
        if not any(w in rest for w in doc_words):
            return "GREETING"
    if any(q.startswith(w) for w in ["who are you", "what are you", "what can you do"]):
        return "GREETING"

    # ── General knowledge (not in documents) ──────────────────────────────────
    if any(p in q for p in ["who is the president", "what is the capital",
                              "prime minister of", "chief minister of", " cm of ",
                              "ceo of ", "history of ", "population of",
                              "who invented", "who discovered", "largest country",
                              "when was born", "tallest building"]):
        return "GENERAL"

    # ── Analysis questions (complex reasoning over documents) ─────────────────
    if any(p in q for p in [
        # Contradiction / conflict
        "contradict", "contradiction", "inconsisten", "conflict",
        "internal conflict", "mutually exclusive", "violat", "comply",
        "complying with", "clause violat", "obligation", "exclusive",
        # Gaps / missing
        "missing", "what is missing", "gaps in", "incomplete",
        "not mentioned", "not covered", "absent", "omitted",
        # Issues / problems
        "issue", "problem", "wrong", "weakness", "flaw",
        "loophole", "ambiguous", "unclear", "vague",
        # Review / audit
        "review", "audit", "evaluate", "assess", "analyse", "analyze",
        "check", "verify", "examine",
        # Improvement
        "improve", "recommendation", "suggest", "better",
        # Completeness
        "complete", "correct", "accurate", "valid",
        # Compliance
        "comply", "complian", "legal", "enforceable",
    ]):
        return "ANALYSIS"

    # ── Compare two documents ─────────────────────────────────────────────────
    if any(w in q for w in ["compare", "difference between", " vs ",
                              "versus", "contrast", "which is better",
                              "how do they differ", "what's the difference"]):
        return "COMPARE"

    # ── Full document request ─────────────────────────────────────────────────
    if any(p in q for p in ["full ", "complete ", "entire ", "whole ",
                              "give me the full", "show me the full",
                              "full offer", "full contract", "full letter",
                              "full handbook", "full document", "full policy"]):
        return "FULL_DOC"

    # ── Summary / overview ────────────────────────────────────────────────────
    if any(w in q for w in ["summarise", "summarize", "summary", "overview",
                              "brief", "in short", "key points", "main points",
                              "highlight", "gist", "tldr", "tl;dr"]):
        return "SUMMARY"

    # ── List questions ────────────────────────────────────────────────────────
    if any(p in q for p in ["list all", "list the", "all the ", "what are all",
                              "give me all", "show all", "what types of",
                              "what kind of", "enumerate", "what policies"]):
        return "LIST"

    # ── Yes/No questions ──────────────────────────────────────────────────────
    if q.startswith(("is ", "are ", "does ", "do ", "has ", "have ",
                      "can ", "will ", "was ", "were ", "should ")):
        return "YESNO"

    # ── Specific fact ─────────────────────────────────────────────────────────
    if any(p in q for p in ["how many", "how much", "how long", "how often",
                              "what is the", "when is", "when does", "who is",
                              "which", "notice period", "salary", "working hours",
                              "leave days", "deadline", "date", "amount",
                              "percentage", "number of"]):
        return "SPECIFIC"

    # ── Explanation ───────────────────────────────────────────────────────────
    if any(p in q for p in ["explain", "how does", "how do", "why is",
                              "why does", "what happens", "walk me through",
                              "tell me about", "describe", "elaborate",
                              "what does it mean", "clarify"]):
        return "EXPLAIN"

    return "SEARCH"



# ── Casual responses ──────────────────────────────────────────────────────────

CASUAL_RESPONSES = {
    "greeting": "Hello! I'm CiteRAG Lab, your document assistant for turabit. Ask me anything about your company documents — policies, contracts, offers, and more.",
    "thanks":   "You're welcome! Let me know if you have any other questions.",
    "bye":      "Goodbye! Come back anytime you need help with your documents.",
    "identity": "I'm CiteRAG Lab — an AI assistant that answers questions based on turabit's Notion documents with citations.",
}


def _casual_response(question: str) -> str:
    q = question.lower().strip().rstrip("!?.")
    if any(q.startswith(w) for w in ["thanks", "thank you", "thx"]):
        return CASUAL_RESPONSES["thanks"]
    if any(q.startswith(w) for w in ["bye", "goodbye"]):
        return CASUAL_RESPONSES["bye"]
    if any(w in q for w in ["who are you", "what are you", "what can you do"]):
        return CASUAL_RESPONSES["identity"]
    return CASUAL_RESPONSES["greeting"]


# ── Tools ─────────────────────────────────────────────────────────────────────

async def tool_search(question: str, filters: dict,
                      session_id: str, top_k: int = 15) -> dict:
    chunks  = await _retrieve(question, filters, top_k)
    context = _build_context(chunks)
    history = await _get_history(session_id)
    answer  = _get_llm().invoke(
        ANSWER_PROMPT.format(history=history, context=context, question=question)
    ).content.strip()
    not_found = "could not find" in answer.lower()
    await _save_turn(session_id, question, answer)
    return {
        "answer":     answer,
        "citations":  [] if not_found else _citations(chunks),
        "chunks":     [] if not_found else chunks,
        "tool_used":  "search",
        "confidence": "low" if not_found else _confidence(chunks),
    }


async def tool_full_doc(question: str, filters: dict,
                        session_id: str) -> dict:
    """For full document requests — retrieve more chunks with higher top_k."""
    chunks  = await _retrieve(question, filters, top_k=25)
    context = _build_context(chunks)
    history = await _get_history(session_id)
    prompt  = ANSWER_PROMPT.format(
        history=history, context=context, question=question)
    answer  = _get_llm().invoke(prompt).content.strip()
    not_found = "could not find" in answer.lower()
    await _save_turn(session_id, question, answer)
    return {
        "answer":     answer,
        "citations":  [] if not_found else _citations(chunks),
        "chunks":     [] if not_found else chunks,
        "tool_used":  "full_doc",
        "confidence": "low" if not_found else _confidence(chunks),
    }


async def tool_refine(question: str, filters: dict,
                      session_id: str, top_k: int = 15) -> dict:
    """HyDE for summaries — generate hypothetical answer first for better retrieval."""
    hyp     = _get_llm().invoke(
        HYDE_PROMPT.format(question=question)).content.strip()
    chunks  = await _retrieve(hyp, filters, top_k)
    context = _build_context(chunks)
    history = await _get_history(session_id)
    answer  = _get_llm().invoke(
        ANSWER_PROMPT.format(history=history, context=context, question=question)
    ).content.strip()
    not_found = "could not find" in answer.lower()
    await _save_turn(session_id, question, answer)
    return {
        "answer":     answer,
        "citations":  [] if not_found else _citations(chunks),
        "chunks":     [] if not_found else chunks,
        "tool_used":  "refine",
        "confidence": "low" if not_found else _confidence(chunks),
    }


async def tool_compare(question: str, doc_a: str, doc_b: str,
                       filters: dict, session_id: str, top_k: int = 6) -> dict:
    import asyncio

    # Retrieve separately with doc-specific filters
    filters_a = {**filters, "doc_title": doc_a} if doc_a != "Document A" else filters
    filters_b = {**filters, "doc_title": doc_b} if doc_b != "Document B" else filters

    # Search with doc name embedded in query for better matching
    query_a = f"{question} {doc_a}"
    query_b = f"{question} {doc_b}"

    chunks_a, chunks_b = await asyncio.gather(
        _retrieve(query_a, filters, top_k * 3),
        _retrieve(query_b, filters, top_k * 3),
    )

    def filter_doc(chunks, title):
        # Try exact title match first
        exact = [c for c in chunks if title.lower() in c["doc_title"].lower()]
        if exact:
            return exact[:top_k]
        # Try partial word match
        words = [w for w in title.lower().split() if len(w) > 3]
        partial = [c for c in chunks if any(w in c["doc_title"].lower() for w in words)]
        if partial:
            return partial[:top_k]
        # Last resort - top scoring chunks
        return sorted(chunks, key=lambda x: x["score"], reverse=True)[:top_k]

    chunks_a = filter_doc(chunks_a, doc_a)
    chunks_b = filter_doc(chunks_b, doc_b)
    content_a = _build_context(chunks_a)
    content_b = _build_context(chunks_b)

    raw = _get_llm().invoke(
        COMPARE_PROMPT.format(
            question=question, doc_a=doc_a, doc_b=doc_b,
            content_a=content_a, content_b=content_b)
    ).content.strip()

    side_a, side_b, summary = "", "", ""
    if "DOCUMENT_A:" in raw and "DOCUMENT_B:" in raw:
        parts_a = raw.split("DOCUMENT_A:", 1)[1]
        if "DOCUMENT_B:" in parts_a:
            side_a  = parts_a.split("DOCUMENT_B:")[0].strip()
            parts_b = parts_a.split("DOCUMENT_B:", 1)[1]
            if "SUMMARY:" in parts_b:
                side_b  = parts_b.split("SUMMARY:")[0].strip()
                summary = parts_b.split("SUMMARY:", 1)[1].strip()
            else:
                side_b = parts_b.strip()
    if not side_a:
        side_a, side_b, summary = content_a[:600], content_b[:600], raw[:300]

    all_chunks = chunks_a + chunks_b
    await _save_turn(session_id, question, summary or raw[:200])
    return {
        "answer":     summary or raw[:200],
        "side_a":     side_a,
        "side_b":     side_b,
        "summary":    summary,
        "doc_a":      doc_a,
        "doc_b":      doc_b,
        "citations":  _citations(all_chunks),
        "chunks":     all_chunks,
        "tool_used":  "compare",
        "confidence": _confidence(all_chunks),
    }


async def tool_analysis(question: str, filters: dict,
                        session_id: str) -> dict:
    """
    For analysis questions — retrieve broad context then reason over it.
    Key insight: never search for 'contradictions' in vector DB —
    instead retrieve actual document content broadly, then let LLM analyze it.
    """
    # Step 1: Extract subject from question to know WHAT to retrieve
    # e.g. "contradictions in this document" → retrieve all available docs
    # e.g. "issues with leave policy" → retrieve leave-related chunks
    subject_prompt = (
        f"Extract the document topic from this question in 3-5 keywords. "
        f"If the question says 'this document' with no specific doc mentioned, "
        f"return: all documents. Question: '{question}'"
    )
    try:
        subject = _get_llm().invoke(subject_prompt).content.strip()
        logger.info("Analysis subject: %s", subject)
    except Exception:
        subject = "company policies documents"

    # Step 2: Retrieve broadly using subject keywords
    chunks = await _retrieve(subject, filters, top_k=25)

    # Step 3: If still empty, retrieve everything available
    if not chunks:
        broad_queries = [
            "company policy employee handbook",
            "terms conditions agreement contract",
            "authorization approval procedure",
            "employment HR leave salary",
        ]
        seen = set()
        for q in broad_queries:
            extra = await _retrieve(q, {}, top_k=8)
            for c in extra:
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

    context = _build_context(chunks[:30])

    answer = _get_llm().invoke(
        ANALYSIS_PROMPT.format(context=context, question=question)
    ).content.strip()

    await _save_turn(session_id, question, answer)
    return {
        "answer":     answer,
        "citations":  _citations(chunks),
        "chunks":     chunks,
        "tool_used":  "analysis",
        "confidence": "high",
    }


async def _rewrite_query(question: str) -> tuple[str, str]:
    """
    Use LLM to understand user intent and rewrite query.
    Returns (rewritten_question, intent)
    Falls back to original question if LLM fails.
    """
    # Skip rewrite for very short/simple questions (speed)
    q = question.strip().lower()
    if len(q.split()) <= 3:
        return question, _classify_intent(question)

    try:
        raw = _get_llm().invoke(
            REWRITE_PROMPT.format(question=question)
        ).content.strip()

        rewritten = question
        intent    = "SEARCH"

        for line in raw.splitlines():
            if line.startswith("REWRITTEN:"):
                rewritten = line.replace("REWRITTEN:", "").strip()
            elif line.startswith("INTENT:"):
                intent = line.replace("INTENT:", "").strip().upper()

        # Validate intent
        valid = {"GREETING","GENERAL","COMPARE","FULL_DOC","SUMMARY",
                 "LIST","YESNO","SPECIFIC","EXPLAIN","ANALYSIS","SEARCH"}
        if intent not in valid:
            intent = "SEARCH"

        logger.info("Rewrite: '%s' → '%s' [%s]", question[:50], rewritten[:50], intent)
        return rewritten, intent

    except Exception as e:
        logger.warning("Query rewrite failed: %s", e)
        return question, _classify_intent(question)


# ── Main dispatcher ───────────────────────────────────────────────────────────

async def answer(
    question:   str,
    filters:    Optional[dict] = None,
    session_id: str = "default",
    top_k:      int = 15,
    doc_a:      str = "",
    doc_b:      str = "",
) -> dict:
    filters = filters or {}

    # Use rule-based for obvious cases (fast, no LLM call)
    quick_intent = _classify_intent(question)
    if quick_intent == "GREETING":
        intent, rewritten = "GREETING", question
    elif quick_intent == "GENERAL":
        intent, rewritten = "GENERAL", question
    elif quick_intent == "ANALYSIS":
        # Never rewrite analysis questions — send as-is to tool_analysis
        intent, rewritten = "ANALYSIS", question
    else:
        # Use LLM to understand and rewrite the question
        rewritten, intent = await _rewrite_query(question)

    logger.info("Intent: %s | Original: '%s' | Rewritten: '%s'",
                intent, question[:50], rewritten[:50])

    # Use rewritten question for all RAG operations
    question = rewritten

    # 1. Greeting — instant response
    if intent == "GREETING":
        response = _casual_response(question)
        await _save_turn(session_id, question, response)
        return {
            "answer":     response,
            "citations":  [],
            "chunks":     [],
            "tool_used":  "chat",
            "confidence": "high",
        }

    # 2. General knowledge — answer with LLM directly
    if intent == "GENERAL":
        gk_answer = _get_llm().invoke(
            f"Answer this general knowledge question briefly and accurately: {question}"
        ).content.strip()
        await _save_turn(session_id, question, gk_answer)
        return {
            "answer":     gk_answer,
            "citations":  [],
            "chunks":     [],
            "tool_used":  "general",
            "confidence": "high",
        }

    # 3. Compare
    if intent == "COMPARE" or (doc_a and doc_b):
        return await tool_compare(
            question, doc_a or "Document A", doc_b or "Document B",
            filters, session_id, top_k)

    # 4. Full document — retrieve 25 chunks
    if intent == "FULL_DOC":
        return await tool_full_doc(question, filters, session_id)

    # 5. Summary / Overview — use HyDE for better retrieval
    if intent == "SUMMARY":
        return await tool_refine(question, filters, session_id, top_k)

    # 6. Analysis (contradictions, gaps, issues, review)
    if intent == "ANALYSIS":
        return await tool_analysis(question, filters, session_id)

    # 7. List — retrieve more for complete lists
    if intent == "LIST":
        return await tool_search(question, filters, session_id, top_k=20)

    # 8. Yes/No — standard search, short answer
    if intent == "YESNO":
        return await tool_search(question, filters, session_id, top_k=10)

    # 9. Specific fact — standard search
    if intent == "SPECIFIC":
        return await tool_search(question, filters, session_id, top_k=10)

    # 10. Explanation — refine for better context
    if intent == "EXPLAIN":
        return await tool_refine(question, filters, session_id, top_k)

    # 11. Default search
    return await tool_search(question, filters, session_id, top_k)