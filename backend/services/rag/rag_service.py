"""
rag_service.py  — Production RAG Service
=========================================
Flow: Query → Embed → ChromaDB Top-K → Build Prompt (with chat history) → LLM → Answer + Citations

Features:
  - Chat history injected into every LLM call (last 4 turns)
  - Redis session cache (30 min TTL)
  - Redis retrieval cache (10 min TTL) — same query = no re-embed
  - 3 auto-detected tools: search, refine (HyDE), compare
  - Casual chat handled without RAG
  - Confidence scoring from retrieval scores
"""

import hashlib
import json
from typing import Optional
from backend.core.config import settings
from backend.core.logger import logger
from backend.services.redis_service import cache

COLLECTION_NAME = "rag_chunks"
MIN_SCORE       = 0.10
TTL_RETRIEVAL   = 600    # 10 min — cache retrieval results
TTL_SESSION     = 1800   # 30 min — cache chat history


# ── Prompts ───────────────────────────────────────────────────────────────────

ANSWER_PROMPT = """\
You are a helpful document assistant. Use only the provided context to answer.
If the context does not contain the answer, respond with:
"I could not find information about this in the available documents."

{history}

Context:
{context}

Question: {question}

Answer (cite sources as "Doc Title → Section"):"""

COMPARE_PROMPT = """\
Review the two document excerpts and answer the question for each separately.

Question: {question}

Document A - {doc_a}:
{content_a}

Document B - {doc_b}:
{content_b}

Respond in JSON format only:
{{"side_a": "answer based on Document A", "side_b": "answer based on Document B", "summary": "key difference in one sentence"}}"""

HYDE_PROMPT = """\
Write a brief factual description (2-3 sentences) about this business topic: {question}"""


# ── Clients ───────────────────────────────────────────────────────────────────

def _get_llm():
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_LLM_ENDPOINT,
        api_key=settings.AZURE_OPENAI_LLM_KEY,
        azure_deployment=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
        api_version="2024-12-01-preview",
        temperature=0.2,
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


# ── Redis helpers ─────────────────────────────────────────────────────────────

def _retrieval_key(query: str, filters: dict, top_k: int) -> str:
    raw = json.dumps({"q": query, "f": filters, "k": top_k}, sort_keys=True)
    return f"docforge:rag:retrieval:{hashlib.md5(raw.encode()).hexdigest()}"


def _session_key(session_id: str) -> str:
    return f"docforge:rag:session:{session_id}"


async def _get_history(session_id: str) -> str:
    """Get last 4 turns formatted for prompt injection."""
    data = await cache.get(_session_key(session_id)) or []
    if not data:
        return ""
    lines = ["Previous conversation:"]
    for turn in data[-4:]:
        lines.append(f"User: {turn['q']}")
        lines.append(f"Assistant: {turn['a'][:200]}...")
    return "\n".join(lines) + "\n"


async def _save_turn(session_id: str, q: str, a: str):
    """Save turn to Redis session."""
    key  = _session_key(session_id)
    data = await cache.get(key) or []
    data.append({"q": q, "a": a})
    await cache.set(key, data[-10:], ttl=TTL_SESSION)


# ── Retriever ─────────────────────────────────────────────────────────────────

async def _retrieve(query: str, filters: dict, top_k: int = 5) -> list:
    """
    Embed query → search ChromaDB → filter by score → return chunks.
    Results cached in Redis for 10 min.
    """
    key    = _retrieval_key(query, filters, top_k)
    cached = await cache.get(key)
    if cached is not None:
        logger.info("Cache HIT: retrieval")
        return cached

    embedder   = _get_embedder()
    collection = _get_collection()
    count      = collection.count()

    if count == 0:
        return []

    query_emb = embedder.embed_query(query)

    # Build metadata filter
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

    chunks    = []
    docs      = results.get("documents", [[]])[0]
    metas     = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
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

    await cache.set(key, chunks, ttl=TTL_RETRIEVAL)
    logger.info("Retrieved %d chunks (score >= %.2f)", len(chunks), MIN_SCORE)
    return chunks


def _build_context(chunks: list) -> str:
    if not chunks:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[{i}] {c['citation']}\n{c['content']}"
        for i, c in enumerate(chunks, 1))


def _citations(chunks: list) -> list:
    seen, out = set(), []
    for c in chunks:
        cit = c.get("citation", "")
        if cit and cit not in seen:
            seen.add(cit)
            out.append(cit)
    return out


def _confidence(chunks: list) -> str:
    if not chunks:
        return "low"
    avg = sum(c["score"] for c in chunks) / len(chunks)
    return "high" if avg >= 0.60 else "medium" if avg >= 0.40 else "low"


# ── Casual chat ───────────────────────────────────────────────────────────────

CASUAL_RESPONSES = {
    "greeting": "Hello! I'm CiteRAG Lab, your document assistant. Ask me anything about your company documents.",
    "howdy":    "Doing great, thanks! Ready to help. What would you like to know about your documents?",
    "thanks":   "You're welcome! Let me know if you have any other questions.",
    "bye":      "Goodbye! Come back anytime you need help with your documents.",
    "help":     "I can search your documents, compare two documents, summarise policies, or find specific clauses. Just ask!",
    "identity": "I'm CiteRAG Lab — an AI assistant that answers questions based on your company's Notion documents with citations.",
}

# General knowledge question keywords — answer with LLM directly, no RAG
GENERAL_KNOWLEDGE_PATTERNS = [
    "who is ", "what is the capital", "when was ", "where is ",
    "how many people", "population of", "president of", "prime minister",
    "chief minister", " cm of ", "governor of", "ceo of", "founded in",
    "when did ", "history of", "what year", "who invented", "who discovered",
    "largest ", "smallest ", "fastest ", "tallest ", "oldest ",
]

def _is_general_knowledge(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in GENERAL_KNOWLEDGE_PATTERNS)


def _is_casual(question: str) -> str:
    q = question.lower().strip().rstrip("!?.")
    if any(w in q for w in ["who are you", "what are you", "what can you do"]):
        return "identity"
    if q in ["help", "?", "how does this work"]:
        return "help"
    if any(q == w or q.startswith(w + " ") for w in ["hey", "hi", "hello", "hiya", "yo"]):
        return "greeting"
    if "how are you" in q or "how's it going" in q:
        return "howdy"
    if any(q.startswith(w) for w in ["thanks", "thank you", "thx"]):
        return "thanks"
    if any(q.startswith(w) for w in ["bye", "goodbye", "see you"]):
        return "bye"
    if len(q.split()) <= 2 and not any(
        w in q for w in ["what", "who", "when", "where", "why", "how",
                          "is", "are", "can", "does", "do"]):
        return "greeting"
    return ""


# ── Tools ─────────────────────────────────────────────────────────────────────

async def tool_search(question: str, filters: dict,
                      session_id: str, top_k: int = 5) -> dict:
    """Standard vector search + grounded answer with chat history."""
    chunks  = await _retrieve(question, filters, top_k)
    context = _build_context(chunks)
    history = await _get_history(session_id)
    prompt  = ANSWER_PROMPT.format(
        history=history, context=context, question=question)
    answer  = _get_llm().invoke(prompt).content.strip()
    await _save_turn(session_id, question, answer)
    # Hide citations when answer not found in documents
    not_found = "could not find" in answer.lower()
    return {
        "answer":     answer,
        "citations":  [] if not_found else _citations(chunks),
        "chunks":     [] if not_found else chunks,
        "tool_used":  "search",
        "confidence": "low" if not_found else _confidence(chunks),
    }


async def tool_refine(question: str, filters: dict,
                      session_id: str, top_k: int = 5) -> dict:
    """HyDE: generate hypothetical answer → better retrieval for vague queries."""
    hyp     = _get_llm().invoke(
        HYDE_PROMPT.format(question=question)).content.strip()
    logger.info("HyDE hypothesis: %s", hyp[:80])
    chunks  = await _retrieve(hyp, filters, top_k)
    context = _build_context(chunks)
    history = await _get_history(session_id)
    prompt  = ANSWER_PROMPT.format(
        history=history, context=context, question=question)
    answer  = _get_llm().invoke(prompt).content.strip()
    await _save_turn(session_id, question, answer)
    return {
        "answer":     answer,
        "citations":  _citations(chunks),
        "chunks":     chunks,
        "tool_used":  "refine",
        "confidence": _confidence(chunks),
    }


async def tool_compare(question: str, doc_a: str, doc_b: str,
                       filters: dict, session_id: str, top_k: int = 4) -> dict:
    """Retrieve from two docs separately and compare side-by-side."""
    import asyncio
    chunks_a, chunks_b = await asyncio.gather(
        _retrieve(question, filters, top_k * 2),
        _retrieve(question, filters, top_k * 2),
    )

    def filter_doc(chunks, title):
        f = [c for c in chunks if title.lower() in c["doc_title"].lower()]
        return f[:top_k] if f else chunks[:top_k]

    chunks_a  = filter_doc(chunks_a, doc_a)
    chunks_b  = filter_doc(chunks_b, doc_b)
    content_a = _build_context(chunks_a)
    content_b = _build_context(chunks_b)

    raw = _get_llm().invoke(
        COMPARE_PROMPT.format(
            question=question, doc_a=doc_a, doc_b=doc_b,
            content_a=content_a, content_b=content_b)
    ).content.strip()

    try:
        parsed  = json.loads(raw)
        side_a  = parsed.get("side_a", "")
        side_b  = parsed.get("side_b", "")
        summary = parsed.get("summary", "")
    except Exception:
        side_a, side_b, summary = content_a[:400], content_b[:400], raw[:200]

    all_chunks = chunks_a + chunks_b
    await _save_turn(session_id, question, summary)
    return {
        "answer":     summary,
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


# ── Auto tool detection ───────────────────────────────────────────────────────

def _detect_tool(question: str, doc_a: str, doc_b: str) -> str:
    if doc_a and doc_b:
        return "compare"
    q = question.lower()
    if any(w in q for w in ["compare", "difference between", "vs ", "versus", "contrast"]):
        return "compare"
    if any(w in q for w in ["summarise", "summarize", "overview", "explain",
                             "what is", "tell me about", "describe"]):
        return "refine"
    return "search"


# ── Main dispatcher ───────────────────────────────────────────────────────────

async def answer(
    question:   str,
    filters:    Optional[dict] = None,
    session_id: str = "default",
    top_k:      int = 5,
    doc_a:      str = "",
    doc_b:      str = "",
) -> dict:
    filters = filters or {}

    # 1. Casual chat — no RAG
    casual_type = _is_casual(question)
    if casual_type:
        response = CASUAL_RESPONSES.get(casual_type, CASUAL_RESPONSES["greeting"])
        await _save_turn(session_id, question, response)
        return {
            "answer":     response,
            "citations":  [],
            "chunks":     [],
            "tool_used":  "chat",
            "confidence": "high",
        }

    # 2. General knowledge — answer with LLM directly, no RAG
    if _is_general_knowledge(question):
        gk_prompt = f"Answer this general knowledge question briefly and accurately: {question}"
        gk_answer = _get_llm().invoke(gk_prompt).content.strip()
        await _save_turn(session_id, question, gk_answer)
        return {
            "answer":     gk_answer,
            "citations":  [],
            "chunks":     [],
            "tool_used":  "general",
            "confidence": "high",
        }

    # 3. RAG — auto-detect tool
    tool = _detect_tool(question, doc_a, doc_b)
    if tool == "compare":
        return await tool_compare(
            question, doc_a or "Document A", doc_b or "Document B",
            filters, session_id, top_k)
    elif tool == "refine":
        return await tool_refine(question, filters, session_id, top_k)
    else:
        return await tool_search(question, filters, session_id, top_k)