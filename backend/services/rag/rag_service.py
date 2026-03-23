"""
rag_service.py — Smart RAG Service
Query → Understand intent → Right tool → Retrieve → LLM → Clean answer
"""

import hashlib
import json
import asyncio
from typing import Optional
from backend.core.config import settings
from backend.core.logger import logger
from backend.services.redis_service import cache

COLLECTION_NAME = "rag_chunks"
MIN_SCORE       = 0.10
TTL_RETRIEVAL   = 600
TTL_SESSION     = 1800
TTL_ANSWER      = 3600   # 1 hour — final LLM answer cache


# ── Prompts ───────────────────────────────────────────────────────────────────

ANSWER_PROMPT = """\
You are CiteRAG — a precise legal and business document analyst for turabit.
Use ONLY the context below. Do NOT use outside knowledge.
If the answer is not in the context, say exactly:
"I could not find information about this in the available documents."

{history}

Context:
{context}

Question: {question}

CRITICAL RULES:
1. Start with FINAL ANSWER — a direct YES/NO or 1-sentence verdict
2. Then provide supporting analysis with specific document names and sections
3. Never use vague labels — always give ACTUAL content (numbers, names, dates, conditions)
4. Cross-check ALL documents, not just the first match
5. Always flag: undefined terms (reasonable, promptly, material breach, good faith)
6. Always flag: missing standard clauses (indemnity, liability cap, force majeure, dispute resolution)
7. For YES/NO questions — still provide evidence and flag risks even if answer is YES
8. Never stop at 1 sentence — always provide structured analysis

OUTPUT FORMAT by question type:

FINAL ANSWER
[YES/NO or direct answer — 1-2 sentences]

[Then one of these structures:]

Single fact → exact value + document reference

What/How/Why → 2-5 sentences with specific document citations

List → bullet points with document references per item

Analysis → use CONTRADICTIONS / INCONSISTENCIES / GAPS / AMBIGUITIES sections

Comparison → per-document breakdown then SUMMARY

Answer:"""

COMPARE_PROMPT = """\
You are CiteRAG — a senior document analyst for turabit.
Compare the two documents on the question below. Follow all 4 steps.

STEP 1 — SCOPE CHECK:
Do the retrieved documents actually match what the question asks?
- If the documents are NOT the type asked for (e.g. question asks SOW vs NDA but retrieved docs are something else):
  → Explicitly state: "The documents retrieved are [X] and [Y], not [asked type]."
  → Then proceed to analyze what IS available.

STEP 2 — PER-DOCUMENT FINDINGS:
For each document answer the question using ONLY its content.
State what is present, what is absent, and what is ambiguous.

STEP 3 — GAP & RISK (if a clause or section is missing):
Identify what is missing, the legal/operational risk, and severity.

STEP 4 — COMPARISON INSIGHT:
State expected best practice vs actual finding, with a fix.

Question: {question}

Content from {doc_a}:
{content_a}

Content from {doc_b}:
{content_b}

Respond in this EXACT format:

FINAL ANSWER
[1-2 sentences. Direct answer. If comparison not possible, state why explicitly.]

DOCUMENT A -- {doc_a}
[Findings: specific facts, numbers, dates. State explicitly if clause is missing.]

DOCUMENT B -- {doc_b}
[Findings: specific facts, numbers, dates. State explicitly if clause is missing.]

COMPARISON TABLE
| Aspect | {doc_a} | {doc_b} |
|---|---|---|
| [Key aspect 1] | [finding] | [finding] |
| [Key aspect 2] | [finding] | [finding] |
| [Key aspect 3] | [finding] | [finding] |

GAP IDENTIFIED:
What: [what is missing or problematic]
Where: [document and section]
Risk:
- [specific legal impact]
- [specific legal impact]
Severity: [🔴 HIGH / 🟡 MEDIUM / 🟢 LOW]
Severity Reason: [1 sentence why this severity]

KEY DIFFERENCE:
[state the actual difference, or "No substantive difference" if same]

SYSTEMIC ISSUE (if applicable):
[If operational docs used instead of formal legal agreements, state it]

COMPARISON INSIGHT:
Expected: [best practice]
Actual: [what was found]
Fix: [concrete recommendation]

SUMMARY: [2-3 sentences covering scope issues, main findings, and recommended action.]"""

HYDE_PROMPT = """\
Write a brief factual description (2-3 sentences) about this business topic: {question}"""

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
- Short, structured, scannable — not a paragraph essay

Summary:"""

EXPAND_PROMPT = """\
Rewrite this question in 3 different ways using different words and synonyms that mean the same thing.
Keep each version short (under 15 words). Return only the 3 versions, one per line, no numbering.

Question: {question}"""

REWRITE_PROMPT = """\
You are a query understanding assistant for a company legal document system at turabit.
The user asked: "{question}"

Your job:
1. Understand what the user REALLY wants
2. Rewrite it as a clear, precise question that will find the right document content
3. Identify the intent type

REWRITING RULES:
- Fix typos and informal language
- Expand abbreviations (HR → Human Resources, IP → Intellectual Property)
- Make vague questions specific
- For legal/contract questions, always include: contract agreement clause legal terms
- For abstract questions, rewrite to find concrete document content

MANDATORY REWRITES (use these exact patterns):
- "do notice periods create any conflicts or risks" → "termination notice period 30 days 60 days conflict risk contracts agreements"
- "does the document follow a logical structure" → "document structure sections headings organization format layout contracts"
- "is there a hierarchy between related agreements" → "master agreement MSA parent child precedence governance supersedes framework"
- "is there a clause hierarchy or precedence rule" → "agreement precedence rule supersedes clause hierarchy governing order MSA"
- "are key terms properly defined" → "undefined key terms material breach reasonable period promptly force majeure good faith definitions contracts"
- "are definitions used consistently" → "consistent definitions key terms material breach reasonable promptly undefined contracts legal agreements"
- "are enforcement mechanisms strong enough" → "enforcement mechanisms penalties financial liability audit compliance contracts legal"
- "are roles and responsibilities clearly defined" → "roles responsibilities RACI accountability defined contracts agreements vendor employment"
- "does this agreement align with industry best practices" → "industry best practices indemnity liability force majeure dispute resolution standards contracts"
- "does the agreement scale well for future changes" → "amendment modification renewal scalability future changes contracts agreements"
- "are there any one-sided or unfair clauses" → "one-sided unfair clauses liability cap indemnity termination fees compensation contracts"
- "does the contract expose one party to excessive liability" → "excessive liability cap indemnity limitation damages force majeure contracts"
- "is there a fair exit mechanism" → "exit mechanism termination notice period fees severance post-termination obligations contracts"
- "are tax responsibilities clearly assigned" → "tax responsibilities GST TDS withholding income tax contracts vendor employment assignment"
- "are penalties or late fees properly defined" → "penalties late fees payment terms interest rate defined contracts invoices"
- "are termination rights clearly defined" → "termination rights notice period grounds conditions both parties contracts"

INTENT CLASSIFICATION RULES — read carefully:
- GENERAL → code generation, math problems, creative writing (poems/jokes/emails), general world knowledge, tech tutorials, how-to guides — anything NOT about turabit company documents
- GREETING → hello, hi, thanks, bye, who are you
- ANALYSIS → review/audit/gaps/contradictions/issues/risks INSIDE the turabit documents
- COMPARE → compare two specific turabit documents against each other
- SUMMARY → summarize a specific turabit document or policy
- YESNO → yes/no question about document content
- SPECIFIC → specific fact lookup inside documents
- LIST → list items from documents
- EXPLAIN → explain something from the documents
- SEARCH → general search inside documents

EXAMPLES — GENERAL intent:
- "write fibonacci in java" → GENERAL
- "write me a python function to reverse a string" → GENERAL
- "explain how neural networks work" → GENERAL
- "what is the capital of france" → GENERAL
- "write me an email to my manager asking for leave" → GENERAL
- "calculate compound interest formula" → GENERAL
- "tell me a joke" → GENERAL
- "what is docker" → GENERAL
- "how do i use git rebase" → GENERAL
- "write a poem about rain" → GENERAL

EXAMPLES — DOCUMENT intent (anything else):
- "what is the notice period in our NDA?" → SPECIFIC
- "are there any conflicting clauses?" → ANALYSIS
- "compare SOW vs employment contract" → COMPARE
- "summarize the vendor agreement" → SUMMARY

Reply in this exact format:
REWRITTEN: [the clear precise question]
INTENT: [one of: GREETING, GENERAL, COMPARE, FULL_DOC, SUMMARY, LIST, YESNO, SPECIFIC, EXPLAIN, ANALYSIS, SEARCH]"""

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
- Be specific: quote exact wording, name exact sections and documents
- Cross-check ALL documents, not just the first match
- Flag undefined terms (reasonable, promptly, material breach) as AMBIGUITIES
- Flag missing standard clauses (indemnity, liability cap, force majeure) as GAPS

Analysis:"""

_GENERAL_SYSTEM_PROMPT = """\
You are a helpful AI assistant — like ChatGPT, but also integrated with a company document system.
For this query, NO document search is needed. Answer directly from your knowledge.

RULES:
- CODE: write clean, working, commented code inside markdown code blocks with the correct language tag (e.g. ```java, ```python)
- MATH: show step-by-step working clearly
- EXPLANATIONS: be clear, use examples, keep it concise
- CREATIVE WRITING: be creative and engaging (poems, jokes, stories, emails)
- GENERAL KNOWLEDGE: be accurate and concise
- Always use markdown formatting when it improves readability
- Be conversational and friendly
- Never say "I can only answer document questions" — answer everything the user asks"""









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


def _answer_key(question: str, filters: dict) -> str:
    """
    Cache key for final LLM answer.
    Includes filters so different departments/doc_types never share answers.
    Safety rule 1: key = hash(question + filters), NOT just question.
    """
    raw = json.dumps({"q": question.strip().lower(), "f": filters}, sort_keys=True)
    return f"docforge:rag:answer:{hashlib.md5(raw.encode()).hexdigest()}"


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


async def _retrieve(query: str, filters: dict, top_k: int = 8) -> list:
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
                    variant, filters, 4, embedder, collection)
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

    # Diversity check: if all chunks from same doc, try to get more variety
    if final:
        unique_titles = {c["doc_title"] for c in final}
        if len(unique_titles) == 1:
            logger.info("Low diversity: all %d chunks from '%s', expanding...",
                        len(final), next(iter(unique_titles)))
            diverse_queries = [
                f"{query} employment contract",
                f"{query} vendor agreement",
                f"{query} sales NDA",
            ]
            seen = {c["notion_page_id"] + c["heading"] for c in final}
            for dq in diverse_queries:
                extra = await _retrieve_single(dq, {}, 3, embedder, collection)
                for c in extra:
                    uid = c["notion_page_id"] + c["heading"]
                    if uid not in seen and c["doc_title"] not in unique_titles:
                        seen.add(uid)
                        unique_titles.add(c["doc_title"])
                        final.append(c)
            final = sorted(final, key=lambda x: x["score"], reverse=True)[:top_k]
            logger.info("After diversity: %d chunks from %d docs",
                        len(final), len({c["doc_title"] for c in final}))

    await cache.set(key, final, ttl=TTL_RETRIEVAL)
    logger.info("Retrieved %d chunks (with expansion) for: %s", len(final), query[:50])
    return final


def _build_context(chunks: list) -> str:
    if not chunks:
        return "No relevant documents found."
    # Only include chunks with score >= 0.20 to avoid noise
    quality = [c for c in chunks if c.get("score", 0) >= 0.20]
    if not quality:
        quality = chunks[:5]  # fallback: take top 5 regardless of score
    return "\n\n---\n\n".join(
        f"Source: {c['citation']}\n{c['content']}"
        for c in quality)


def _citations(chunks: list) -> list:
    """Only return citations for chunks actually used (top 8 by score max)."""
    seen, out = set(), []
    # Sort by score and take only top chunks — these are what LLM actually used
    top_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)[:8]
    for c in top_chunks:
        cit     = c.get("citation", "")
        page_id = c.get("notion_page_id", "")
        url     = f"https://www.notion.so/{page_id}" if page_id else ""
        # Deduplicate by doc_title + heading combo
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

    # ── General knowledge / code / creative (not in documents) ────────────────
    if any(p in q for p in ["who is the president", "what is the capital",
                              "prime minister of", "chief minister of", " cm of ",
                              "ceo of ", "history of ", "population of",
                              "who invented", "who discovered", "largest country",
                              "when was born", "tallest building"]):
        return "GENERAL"

    # Code generation — "write a fibonacci", "write me a function", etc.
    _doc_words = ["policy", "contract", "letter", "offer", "document",
                  "clause", "agreement", "nda", "sow", "handbook",
                  "employee", "leave", "salary", "hr", "turabit"]
    if any(q.startswith(p) for p in ["write a", "write the", "write me",
                                      "code a", "code the", "program a",
                                      "create a function", "implement a",
                                      "give me a function", "build a function"]):
        if not any(w in q for w in _doc_words):
            return "GENERAL"

    if any(p in q for p in [
        # Code / algorithms
        "fibonacci", "factorial", "sorting algorithm", "binary search",
        "linked list", "data structure", "write code", "python code",
        "java code", "javascript code", " in python", " in java",
        " in javascript", " in c++", " in golang", " in typescript",
        "function that ", "algorithm for", "regex for", "sql query for",
        "write a program", "write a script", "write a class",
        # Math
        "calculate ", "solve for", "integral of", "derivative of",
        "what is 2+", "what is 3+", "square root of",
        # Creative writing
        "write a poem", "write a joke", "tell me a joke",
        "write a story", "write an essay", "write a haiku",
        "write an email to", "draft an email", "write a message to",
        "write a cover letter for", "write a resignation",
        # General how-to (tech, not document related)
        "how to install", "how to setup", "how to configure",
        "how to use git", "how to deploy", "how to fix error",
        # Conversational / opinion
        "what do you think about", "what is your opinion",
        "recommend a book", "suggest a movie", "best way to learn",
        # General CS / tech concepts
        "what is machine learning", "what is artificial intelligence",
        "what is blockchain", "explain quantum", "what is photosynthesis",
        "how does the internet", "what is tcp/ip", "what is http",
        "what is rest api", "what is docker", "what is kubernetes",
        # Template writing — write a generic template, not search documents
        "write a template", "write an nda template", "write a contract template",
        "draft a template", "create a template", "make a template",
        "write a sample contract", "write a sample agreement",
        # Translation tasks
        "translate to ", "translate the ", "translate this ",
        "convert to hindi", "convert to spanish", "convert to french",
        " in hindi", " in spanish", " in french", " in german",
        " in gujarati", " in marathi",
    ]):
        if not any(w in q for w in _doc_words):
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
        "is there a lack", "is there an absence", "insufficiently",
        "are provisions", "are there missing", "lack of",
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
        # Liability / fairness
        "expose one party", "excessive liability", "one-sided",
        "unfair clause", "disproportionate", "favor one party",
        "unintentionally favor",
        # Terms / definitions
        "key terms properly", "properly defined", "subjective terms",
        "reasonable or promptly", "multiple interpretations",
        "could any clause", "are there vague", "vague terms",
        "defined consistently", "used consistently",
        "definitions used", "terms defined", "defined throughout",
        "defined across", "consistently defined", "consistently used",
        # Duration / timeline
        "duration of confidentiality", "is the duration",
        "clearly defined for both", "timelines aligned",
        "deadlines aligned", "durations aligned",
        # Exit / termination
        "fair exit", "exit mechanism", "notice periods create",
        "notice period create", "notice period conflict",
        "notice periods conflict", "do notice", "notice period risk",
        "triggered arbitrarily", "misused", "be bypassed",
        # Enforcement / penalties
        "enforcement mechanisms", "strong enough",
        "sufficient to deter", "penalties sufficient",
        "penalties defined", "late fees", "tax responsibilities",
        "tax responsibility", "clearly assigned",
        "payment penalties", "properly defined",
        # Structure
        "logical structure", "follow a logical", "hierarchy between",
        "clause hierarchy", "precedence rule", "scale well",
        "roles and responsibilities", "cross-references",
        "master agreement governing", "align with industry",
        "scale for future",
        # IP / data
        "intellectual property rights", "ip rights",
        "data protection obligations", "regulatory compliance gaps",
        # Fairness / balance
        "fair and enforceable", "clearly stated", "clearly defined",
        "one-sided", "favor one", "unintentionally",
        "proportionate", "balanced",
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
                      session_id: str, top_k: int = 8) -> dict:
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
    chunks  = await _retrieve(question, filters, top_k=15)
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
    # Use dedicated SUMMARY_PROMPT for structured, scannable output
    answer  = _get_llm().invoke(
        SUMMARY_PROMPT.format(context=context, question=question)
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

    # Retrieve separately with doc-specific filters
    # Search with doc name embedded in query for better matching
    _boost = "contract agreement clause legal terms obligations"
    query_a = f"{question} {_boost} {doc_a}"
    query_b = f"{question} {_boost} {doc_b}"

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
    await _save_turn(session_id, question, summary or raw[:200])
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
        "confidence": _confidence(all_chunks),
    }


async def tool_analysis(question: str, filters: dict,
                        session_id: str) -> dict:
    """
    For analysis questions — retrieve broad context then reason over it.
    Key insight: never search for 'contradictions' in vector DB —
    instead retrieve actual document content broadly, then let LLM analyze it.
    """
    # Step 1: Build smart retrieval queries based on question type
    # Key insight: for legal/contract questions, always boost with contract keywords
    # For Q28 "notice periods" — force retrieval from contract termination sections
    q_lower = question.lower()

    # Detect legal/contract questions and build targeted query
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
        # For legal questions: retrieve from contract documents specifically
        contract_boost = "contract agreement clause legal terms obligations termination"
        primary_query = f"{question} {contract_boost}"
    else:
        primary_query = question

    # Step 2: Retrieve broadly using subject keywords
    chunks = await _retrieve(primary_query, filters, top_k=15)

    # Step 3: For legal questions, also retrieve with specific contract focus
    if is_legal and len(chunks) < 10:
        # Try alternative retrieval with pure legal terms
        legal_queries = [
            f"termination notice period contract agreement {question}",
            f"clause obligation legal contract {question}",
            f"employment vendor sales NDA agreement {question}",
        ]
        seen = {c["notion_page_id"] + c["heading"] for c in chunks}
        for lq in legal_queries[:2]:
            extra = await _retrieve(lq, {}, top_k=5)
            for c in extra:
                uid = c["notion_page_id"] + c["heading"]
                if uid not in seen:
                    seen.add(uid)
                    chunks.append(c)

    # Step 3: If still empty, retrieve everything available
    # Always do a second pass for legal/cross-document analysis
    # This ensures we get diverse document types not just the closest semantic match
    legal_contract_queries = [
        "employment contract termination confidentiality obligations",
        "vendor contract payment liability dispute resolution",
        "sales agreement NDA governing law jurisdiction",
        "service agreement indemnity force majeure clause",
    ]
    seen = {c["notion_page_id"] + c["heading"] for c in chunks}
    for lq in legal_contract_queries:
        extra = await _retrieve(lq, {}, top_k=4)
        for c in extra:
            uid = c["notion_page_id"] + c["heading"]
            if uid not in seen:
                seen.add(uid)
                chunks.append(c)

    if not chunks:
        broad_queries = [
            "company policy employee handbook",
            "terms conditions agreement contract",
            "authorization approval procedure",
            "employment HR leave salary",
        ]
        seen2 = set()
        for q in broad_queries:
            extra = await _retrieve(q, {}, top_k=4)
            for c in extra:
                uid = c["notion_page_id"] + c["heading"]
                if uid not in seen2:
                    seen2.add(uid)
                    chunks.append(c)

    if not chunks:
        return {
            "answer":     "No documents found in the knowledge base to analyze. Please run ingest first.",
            "citations":  [],
            "chunks":     [],
            "tool_used":  "analysis",
            "confidence": "low",
        }

    # Sort by score and take best 35 for rich context
    chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
    context = _build_context(chunks[:20])

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


def _get_general_llm():
    """Shared LLM client for GENERAL (ChatGPT-style) responses. Temperature 0.7."""
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_LLM_ENDPOINT,
        api_key=settings.AZURE_OPENAI_LLM_KEY,
        azure_deployment=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
        api_version="2024-12-01-preview",
        temperature=0.7,
        max_tokens=3000,
    )


async def _call_general_llm(question: str, session_id: str) -> dict:
    """
    Shared handler for GENERAL (ChatGPT-style) queries.
    Used by both the hard-guard and the GENERAL intent branch in answer().
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    history = await _get_history(session_id)
    user_content = f"{history}\n{question}".strip() if history else question
    ans = _get_general_llm().invoke([
        SystemMessage(content=_GENERAL_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]).content.strip()
    await _save_turn(session_id, question, ans)
    return {
        "answer":     ans,
        "citations":  [],
        "chunks":     [],
        "tool_used":  "general",
        "confidence": "high",
    }


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

    # ── Cache check on ORIGINAL question — BEFORE rewrite (Fix 3) ────────────
    # This means 2nd+ identical queries skip the 2s rewrite LLM call entirely
    # Key includes filters so HR dept never gets Finance dept cached answer
    _q_lower = question.lower().strip()
    _skip_cache = any(_q_lower.startswith(w) for w in
                      ["hey", "hi", "hello", "thanks", "bye", "ok", "okay"])
    if not _skip_cache and not doc_a and not doc_b:
        _akey_orig = _answer_key(question, filters)
        _cached_orig = await cache.get(_akey_orig)
        if _cached_orig is not None:
            logger.info("Answer cache HIT (pre-rewrite) for: %s", question[:50])
            return _cached_orig
    # Catches code/math/creative queries that must NEVER hit RAG
    _q = question.lower().strip()
    _doc_signals = ["policy", "contract", "agreement", "nda", "sow", "clause",
                    "document", "employee", "leave", "salary", "hr", "turabit",
                    "handbook", "offer letter", "notice period", "vendor"]
    _hard_general_patterns = [
        "fibonacci", "factorial", "merge sort", "quick sort", "bubble sort",
        "binary search", "linked list", "stack overflow", "write code",
        "write a program", "write a script", "write a function", "write a class",
        "code for ", "program for ", "algorithm for ",
        " in python", " in java", " in javascript", " in c++", " in golang",
        " in typescript", " in ruby", " in swift", " in kotlin",
        "python code", "java code", "javascript code",
        "tell me a joke", "write a poem", "write a haiku", "write a story",
        "write an essay", "draft an email to", "write an email to",
        "square root", "prime number", "calculate ", "integrate ",
        "what is docker", "what is kubernetes", "what is git",
        "what is machine learning", "what is deep learning", "what is ai",
        "what is blockchain", "explain quantum", "how does tcp",
        "how to install", "how to setup", "how to deploy",
        # Fix 1 — template writing tasks
        "write a template", "write an nda template", "write a contract template",
        "write a non-disclosure agreement template", "write an agreement template",
        "draft a template", "create a template", "make a template",
        "write a sample contract", "write a sample agreement",
        # Fix 2 — translation tasks
        "translate to ", "translate the ", "translate this ",
        "convert to hindi", "convert to spanish", "convert to french",
        "convert to gujarati", "convert to marathi", "convert to tamil",
        " in hindi", " in spanish", " in french", " in german",
        " in gujarati", " in marathi", " in tamil", " in telugu",
    ]
    if any(p in _q for p in _hard_general_patterns):
        if not any(s in _q for s in _doc_signals):
            logger.info("Hard GENERAL guard triggered for: %s", question[:50])
            result = await _call_general_llm(question, session_id)
            if not _skip_cache:
                await cache.set(_akey_orig, result, ttl=TTL_ANSWER)
            return result

    # Use rule-based for obvious cases (fast, no LLM call)
    quick_intent = _classify_intent(question)
    if quick_intent == "GREETING":
        intent, rewritten = "GREETING", question
    elif quick_intent == "GENERAL":
        intent, rewritten = "GENERAL", question
    elif quick_intent == "ANALYSIS":
        # NEVER rewrite analysis questions — LLM rewrite can downgrade to YESNO/SPECIFIC
        # Pass original question directly to tool_analysis which handles its own query building
        intent, rewritten = "ANALYSIS", question
    elif quick_intent in ("YESNO", "SPECIFIC"):
        # For yes/no and specific: use LLM rewrite BUT preserve ANALYSIS if LLM upgrades it
        rewritten, llm_intent = await _rewrite_query(question)
        # LLM can upgrade YESNO→ANALYSIS but never downgrade ANALYSIS
        intent = llm_intent if llm_intent != "YESNO" or quick_intent == "YESNO" else quick_intent
        # If rewritten question now sounds analytical, force ANALYSIS
        if _classify_intent(rewritten) == "ANALYSIS":
            intent = "ANALYSIS"
            rewritten = question  # Use original for analysis
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

    # 2. General — answer with LLM directly (ChatGPT-style, no RAG)
    if intent == "GENERAL":
        result = await _call_general_llm(question, session_id)
        if not _skip_cache:
            await cache.set(_akey_orig, result, ttl=TTL_ANSWER)
        return result

    # 3. Compare — skip cache (doc_a/doc_b already excluded above)
    if intent == "COMPARE" or (doc_a and doc_b):
        return await tool_compare(
            question, doc_a or "Document A", doc_b or "Document B",
            filters, session_id, top_k)

    # 4. Full document
    if intent == "FULL_DOC":
        result = await tool_full_doc(question, filters, session_id)
        if not _skip_cache:
            await cache.set(_akey_orig, result, ttl=TTL_ANSWER)
        return result

    # 5. Summary / Overview
    if intent == "SUMMARY":
        result = await tool_refine(question, filters, session_id, top_k)
        if not _skip_cache:
            await cache.set(_akey_orig, result, ttl=TTL_ANSWER)
        return result

    # 6. Analysis (contradictions, gaps, issues, review)
    if intent == "ANALYSIS":
        result = await tool_analysis(question, filters, session_id)
        if not _skip_cache:
            await cache.set(_akey_orig, result, ttl=TTL_ANSWER)
        return result

    # 7. List — retrieve more for complete lists
    if intent == "LIST":
        result = await tool_search(question, filters, session_id, top_k=12)
    # 8. Yes/No — standard search, short answer
    elif intent == "YESNO":
        result = await tool_search(question, filters, session_id, top_k=6)
    # 9. Specific fact — standard search
    elif intent == "SPECIFIC":
        result = await tool_search(question, filters, session_id, top_k=6)
    # 10. Explanation — refine for better context
    elif intent == "EXPLAIN":
        result = await tool_refine(question, filters, session_id, top_k)
    # 11. Default search
    else:
        result = await tool_search(question, filters, session_id, top_k)

    # ── Save answer to cache using ORIGINAL question key ─────────────────────
    # Using _akey_orig ensures the pre-rewrite cache check hits next time
    # Same question typed again → instant, no rewrite LLM call needed
    if not _skip_cache and not doc_a and not doc_b:
        await cache.set(_akey_orig, result, ttl=TTL_ANSWER)
        logger.info("Answer cached for: %s", question[:50])

    return result