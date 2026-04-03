"""
ticket_dedup.py — LLM-based duplicate ticket detection
=======================================================

New approach (replaces Redis + cosine similarity):
  1. Fetch all Open + "In Progress" tickets from Notion (live, always fresh)
  2. Ask the LLM if the new question is semantically the same as any existing ticket
  3. Duplicate found  → return existing ticket info, block creation
  4. No duplicate     → allow creation

Why this is better than Redis embeddings:
  - Live Notion data — no stale 1-hour caches
  - Works across server restarts
  - LLM understands intent & meaning better than cosine similarity
  - Zero Redis dependency for deduplication
  - Only compares against ACTIVE tickets (Open / In Progress) —
    Resolved tickets are ignored on purpose

Usage:
    dup = await find_duplicate(question)
    if dup:
        # {"ticket_id": ..., "ticket_url": ..., "question": ...}
        return already-exists reply
    # else: create the ticket
"""

# ── Standard library ──────────────────────────────────────────────────────────
import asyncio
import logging
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import httpx  # async Notion API calls

# ── Internal ──────────────────────────────────────────────────────────────────
from backend.core.logger import logger
from backend.services.notion_service import _headers as _notion_headers, NOTION_API_URL as NOTION_API
from backend.api.agent_routes import _get_ticket_db_id
from backend.rag.rag_service import _get_llm

# Max tickets sent to LLM to avoid huge prompts
_MAX_TICKETS_FOR_LLM = 50

_DEDUP_PROMPT = """\
You are a support ticket duplicate detector.

Below are existing OPEN or IN-PROGRESS support tickets (ID + original question):
{ticket_list}

New question from user:
\"{new_question}\"

Task: Decide if the new question is asking about the SAME TOPIC and INTENT as any existing ticket.
Count as a duplicate even if the words are different, as long as the meaning is the same.

Examples of DUPLICATES:
- "who is raju" vs "tell me about raju"         → same person lookup
- "what is notice period" vs "how long is notice period" → same policy question

Examples of NOT DUPLICATES:
- "who is raju" vs "what is leave policy"        → completely different topics
- "salary structure" vs "notice period"          → different HR topics

Reply in EXACTLY this format, nothing else:
DUPLICATE: YES
TICKET_ID: <the matching ticket id>

OR:
DUPLICATE: NO\
"""


async def _fetch_open_tickets() -> list[dict]:
    """
    Fetch Open + In Progress tickets from Notion.
    Returns list of {ticket_id, page_id, question, url}.
    Returns [] on any error (fail-open — allow ticket creation).
    """
    try:

        db_id   = _get_ticket_db_id()
        headers = _notion_headers()

        body = {
            "page_size": _MAX_TICKETS_FOR_LLM,
            "filter": {
                "or": [
                    {"property": "Status", "select": {"equals": "Open"}},
                    {"property": "Status", "select": {"equals": "In Progress"}},
                ]
            },
        }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{NOTION_API}/databases/{db_id}/query",
                headers=headers,
                json=body,
            )

        if resp.status_code == 404:
            logger.warning("[dedup] Notion DB not found (404) — skipping dedup")
            return []

        resp.raise_for_status()
        results = resp.json().get("results", [])

        tickets = []
        for page in results:
            props = page.get("properties", {})

            # Get ticket question (title field)
            q_items = (
                props.get("Question", {}).get("title", [])
                or props.get("Name", {}).get("title", [])
                or props.get("Title", {}).get("title", [])
            )
            question = "".join(t.get("plain_text", "") for t in q_items).strip()

            if not question:
                continue

            ticket_id = page["id"].replace("-", "")[:8].upper()
            tickets.append({
                "ticket_id": ticket_id,
                "page_id":   page["id"],
                "question":  question,
                "url":       page.get("url", ""),
            })

        logger.info("Fetched %d open/in-progress tickets from Notion", len(tickets))
        return tickets

    except Exception as e:
        logger.warning("Could not fetch Notion tickets: %s — skipping dedup", e)
        return []


async def _llm_duplicate_check(new_question: str, tickets: list[dict]) -> Optional[dict]:
    """
    Ask the LLM if new_question duplicates any existing ticket.
    Returns the matching ticket dict, or None if no duplicate.
    """
    try:

        # Build the ticket list string for the prompt
        ticket_lines = "\n".join(
            f"  [{t['ticket_id']}] {t['question']}"
            for t in tickets
        )
        prompt = _DEDUP_PROMPT.format(
            ticket_list=ticket_lines,
            new_question=new_question,
        )

        # Run LLM in executor (it's a sync call)
        raw = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _get_llm().invoke(prompt).content.strip()
        )

        logger.debug("[dedup] LLM response: %s", raw)

        # Parse response
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

        # Find the ticket dict for the matched ID
        for t in tickets:
            if t["ticket_id"] == matched_id:
                logger.info(
                    "✅ LLM found duplicate  ticket=%s  q='%s'",
                    matched_id, t["question"][:60],
                )
                return t

        logger.warning("LLM returned ticket_id=%s but not found in list", matched_id)
        return None

    except Exception as e:
        logger.warning("LLM duplicate check failed: %s — allowing ticket creation", e)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

async def find_duplicate(question: str) -> Optional[dict]:
    """
    Check if an Open/In-Progress Notion ticket already exists for this question.

    Flow:
      1. Fetch Open + In Progress tickets from Notion
      2. If none exist → no duplicate possible, return None immediately
      3. Ask LLM to compare new question against all existing tickets
      4. Return matching ticket or None

    Fails open: any error returns None so ticket creation is never blocked by a bug.
    """
    tickets = await _fetch_open_tickets()
    if not tickets:
        logger.info("No open tickets in Notion — no duplicate possible")
        return None

    return await _llm_duplicate_check(question, tickets)


async def flush_dedup_cache() -> None:
    """No-op — dedup now uses live Notion data, no cache to flush."""
    logger.info("[dedup] Nothing to flush — using live Notion data for dedup")