"""
ticket_dedup.py — LLM-based duplicate ticket detection
=======================================================

Prevents duplicate Notion support tickets by comparing new questions
against all currently Open / In Progress tickets using an LLM judge
rather than a vector-similarity cache.

Pipeline:
    1. Fetch all Open + In Progress tickets from Notion (always live).
    2. Ask the LLM whether the new question matches any existing ticket.
    3. If a duplicate is found, return the existing ticket and block creation.
    4. If no duplicate, allow creation to proceed.

Public API:
    find_duplicate(question)  — returns a matching ticket dict or None.
    flush_dedup_cache()       — no-op (no cache exists; included for interface compatibility).
"""

from typing import Optional

import httpx

from backend.core.logger import logger
from backend.services.notion_service import _headers as _notion_headers, NOTION_API_URL as NOTION_API
from backend.api.agent_routes import _get_ticket_db_id
from backend.core.llm import get_llm as _get_llm

# Max tickets sent to LLM to avoid huge prompts
_MAX_TICKETS_FOR_LLM = 50

_DEDUP_PROMPT = """\
You are a support ticket duplicate detector.

Below are existing OPEN or IN-PROGRESS support tickets (ID + original question):
{ticket_list}

New question from user:
"{new_question}"

Task: Decide if the new question is asking about the EXACT SAME TOPIC and INTENT as any existing ticket.
Only count as a duplicate if the user is asking about the same specific entity (person, policy, document, or comparison).

Examples of DUPLICATES:
- "who is raju" vs "tell me about raju"         → same person lookup
- "what is notice period" vs "how long is notice period" → same policy question

Examples of NOT DUPLICATES:
- "who is raju" vs "who is ramesh"            → DIFFERENT people
- "is NDA mandatory" vs "is SOW mandatory"    → DIFFERENT documents
- "salary structure" vs "notice period"          → DIFFERENT HR topics
- "who is raju" vs "what is leave policy"        → completely different topics

Reply in EXACTLY this format, nothing else:
DUPLICATE: YES
TICKET_ID: <the matching ticket id>

OR:
DUPLICATE: NO\
"""


async def _fetch_open_tickets() -> list[dict]:
    """
    Fetch all Open and In Progress tickets from Notion.

    Results are capped at `_MAX_TICKETS_FOR_LLM` to avoid oversized prompts.
    Fails open: any Notion API error returns an empty list so that ticket
    creation is never blocked by a network or configuration issue.

    Returns:
        A list of dicts with keys: ticket_id, page_id, question, url.
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

            q_items = (
                props.get("Question", {}).get("title", [])
                or props.get("Name", {}).get("title", [])
                or props.get("Title", {}).get("title", [])
            )
            question = "".join(t.get("plain_text", "") for t in q_items).strip()

            if not question:
                continue

            id_prop   = props.get("Ticket ID", {})
            id_items  = id_prop.get("rich_text", []) or id_prop.get("title", [])
            manual_id = "".join(t.get("plain_text", "") for t in id_items).strip()

            ticket_id = manual_id if manual_id else page["id"].replace("-", "")[:8].upper()
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
    Ask the LLM whether `new_question` is semantically identical to any existing ticket.

    Fails open: if the LLM call raises any exception, None is returned so that
    ticket creation is never silently blocked.

    Args:
        new_question: The unanswered question the user wants to file a ticket for.
        tickets:      List of existing open/in-progress ticket dicts from Notion.

    Returns:
        The matching ticket dict if a duplicate is found, otherwise None.
    """
    try:
        ticket_lines = "\n".join(
            f"  [{t['ticket_id']}] {t['question']}"
            for t in tickets
        )
        prompt = _DEDUP_PROMPT.format(
            ticket_list=ticket_lines,
            new_question=new_question,
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