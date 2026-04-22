"""
FastAPI routes for the CiteRAG Agent layer.

Handles support ticket lifecycle management (create, list, update)
and session memory persistence via Redis. Ticket data is stored in
a dedicated Notion database; the route layer provides both an HTTP
interface and internal helpers used directly by the agent graph.

Route prefix: /api/agent/

Endpoints:
    GET    /tickets          — List all tickets (60-second Redis cache)
    POST   /tickets/update   — Update ticket status (Open / In Progress / Resolved)
    POST   /memory           — Save or merge session context into Redis
    POST   /ticket/create    — Create a new Notion support ticket
    DELETE /dedup/flush      — No-op (dedup uses live Notion data)
"""

from datetime import datetime, timezone
from typing import Optional, List
import string
import random
import httpx as _httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.config import settings
from backend.core.logger import logger
from backend.services.redis_service import cache

router = APIRouter(prefix="/agent", tags=["Agent"])

NOTION_API        = "https://api.notion.com/v1"
NOTION_VER        = "2022-06-28"
TICKETS_CACHE_KEY = "docforge:agent:tickets"


class TicketUpdateRequest(BaseModel):
    """Schema for updating the status of an existing Notion support ticket."""

    ticket_id: str
    status:    str


class TicketCreateRequest(BaseModel):
    """Schema for creating a new Notion support ticket from the CiteRAG agent context."""

    question:          str
    session_id:        str       = "default"
    attempted_sources: List[str] = []
    summary:           str       = ""
    priority:          str       = "Medium"
    confidence:        str       = "low"
    user_info:         str       = "Anonymous"
    ticket_id:         Optional[str] = None
    raw_chunks:        list      = []


class MemorySaveRequest(BaseModel):
    """Schema for saving dynamic session memory (such as user name or role) to Redis."""

    session_id: str
    memory:     dict


def _notion_headers() -> dict:
    """
    Build the HTTP headers required by the Notion API.

    Reads from `NOTION_TOKEN` or `NOTION_API_KEY` in settings.

    Returns:
        A dict with Authorization, Content-Type, and Notion-Version headers.

    Raises:
        ValueError: If neither token setting is configured.
    """
    token = (
        getattr(settings, "NOTION_TOKEN", "")
        or getattr(settings, "NOTION_API_KEY", "")
    )
    if not token:
        raise ValueError(
            "Notion token not set. Add NOTION_TOKEN to your .env file."
        )
    return {
        "Authorization":  f"Bearer {token}",
        "Content-Type":   "application/json",
        "Notion-Version": NOTION_VER,
    }


def _get_ticket_db_id() -> str:
    """
    Return the Notion database ID for ticket storage.

    Prefers `NOTION_TICKET_DB_ID`; falls back to the primary
    `NOTION_DATABASE_ID` if the dedicated ticket DB is not configured.

    Returns:
        The Notion database ID string.
    """
    return getattr(settings, "NOTION_TICKET_DB_ID", None) or settings.NOTION_DATABASE_ID


def _page_to_ticket(page: dict) -> dict:
    """
    Convert a raw Notion page object into a structured ticket dict.

    Args:
        page: A raw Notion page object from the REST API response.

    Returns:
        A flat dict with ticket_id, page_id, url, question, status,
        priority, summary, session_id, attempted_sources, created,
        assigned_owner, user_info, and created_time.
    """
    props = page.get("properties", {})

    def _text(key: str) -> str:
        prop  = props.get(key, {})
        ptype = prop.get("type", "")
        items = prop.get("title", []) if ptype == "title" else prop.get("rich_text", [])
        return "".join(t.get("plain_text", "") for t in items)

    def _select(key: str) -> str:
        sel = props.get(key, {}).get("select") or {}
        return sel.get("name", "")

    def _multi(key: str) -> list:
        return [o.get("name", "") for o in props.get(key, {}).get("multi_select", [])]

    def _date(key: str) -> str:
        d = props.get(key, {}).get("date") or {}
        return d.get("start", "")

    manual_id = _text("Ticket ID")
    ticket_id = manual_id if manual_id else page.get("id", "").replace("-", "")[:8].upper()

    return {
        "ticket_id":         ticket_id,
        "page_id":           page.get("id", ""),
        "url":               page.get("url", ""),
        "question":          _text("Question"),
        "status":            _select("Status"),
        "priority":          _select("Priority"),
        "summary":           _text("Summary"),
        "session_id":        _text("Session ID"),
        "attempted_sources": _multi("Attempted Sources"),
        "created":           _date("Created"),
        "assigned_owner":    _text("Assigned Owner"),
        "user_info":         _text("User Info"),
        "created_time":      page.get("created_time", ""),
    }


async def _fetch_session_tickets(session_id: str) -> list[dict]:
    """
    Query Notion for all open tickets belonging to a specific session.

    Used by the agent graph as a fallback when session memory is empty —
    for example, after a server restart or when tickets were created via
    the Notion UI rather than through the chat interface.

    Args:
        session_id: The session identifier to filter by.

    Returns:
        A list of ticket dicts, or an empty list on any error (fail-open
        so ticket updates are never silently blocked).
    """
    try:
        db_id = _get_ticket_db_id()
        body  = {
            "page_size": 100,
            "filter": {
                "property": "Session ID",
                "rich_text": {"equals": session_id},
            },
            "sorts": [{"timestamp": "created_time", "direction": "descending"}],
        }
        async with _httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{NOTION_API}/databases/{db_id}/query",
                headers=_notion_headers(),
                json=body,
            )
            resp.raise_for_status()

        pages   = resp.json().get("results", [])
        tickets = [_page_to_ticket(p) for p in pages]
        logger.info("_fetch_session_tickets: found %d for session=%s", len(tickets), session_id)
        return tickets

    except Exception as e:
        logger.warning("_fetch_session_tickets failed for session=%s: %s", session_id, e)
        return []


async def _create_notion_ticket(req: TicketCreateRequest) -> dict:
    """
    Core ticket-creation logic, shared by the HTTP route and the agent graph.

    Builds the Notion page payload, optionally attaches retrieved context
    snippets as callout blocks, and invalidates the ticket cache after
    a successful create.

    Args:
        req: A `TicketCreateRequest` with question, session, and metadata.

    Returns:
        A dict with `success`, `ticket_id`, `page_id`, and `url`.

    Raises:
        `httpx.HTTPStatusError`: If the Notion API returns an error response.
    """
    db_id     = _get_ticket_db_id()
    headers   = _notion_headers()
    priority  = req.priority if req.priority in {"High", "Medium", "Low"} else "Medium"
    live_date = datetime.now(timezone.utc).isoformat()

    ticket_id = req.ticket_id or "".join(random.choices(string.digits, k=8))

    properties: dict = {
        "Question":  {"title":     [{"text": {"content": req.question[:2000]}}]},
        "Status":    {"select":    {"name": "Open"}},
        "Priority":  {"select":    {"name": priority}},
        "User Info": {"rich_text": [{"text": {"content": req.user_info or "Anonymous"}}]},
        "Created":   {"date":      {"start": live_date[:10]}},
        "Assigned Owner": {"rich_text": [{"text": {"content": "Support Team"}}]},
        "Ticket ID": {"rich_text": [{"text": {"content": str(ticket_id)}}]},
    }

    if req.summary:
        properties["Summary"] = {"rich_text": [{"text": {"content": req.summary[:2000]}}]}
    if req.session_id:
        properties["Session ID"] = {"rich_text": [{"text": {"content": req.session_id}}]}
    if req.attempted_sources:
        properties["Attempted Sources"] = {
            "multi_select": [{"name": s[:100]} for s in req.attempted_sources[:10]]
        }

    payload: dict = {"parent": {"database_id": db_id}, "properties": properties}

    if req.raw_chunks:
        children = [
            {
                "object": "block", "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Attempted Context (Snippets)"}}]},
            }
        ]
        for c in req.raw_chunks[:5]:
            children.append({
                "object": "block", "type": "callout",
                "callout": {
                    "rich_text": [
                        {"text": {"content": f"Source: {c.get('doc_id', 'Unknown')}\n\n"},
                         "annotations": {"bold": True}},
                        {"text": {"content": c.get("text", "")[:1500]}},
                    ],
                    "icon": {"type": "emoji", "emoji": "📄"},
                },
            })
        payload["children"] = children

    async with _httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{NOTION_API}/pages", headers=headers, json=payload)

    resp.raise_for_status()
    page = resp.json()

    await cache.flush_pattern(f"{TICKETS_CACHE_KEY}*")
    logger.info("Ticket created: %s for: %s", ticket_id, req.question[:60])

    result = {
        "success":   True,
        "ticket_id": str(ticket_id),
        "page_id":   page["id"],
        "url":       page.get("url", ""),
    }

    # Immediately add the new ticket to the dedup vector cache so that
    # subsequent dedup calls can detect it without waiting for a full re-sync.
    try:
        from backend.rag.ticket_dedup import insert_ticket as _insert_ticket
        await _insert_ticket({
            "ticket_id": str(ticket_id),
            "page_id":   page["id"],
            "question":  req.question,
            "url":       page.get("url", ""),
        })
    except Exception as _e:
        logger.warning("Failed to cache dedup embedding for ticket %s: %s", ticket_id, _e)

    return result


@router.get("/tickets")
async def get_tickets():
    """
    Fetch all support tickets from Notion with a 60-second Redis cache.

    Returns an empty list (not an error) when `NOTION_TICKET_DB_ID` is
    not configured.
    """
    cached = await cache.get(TICKETS_CACHE_KEY)
    if cached is not None:
        return {"tickets": cached, "source": "cache"}

    ticket_db_id = getattr(settings, "NOTION_TICKET_DB_ID", None)
    if not ticket_db_id:
        return {
            "tickets": [], "source": "not_configured",
            "hint": "Add NOTION_TICKET_DB_ID to .env to enable ticket tracking",
        }

    try:
        headers = _notion_headers()
        results = []
        cursor  = None

        async with _httpx.AsyncClient(timeout=30) as client:
            while True:
                body: dict = {"page_size": 100}
                if cursor:
                    body["start_cursor"] = cursor

                resp = await client.post(
                    f"{NOTION_API}/databases/{ticket_db_id}/query",
                    headers=headers, json=body,
                )

                if resp.status_code == 404:
                    return {
                        "tickets": [], "source": "error",
                        "hint": (
                            f"Notion returned 404 for DB {ticket_db_id}. "
                            "Open the database in Notion → Share → Invite your integration."
                        ),
                    }

                resp.raise_for_status()
                data = resp.json()
                results.extend(data.get("results", []))

                if not data.get("has_more"):
                    break
                cursor = data.get("next_cursor")
                if not cursor:
                    break

        results.sort(key=lambda p: p.get("created_time", ""), reverse=True)
        tickets = [_page_to_ticket(p) for p in results]

        await cache.set(TICKETS_CACHE_KEY, tickets, ttl=60)
        logger.info("Fetched %d tickets from Notion", len(tickets))
        return {"tickets": tickets, "source": "notion"}

    except Exception as e:
        logger.error("get_tickets error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tickets/update")
async def update_ticket(req: TicketUpdateRequest):
    """
    Update the status of an existing support ticket.

    Looks up the Notion page_id from the Redis cache first, then falls
    back to a direct Notion database query by Ticket ID. Invalidates
    the ticket cache after a successful update.

    Valid status values: `Open`, `In Progress`, `Resolved`.
    """
    VALID = {"Open", "In Progress", "Resolved"}
    if req.status not in VALID:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{req.status}'. Must be one of: {VALID}",
        )

    try:
        cached  = await cache.get(TICKETS_CACHE_KEY) or []
        page_id = next(
            (t["page_id"] for t in cached if t.get("ticket_id") == req.ticket_id),
            None,
        )

        async with _httpx.AsyncClient(timeout=30) as client:
            if not page_id:
                db_id = _get_ticket_db_id()
                body  = {
                    "page_size": 10,
                    "filter": {
                        "property": "Ticket ID",
                        "rich_text": {"equals": req.ticket_id},
                    },
                }
                qresp = await client.post(
                    f"{NOTION_API}/databases/{db_id}/query",
                    headers=_notion_headers(), json=body,
                )
                qresp.raise_for_status()
                pages = qresp.json().get("results", [])
                if not pages:
                    raise HTTPException(status_code=404, detail=f"Ticket {req.ticket_id} not found")
                page_id = pages[0]["id"]

            upd = await client.patch(
                f"{NOTION_API}/pages/{page_id}",
                headers=_notion_headers(),
                json={"properties": {"Status": {"select": {"name": req.status}}}},
            )
            upd.raise_for_status()

        await cache.flush_pattern(f"{TICKETS_CACHE_KEY}*")
        logger.info("Ticket %s → %s", req.ticket_id, req.status)
        return {"success": True, "ticket_id": req.ticket_id, "new_status": req.status}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("update_ticket error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory")
async def save_memory(req: MemorySaveRequest):
    """Save or merge agent session context into Redis (TTL: 24 hours)."""
    try:
        mem_key  = f"docforge:agent:memory:{req.session_id}"
        existing = await cache.get(mem_key) or {}
        existing.update(req.memory)
        await cache.set(mem_key, existing, ttl=86_400)
        return {"success": True, "session_id": req.session_id, "memory": existing}
    except Exception as e:
        logger.error("save_memory error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ticket/create")
async def create_ticket(req: TicketCreateRequest):
    """HTTP endpoint that delegates ticket creation to `_create_notion_ticket()`."""
    try:
        return await _create_notion_ticket(req)
    except Exception as e:
        logger.error("create_ticket error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/dedup/flush")
async def flush_dedup_cache():
    """
    Flush the ticket dedup vector cache from Redis.
    The next call to find_duplicate() will re-fetch all tickets and re-embed them.
    """
    try:
        from backend.rag.ticket_dedup import flush_dedup_cache as _flush
        await _flush()
        return {"flushed": True, "note": "Ticket vector cache cleared. Will re-index on next dedup call."}
    except Exception as e:
        logger.error("flush_dedup_cache error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))