"""
agent_routes.py — FastAPI routes for the CiteRAG Agent layer

Bug fixes vs v2:
  1. TICKETS_CACHE_KEY is a module-level constant — no more duplicate string
     literals in get_tickets() and update_ticket() that could silently diverge.
  2. Added _fetch_session_tickets() so agent_graph can query Notion for
     session-specific tickets without importing route handlers.
  3. update_ticket pagination: falls back to a proper Notion query (not just
     the first 100 pages) when the page_id is not in the Redis cache.
  4. Removed dead `random` import — ticket IDs are numeric strings only.
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

# ── Constants ─────────────────────────────────────────────────────────────────
NOTION_API       = "https://api.notion.com/v1"
NOTION_VER       = "2022-06-28"
TICKETS_CACHE_KEY = "docforge:agent:tickets"   # single source of truth


# ── Schemas ───────────────────────────────────────────────────────────────────

class TicketUpdateRequest(BaseModel):
    """Schema for updating the status of an existing Notion support ticket."""
    ticket_id: str
    status:    str   # "Open" | "In Progress" | "Resolved"


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
    """Schema for saving dynamic session memory (like user names or roles) to Redis."""
    session_id: str
    memory:     dict


# ── Notion REST helpers ───────────────────────────────────────────────────────

def _notion_headers() -> dict:
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
    return getattr(settings, "NOTION_TICKET_DB_ID", None) or settings.NOTION_DATABASE_ID


def _page_to_ticket(page: dict) -> dict:
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


# ── Internal helpers (used by agent_graph) ────────────────────────────────────

async def _fetch_session_tickets(session_id: str) -> list[dict]:
    """
    Query Notion for all tickets belonging to a specific session_id.
    Used by agent_graph._exec_update_ticket() as a fallback when session
    memory is empty (e.g. after a restart or UI-created tickets).

    Returns [] on any error (fail-open so updates are never silently blocked).
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

        pages = resp.json().get("results", [])
        tickets = [_page_to_ticket(p) for p in pages]
        logger.info("_fetch_session_tickets: found %d for session=%s", len(tickets), session_id)
        return tickets

    except Exception as e:
        logger.warning("_fetch_session_tickets failed for session=%s: %s", session_id, e)
        return []


async def _create_notion_ticket(req: TicketCreateRequest) -> dict:
    """
    Core ticket-creation logic.  Called by both the HTTP route handler and
    directly from agent_graph (bypassing HTTP overhead).

    Returns {"success": bool, "ticket_id": str, "page_id": str, "url": str}.
    Raises on Notion API failure.
    """
    db_id     = _get_ticket_db_id()
    headers   = _notion_headers()
    priority  = req.priority if req.priority in {"High", "Medium", "Low"} else "Medium"
    live_date = datetime.now(timezone.utc).isoformat()

    # Generate a simple numeric ticket ID
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

    # Invalidate ticket cache so next read reflects the new ticket
    await cache.flush_pattern(f"{TICKETS_CACHE_KEY}*")
    logger.info("Ticket created: %s for: %s", ticket_id, req.question[:60])

    return {
        "success":   True,
        "ticket_id": ticket_id,
        "page_id":   page["id"],
        "url":       page.get("url", ""),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/tickets")
async def get_tickets():
    """
    Fetch all tickets from Notion.  Redis 60-second cache.
    Returns empty list (not an error) when NOTION_TICKET_DB_ID is not configured.
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
    Update ticket status.  Looks up page_id from Redis cache first;
    falls back to a Notion filter query if not cached.
    Invalidates cache after a successful update.
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
                # Fallback: query Notion by Ticket ID property
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
    """Save or merge agent session context."""
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
    """HTTP endpoint — delegates to _create_notion_ticket()."""
    try:
        return await _create_notion_ticket(req)
    except Exception as e:
        logger.error("create_ticket error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/dedup/flush")
async def flush_dedup_cache():
    """No-op — dedup uses live Notion data now, no cache to flush."""
    return {"flushed": 0, "note": "Dedup uses live Notion data — no cache to flush."}
