"""
agent_routes.py — FastAPI routes for the CiteRAG Agent layer
==============================================================

Endpoints:
  GET  /api/agent/tickets           — Fetch all knowledge-gap tickets from Notion DB
  POST /api/agent/tickets/update    — Update ticket status in Notion
  GET  /api/agent/memory            — Read agent memory for current session
  POST /api/agent/ticket/create     — (internal) Create a new ticket from low-confidence RAG answers
  DELETE /api/agent/dedup/flush     — Clear dedup cache (admin)
"""

# ── Standard library ──────────────────────────────────────────────────────────
from datetime import datetime, timezone
from typing import Optional, List

# ── Third-party ───────────────────────────────────────────────────────────────
import httpx as _httpx
from fastapi import APIRouter, HTTPException  # FastAPI routing + error responses
from pydantic import BaseModel                 # Request/response schema validation

# ── Internal ──────────────────────────────────────────────────────────────────
from backend.core.logger import logger         # Structured logger
from backend.services.redis_service import cache  # Redis client for caching tickets

router = APIRouter(prefix="/agent", tags=["Agent"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class TicketUpdateRequest(BaseModel):
    ticket_id: str
    status:    str   # "Open" | "In Progress" | "Resolved"


class TicketCreateRequest(BaseModel):
    question:          str
    session_id:        str = "default"
    attempted_sources: List[str] = []
    summary:           str = ""
    priority:          str = "Medium"  # "High" | "Medium" | "Low"
    confidence:        str = "low"
    user_info:         str = "Admin"



# ── Notion REST helpers ───────────────────────────────────────────────────────

NOTION_API  = "https://api.notion.com/v1"
NOTION_VER  = "2022-06-28"


def _notion_headers() -> dict:
    """
    Build Notion API request headers using the token from .env.
    Supports both NOTION_TOKEN and legacy NOTION_API_KEY env variable names.
    Raises ValueError if no token is found.
    """
    from backend.core.config import settings  # lazy: avoids circular import on startup
    token = (
        getattr(settings, "NOTION_TOKEN", "")
        or getattr(settings, "NOTION_API_KEY", "")
    )
    if not token:
        raise ValueError(
            "Notion token not set. Add NOTION_TOKEN (or NOTION_API_KEY) to your .env file."
        )
    return {
        "Authorization":  f"Bearer {token}",
        "Content-Type":   "application/json",
        "Notion-Version": NOTION_VER,
    }


def _get_ticket_db_id() -> str:
    """Return the Notion database ID for support tickets (NOTION_TICKET_DB_ID), falling back to NOTION_DATABASE_ID."""
    from backend.core.config import settings  # lazy: avoids circular import on startup
    return getattr(settings, "NOTION_TICKET_DB_ID", None) or settings.NOTION_DATABASE_ID


def _page_to_ticket(page: dict) -> dict:
    """Map a raw Notion page object to a flat ticket dict for the frontend."""
    props = page.get("properties", {})

    def _text(prop_name):
        prop = props.get(prop_name, {})
        ptype = prop.get("type", "")
        if ptype == "title":
            items = prop.get("title", [])
        elif ptype == "rich_text":
            items = prop.get("rich_text", [])
        else:
            return ""
        return "".join(t.get("plain_text", "") for t in items)

    def _select(prop_name):
        prop = props.get(prop_name, {})
        sel = prop.get("select") or {}
        return sel.get("name", "")

    def _multi_select(prop_name):
        prop = props.get(prop_name, {})
        return [s.get("name", "") for s in prop.get("multi_select", [])]

    def _date(prop_name):
        prop = props.get(prop_name, {})
        date = prop.get("date") or {}
        return date.get("start", "")

    ticket_id = page.get("id", "").replace("-", "")[:8].upper()

    return {
        "ticket_id":         ticket_id,
        "page_id":           page.get("id", ""),
        "question":          _text("Question") or _text("Name") or _text("Title"),
        "status":            _select("Status") or "Open",
        "priority":          _select("Priority") or "Medium",
        "summary":           _text("Summary"),
        "attempted_sources": _multi_select("Attempted Sources"),
        "created_at":        _date("Created") or page.get("created_time", "")[:10],
        "url":               page.get("url", ""),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/tickets")
async def get_tickets():
    """
    Fetch all knowledge-gap tickets from Notion.
    Returns empty list (not an error) when NOTION_TICKET_DB_ID isn't configured yet.
    """
    CACHE_KEY = "docforge:agent:tickets"
    cached = await cache.get(CACHE_KEY)
    if cached is not None:
        return {"tickets": cached, "source": "cache"}

    # If no dedicated ticket DB is configured, return empty gracefully
    from backend.core.config import settings
    ticket_db_id = getattr(settings, "NOTION_TICKET_DB_ID", None)
    if not ticket_db_id:
        logger.info("NOTION_TICKET_DB_ID not set — returning empty ticket list")
        return {"tickets": [], "source": "not_configured",
                "hint": "Add NOTION_TICKET_DB_ID to .env to enable ticket tracking"}

    try:
        import httpx as _httpx
        headers = _notion_headers()
        body    = {"page_size": 100}

        results  = []
        cursor   = None
        async with _httpx.AsyncClient(timeout=30) as client:
            while True:
                if cursor:
                    body["start_cursor"] = cursor
                resp = await client.post(
                    f"{NOTION_API}/databases/{ticket_db_id}/query",
                    headers=headers, json=body
                )
                resp_data = resp.json()
                if resp.status_code == 404:
                    logger.warning(
                        "Notion DB %s not found (404) — integration may lack access", ticket_db_id
                    )
                    return {
                        "tickets": [],
                        "source":  "error",
                        "hint":    (
                            f"Notion returned 404 for DB {ticket_db_id}. "
                            "Fix: open the database in Notion → Share → Invite your integration."
                        ),
                    }
                resp.raise_for_status()
                data = resp_data
                results.extend(data.get("results", []))
                if not data.get("has_more"):
                    break
                cursor = data.get("next_cursor")
                body.pop("start_cursor", None)

        # Sort by created_time descending (client-side, always safe)
        results.sort(key=lambda p: p.get("created_time", ""), reverse=True)

        tickets = [_page_to_ticket(p) for p in results]
        await cache.set(CACHE_KEY, tickets, ttl=60)
        logger.info("Fetched %d tickets from Notion DB", len(tickets))
        return {"tickets": tickets, "source": "notion"}

    except Exception as e:
        logger.error("get_tickets error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tickets/update")
async def update_ticket(req: TicketUpdateRequest):
    """
    Update the Notion status of a ticket by its ticket_id.
    Looks up the page_id from Redis cache first; falls back to querying Notion directly.
    Invalidates the ticket cache after a successful update.
    """
    VALID_STATUSES = {"Open", "In Progress", "Resolved"}
    if req.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{req.status}'. Must be one of: {VALID_STATUSES}"
        )

    try:
        import httpx as _httpx
        CACHE_KEY = "docforge:agent:tickets"
        tickets   = await cache.get(CACHE_KEY) or []

        # Find full page_id from cache first
        page_id = None
        for t in tickets:
            if t.get("ticket_id") == req.ticket_id:
                page_id = t.get("page_id")
                break

        async with _httpx.AsyncClient(timeout=30) as client:
            # If not in cache, re-query Notion
            if not page_id:
                db_id = _get_ticket_db_id()
                resp  = await client.post(
                    f"{NOTION_API}/databases/{db_id}/query",
                    headers=_notion_headers(), json={"page_size": 100}
                )
                resp.raise_for_status()
                for p in resp.json().get("results", []):
                    if p["id"].replace("-", "")[:8].upper() == req.ticket_id:
                        page_id = p["id"]
                        break

            if not page_id:
                raise HTTPException(status_code=404, detail=f"Ticket {req.ticket_id} not found")

            upd = await client.patch(
                f"{NOTION_API}/pages/{page_id}",
                headers=_notion_headers(),
                json={"properties": {"Status": {"select": {"name": req.status}}}},
            )
            upd.raise_for_status()

        await cache.flush_pattern("docforge:agent:tickets*")
        logger.info("Ticket %s → %s", req.ticket_id, req.status)
        return {"success": True, "ticket_id": req.ticket_id, "new_status": req.status}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("update_ticket error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory")
async def get_memory(session_id: str = "default"):
    """
    Return the agent memory for the current session.
    Reads from Redis — same keys used by agent_graph.py.
    """
    try:
        mem_key   = f"docforge:agent:memory:{session_id}"
        agent_mem = await cache.get(mem_key) or {}
        return {"session_id": session_id, "memory": agent_mem}

    except Exception as e:
        logger.error("get_memory error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _create_notion_ticket(req: "TicketCreateRequest") -> dict:
    """
    Plain async function — core Notion ticket creation logic.
    Called directly by the LangGraph agent (Bug #8 fix) and also by
    the HTTP route handler below.

    Returns: {"success": bool, "ticket_id": str, "page_id": str, "url": str}
    Raises on Notion API failure.
    """
    db_id     = _get_ticket_db_id()
    headers   = _notion_headers()
    priority  = req.priority if req.priority in {"High", "Medium", "Low"} else "Medium"
    live_date = datetime.now(timezone.utc).isoformat()

    # ── Property names must EXACTLY match the Notion DB columns ──────────────
    properties: dict = {
        "Question": {
            "title": [{"type": "text", "text": {"content": req.question[:2000]}}]
        },
        "Status":   {"select": {"name": "Open"}},
        "Priority": {"select": {"name": priority}},
        "User Info": {
            "rich_text": [{"type": "text", "text": {"content": req.user_info or "Admin"}}]
        },
        "Created": {
            "date": {"start": live_date[:10]}
        },
        "Assigned Owner": {
            "rich_text": [{"type": "text", "text": {"content": "Support Team"}}]
        },
    }
    if req.summary:
        properties["Summary"] = {
            "rich_text": [{"type": "text", "text": {"content": req.summary[:2000]}}]
        }
    if req.session_id:
        properties["Session ID"] = {
            "rich_text": [{"type": "text", "text": {"content": req.session_id}}]
        }
    if req.attempted_sources:
        properties["Attempted Sources"] = {
            "multi_select": [{"name": s[:100]} for s in req.attempted_sources[:10]]
        }

    async with _httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{NOTION_API}/pages",
            headers=headers,
            json={"parent": {"database_id": db_id}, "properties": properties},
        )
    resp.raise_for_status()
    page = resp.json()

    ticket_id = page["id"].replace("-", "")[:8].upper()
    await cache.flush_pattern("docforge:agent:tickets*")
    logger.info("Created ticket %s for: %s", ticket_id, req.question[:60])
    return {
        "success":   True,
        "ticket_id": ticket_id,
        "page_id":   page["id"],
        "url":       page.get("url", ""),
    }


@router.post("/ticket/create")
async def create_ticket(req: TicketCreateRequest):
    """
    HTTP endpoint — delegates to _create_notion_ticket().
    The LangGraph agent now calls _create_notion_ticket() directly
    instead of importing this route handler (Bug #8 fix).
    """
    try:
        return await _create_notion_ticket(req)
    except Exception as e:
        logger.error("create_ticket error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ── DELETE /api/agent/dedup/flush (admin) ────────────────────────────────────

@router.delete("/dedup/flush")
async def flush_dedup():
    """Clear the duplicate-detection cache (exact hashes + embeddings)."""
    try:
        from backend.services.rag.ticket_dedup import flush_dedup_cache
        await flush_dedup_cache()
        return {"success": True, "message": "Dedup cache flushed."}
    except Exception as e:
        logger.error("flush_dedup error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))