"""
agent_graph.py — Agentic CiteRAG using LangGraph StateGraph
=============================================================

Graph topology (3-intent router):

  START
    │
    ▼
  classify_intent
    │
    ├── "create_ticket" ──► handle_create_ticket ──────────────────────────► END
    │
    ├── "update_ticket" ──► handle_update_ticket ──────────────────────────► END
    │
    └── "rag"           ──► classify_confidence ──► (needs_ticket) ──► summarize ──► create_ticket ──┐
                                                └── (skip_ticket) ──────────────────────────────────┴─► update_memory ──► END

Usage (from rag_routes.py):
    from backend.services.rag.agent_graph import run_agent
    result = await run_agent(question, rag_result, session_id)
    # result["intent"] tells you which path was taken
    # result["agent_reply"] is set for create_ticket / update_ticket paths
"""

import logging
import re
from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from backend.services.redis_service import cache

logger = logging.getLogger(__name__)

MEMORY_TTL = 86400
MEMORY_KEY = "docforge:agent:memory:{session_id}"


# ── Typed state ─────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    question:        str
    session_id:      str
    rag_result:      dict     # empty dict when intent != "rag"

    # Intent routing
    intent:          str      # "rag" | "create_ticket" | "update_ticket"
    new_status:      str      # extracted status for update_ticket intent

    # RAG sub-path
    confidence:      str
    chunks:          list
    should_ticket:   bool
    ticket_summary:  str
    ticket_id:       Optional[str]

    # Final reply for non-RAG paths
    agent_reply:     str

    # Memory
    memory:          dict


# ── Memory helpers ───────────────────────────────────────────────────────────────

async def _load_memory(session_id: str) -> dict:
    return await cache.get(MEMORY_KEY.format(session_id=session_id)) or {}


async def _save_memory(session_id: str, memory: dict):
    await cache.set(MEMORY_KEY.format(session_id=session_id), memory, ttl=MEMORY_TTL)


# ── Intent classification (keyword-based, no extra LLM call) ────────────────────

def _classify_intent(question: str) -> str:
    """We removed manual ticket creation. All questions route to RAG."""
    return "rag"



# ── Nodes ────────────────────────────────────────────────────────────────────────

async def node_classify_intent(state: AgentState) -> AgentState:
    """Route the user message: RAG or create ticket."""
    intent = _classify_intent(state["question"])
    logger.info("Intent classified: %s (session=%s)", intent, state["session_id"])
    return {**state, "intent": intent, "new_status": ""}


async def node_handle_create_ticket(state: AgentState) -> AgentState:
    """
    User said 'create ticket'.
    Check Redis memory: was the last RAG answer low-confidence?
      Yes → create ticket in Notion → reply with link
      No  → politely decline
    """
    memory   = await _load_memory(state["session_id"])
    pending  = memory.get("ticket_pending", False)
    last_q   = memory.get("last_question", "")
    last_tid = memory.get("last_ticket_id")

    # Don't create duplicate if one already exists for this turn
    if last_tid and not memory.get("ticket_consumed", False):
        reply = (
            f"🎫 I have already notified the team about this missing information."
        )
        return {**state, "agent_reply": reply}

    if not pending:
        reply = (
            "✅ Your last question was answered well from the documents. "
            "No knowledge gap found — no ticket needed!\n\n"
            "_If you feel the answer was insufficient, ask your question again "
            "and I'll reassess._"
        )
        return {**state, "agent_reply": reply}

    # Ticket is warranted — call create_ticket
    try:
        from backend.services.rag.agent_routes import create_ticket, TicketCreateRequest
        req    = TicketCreateRequest(
            question=last_q or state["question"],
            session_id=state["session_id"],
            attempted_sources=memory.get("last_attempted_sources", []),
            summary=(
                f"User manually requested ticket. "
                f"Original question: \"{(last_q or state['question'])[:200]}\". "
                f"RAG confidence: {memory.get('last_confidence', 'low')}."
            ),
            priority="High" if not memory.get("last_chunks") else "Medium",
            confidence=memory.get("last_confidence", "low"),
        )
        result = await create_ticket(req)
        tid    = result.get("ticket_id", "")
        url    = result.get("url", "")

        # Mark ticket as consumed so we don't duplicate
        memory["last_ticket_id"]      = tid
        memory["last_ticket_url"]     = url
        memory["ticket_pending"]      = False
        memory["ticket_consumed"]     = True
        await _save_memory(state["session_id"], memory)

        reply = (
            f"✅ **Got it!** I've logged this as a knowledge gap and notified the team.\n\n"
            f"_If you want to check or change its status later, just say "
            f"'mark as resolved' or 'set to in progress'._"
        )
    except Exception as exc:
        logger.error("handle_create_ticket failed: %s", exc, exc_info=True)
        reply = "⚠️ Ticket creation failed. Please check the backend logs or Notion permissions."

    return {**state, "agent_reply": reply}


async def node_handle_update_ticket(state: AgentState) -> AgentState:
    """
    User said 'mark as resolved' / 'set to in progress' etc.
    Loads the last ticket_id from Redis and PATCHes Notion.
    """
    memory    = await _load_memory(state["session_id"])
    ticket_id = memory.get("last_ticket_id")
    new_status = state["new_status"]

    if not ticket_id:
        reply = (
            "⚠️ I don't have a ticket on record for this session yet. "
            "Ask a question first, then say 'create ticket' if it couldn't be answered."
        )
        return {**state, "agent_reply": reply}

    try:
        import httpx as _httpx
        from backend.services.rag.agent_routes import _notion_headers, NOTION_API

        headers = _notion_headers()
        body    = {
            "properties": {
                "Status": {"select": {"name": new_status}}
            }
        }
        resp = _httpx.patch(
            f"{NOTION_API}/pages/{ticket_id}",
            headers=headers, json=body, timeout=15,
        )
        resp.raise_for_status()

        # Persist updated status
        memory["last_ticket_status"] = new_status
        await _save_memory(state["session_id"], memory)

        reply = (
            f"✅ **Status updated successfully!**\n\n"
            f"The issue has now been marked as **{new_status}**."
        )
    except Exception as exc:
        logger.error("handle_update_ticket failed: %s", exc, exc_info=True)
        reply = f"⚠️ Could not update ticket status. Error: {exc}"

    return {**state, "agent_reply": reply}


# ── RAG sub-path nodes ───────────────────────────────────────────────────────────

async def node_classify_confidence(state: AgentState) -> AgentState:
    rag    = state["rag_result"]
    conf   = rag.get("confidence", "high")
    chunks = rag.get("chunks", [])
    answer = rag.get("answer", "")
    tool   = rag.get("tool_used", "")

    if tool == "chat":
        should = False
    else:
        total_miss     = (conf == "low") and not chunks
        said_not_found = (conf == "low") and ("could not find" in answer.lower())
        should = total_miss or said_not_found

    return {
        **state,
        "should_ticket": should,
        "confidence":    conf,
        "chunks":        chunks,
        "rag_result": {
            **rag,
            "ticket_pending": should and not rag.get("ticket_id"),
        },
    }


async def node_summarize(state: AgentState) -> AgentState:
    rag       = state["rag_result"]
    citations = rag.get("citations", [])
    chunks    = rag.get("chunks", [])
    question  = state["question"]

    attempted: list[str] = []
    for c in (citations or chunks)[:5]:
        if isinstance(c, dict):
            # Citations use 'text' for the display title, chunks use 'doc_title'
            raw_title = c.get("doc_title") if "doc_title" in c else c.get("text", "")
        else:
            raw_title = str(c)
            
        # Clean for Notion multi-select: max 100 chars, no commas
        clean_title = raw_title.replace(",", "").strip()[:100]
        if clean_title and clean_title not in attempted:
            attempted.append(clean_title)

    summary = (
        f"User asked: \"{question[:200]}\". "
        f"RAG returned {len(chunks)} chunk(s), confidence={state['confidence']}. "
    )
    summary += (
        f"Attempted sources: {', '.join(attempted[:5])}."
        if attempted else
        "No relevant documents found in ChromaDB."
    )
    rag_upd = {**rag, "attempted_sources": attempted}
    return {**state, "ticket_summary": summary, "rag_result": rag_upd}


async def node_auto_create_ticket(state: AgentState) -> AgentState:
    """Auto-create Notion ticket when RAG confidence is too low."""
    try:
        from backend.services.rag.agent_routes import create_ticket, TicketCreateRequest
        
        # Prevent exact duplicate tickets if user spams the same question
        memory = await _load_memory(state["session_id"])
        last_q = memory.get("last_question", "").strip().lower()
        curr_q = state["question"].strip().lower()
        
        if last_q == curr_q and memory.get("last_ticket_id"):
            logger.info("Skipping duplicate auto-ticket for: %s", curr_q)
            ans = state["rag_result"].get("answer", "")
            ans += "\n\n*(Note: Our team is already aware of this missing information and is working on it!)*"
            
            rag_upd = {
                **state["rag_result"],
                "answer": ans,
                "ticket_pending": False,
            }
            return {**state, "rag_result": rag_upd}

        req    = TicketCreateRequest(
            question=state["question"],
            session_id=state["session_id"],
            attempted_sources=state["rag_result"].get("attempted_sources", []),
            summary=state["ticket_summary"],
            priority="High" if not state["chunks"] else "Medium",
            confidence=state["confidence"],
        )
        result = await create_ticket(req)
        tid    = result.get("ticket_id")
        url    = result.get("url")
        logger.info("Auto-ticket created: %s", tid)
        
        ans = state["rag_result"].get("answer", "")
        ans += "\n\n*(Note: A knowledge gap ticket has been automatically created for our team!)*"
        
        rag_upd = {
            **state["rag_result"],
            "answer":         ans,
            "ticket_id":      tid,
            "ticket_url":     url,
            "ticket_pending": False,
        }
        return {**state, "ticket_id": tid, "rag_result": rag_upd}
    except Exception as exc:
        logger.error("node_auto_create_ticket failed: %s", exc, exc_info=True)
        return state


async def node_update_memory(state: AgentState) -> AgentState:
    """Persist session memory to Redis after every turn."""
    try:
        memory = await _load_memory(state["session_id"])
        rag    = state["rag_result"]

        tool_to_intent = {
            "compare":  "compare_docs",
            "analysis": "analyse_docs",
            "refine":   "summarise_doc",
            "full_doc": "full_document",
            "search":   "search",
            "chat":     "greeting",
        }
        tool = rag.get("tool_used", "")
        memory["last_intent"]           = tool_to_intent.get(tool, tool)
        memory["turn_count"]            = memory.get("turn_count", 0) + 1
        memory["last_question"]         = state["question"]
        memory["last_confidence"]       = state["confidence"]
        memory["last_chunks"]           = bool(state["chunks"])
        memory["ticket_pending"]        = rag.get("ticket_pending", False)
        memory["last_attempted_sources"] = rag.get("attempted_sources", [])

        # Track auto-created ticket
        if rag.get("ticket_id"):
            memory["last_ticket_id"]   = rag["ticket_id"]
            memory["last_ticket_url"]  = rag.get("ticket_url", "")
            memory["ticket_consumed"]  = False

        citations = rag.get("citations", [])
        if citations:
            first = citations[0]
            memory["last_doc"] = first.get("text", "") if isinstance(first, dict) else str(first)

        await _save_memory(state["session_id"], memory)
        return {**state, "memory": memory}
    except Exception as exc:
        logger.warning("Memory update failed: %s", exc)
        return state


# ── Conditional edges ────────────────────────────────────────────────────────────

def route_intent(state: AgentState) -> str:
    return state["intent"]   # "rag" | "create_ticket" | "update_ticket"


def route_confidence(state: AgentState) -> str:
    return "needs_ticket" if state["should_ticket"] else "skip_ticket"


# ── Build and compile the graph ──────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(AgentState)

    # Register all nodes
    g.add_node("classify_intent",      node_classify_intent)
    g.add_node("classify_confidence",  node_classify_confidence)
    g.add_node("summarize",            node_summarize)
    g.add_node("auto_create_ticket",   node_auto_create_ticket)
    g.add_node("update_memory",        node_update_memory)

    # Entry
    g.set_entry_point("classify_intent")

    # Top-level intent routing
    g.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "rag": "classify_confidence",
        },
    )

    # RAG path: confidence routing
    g.add_conditional_edges(
        "classify_confidence",
        route_confidence,
        {
            "needs_ticket": "summarize",
            "skip_ticket":  "update_memory",
        },
    )
    g.add_edge("summarize",          "auto_create_ticket")
    g.add_edge("auto_create_ticket", "update_memory")
    g.add_edge("update_memory",      END)

    return g.compile()


_graph = _build_graph()
logger.info("LangGraph agent compiled — nodes: %s", list(_graph.nodes.keys()))


# ── Public interface ─────────────────────────────────────────────────────────────

async def run_agent(
    question:   str,
    rag_result: dict,
    session_id: str = "default",
) -> dict:
    """
    Run the agentic LangGraph graph over a user message.

    For RAG questions: enriches rag_result with ticket_pending / ticket_id / ticket_url.
    For ticket commands: returns rag_result with agent_reply set and intent set.

    The caller (rag_routes.py) checks result["intent"] to decide the response.
    """
    initial: AgentState = {
        "question":        question,
        "session_id":      session_id,
        "rag_result":      rag_result,
        "intent":          "rag",
        "new_status":      "",
        "confidence":      rag_result.get("confidence", "high"),
        "chunks":          rag_result.get("chunks", []),
        "should_ticket":   False,
        "ticket_summary":  "",
        "ticket_id":       None,
        "agent_reply":     "",
        "memory":          {},
    }

    try:
        final = await _graph.ainvoke(initial)
        result = dict(final["rag_result"])
        result["intent"]      = final["intent"]
        result["agent_reply"] = final.get("agent_reply", "")
        return result
    except Exception as exc:
        logger.error("LangGraph agent error: %s", exc, exc_info=True)
        rag_result["intent"]      = "rag"
        rag_result["agent_reply"] = ""
        return rag_result