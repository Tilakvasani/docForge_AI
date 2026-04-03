"""
logger.py — Pretty PowerShell-friendly logging for CiteRAG / DocForge

Output format:
  14:32:05  ℹ  Redis connected
  14:32:05  ✅ Agent compiled — nodes: [...]
  14:32:08  ⚠️  Classifier LLM failed — defaulting to DOCUMENT
  14:32:09  ❌ create_ticket error: ...
"""

import logging
import sys
from backend.core.config import settings


# ── Emoji map per level ───────────────────────────────────────────────────────
_LEVEL_EMOJI = {
    logging.DEBUG:    "🔎",
    logging.INFO:     "ℹ️ ",
    logging.WARNING:  "⚠️ ",
    logging.ERROR:    "❌",
    logging.CRITICAL: "🔥",
}

# Shorten noisy module names for cleaner output
_MODULE_ALIASES = {
    "backend.agents.agent_graph":         "agent",
    "backend.api.agent_routes":           "routes",
    "backend.rag.rag_service":            "rag",
    "backend.api.rag_routes":             "rag_api",
    "backend.rag.ticket_dedup":           "dedup",
    "backend.rag.ragas_scorer":           "ragas",
    "backend.rag.ingest_service":         "ingest",
    "backend.services.redis_service":     "redis",
    "backend.core.logger":                "core",
    "ai-doc-generator":                   "app",
    "uvicorn":                            "uvicorn",
    "uvicorn.error":                      "uvicorn",
    "uvicorn.access":                     "access",
    "fastapi":                            "fastapi",
}


class _PrettyFormatter(logging.Formatter):
    """
    Compact, emoji-prefixed formatter.
    Format:  HH:MM:SS  EMOJI  [module]  message
    """

    def format(self, record: logging.LogRecord) -> str:
        # Time — only HH:MM:SS, no date
        time_str = self.formatTime(record, "%H:%M:%S")

        # Emoji for level
        emoji = _LEVEL_EMOJI.get(record.levelno, "  ")

        # Short module name
        module = _MODULE_ALIASES.get(record.name, record.name.split(".")[-1])

        # Message
        msg = record.getMessage()

        # Attach exception if present
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            # indent exception lines for readability
            exc_lines = "\n".join("      " + l for l in exc_text.splitlines())
            msg = f"{msg}\n{exc_lines}"

        return f"  {time_str}  {emoji}  [{module}]  {msg}"


def _setup_logging():
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_PrettyFormatter())

    root = logging.getLogger()
    root.setLevel(level)

    # FIX: use force=True instead of root.handlers.clear()
    # clear() was crashing uvicorn SpawnProcess on Windows (Python 3.11)
    logging.basicConfig(handlers=[handler], level=level, force=True)

    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "langsmith"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if level > logging.DEBUG:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# _setup_logging() is now called by main.py lifespan

logger = logging.getLogger("app")