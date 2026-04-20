"""
Structured console logging for DocForge AI.

Produces compact, emoji-prefixed log lines readable in any terminal:

    14:32:05  ℹ️   [app]    DocForge AI backend started
    14:32:08  ⚠️   [agent]  Classifier LLM failed — defaulting to DOCUMENT
    14:32:09  ❌  [routes]  Notion API 429: rate limit exceeded

The logger is initialized lazily via `_setup_logging()`, which is called
once during the FastAPI lifespan startup to avoid conflicts with uvicorn's
own logging configuration on hot-reload.
"""

import logging
import sys
from backend.core.config import settings


_LEVEL_EMOJI = {
    logging.DEBUG:    "🔎",
    logging.INFO:     "ℹ️ ",
    logging.WARNING:  "⚠️ ",
    logging.ERROR:    "❌",
    logging.CRITICAL: "🔥",
}

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
    Compact, emoji-prefixed log formatter for console output.

    Produces lines in the format:
        [HH:MM:SS]  [EMOJI]  [module]  message

    Long module names are replaced with short aliases defined in
    `_MODULE_ALIASES`. Exception tracebacks are indented for readability.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a `LogRecord` into a single, human-readable console line.

        Args:
            record: The log record to format.

        Returns:
            A formatted string including timestamp, emoji, module alias,
            message, and any attached exception traceback.
        """
        time_str = self.formatTime(record, "%H:%M:%S")
        emoji    = _LEVEL_EMOJI.get(record.levelno, "  ")
        module   = _MODULE_ALIASES.get(record.name, record.name.split(".")[-1])
        msg      = record.getMessage()

        if record.exc_info:
            exc_text  = self.formatException(record.exc_info)
            exc_lines = "\n".join("      " + l for l in exc_text.splitlines())
            msg = f"{msg}\n{exc_lines}"

        return f"  {time_str}  {emoji}  [{module}]  {msg}"


def _setup_logging():
    """
    Initialize the global logging configuration for DocForge AI.

    Attaches `_PrettyFormatter` to the root logger's stdout `StreamHandler`
    and silences noisy third-party libraries. Safe to call multiple times —
    if a stdout `StreamHandler` already exists (e.g. installed by uvicorn),
    the formatter is updated in place rather than adding a duplicate handler.
    This prevents double-printing on hot-reload and avoids conflicts with
    uvicorn's internal `dictConfig` setup on Windows.
    """
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    already_has_stdout = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        for h in root.handlers
    )
    if not already_has_stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_PrettyFormatter())
        handler.setLevel(level)
        root.addHandler(handler)
    else:
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout:
                h.setFormatter(_PrettyFormatter())
                h.setLevel(level)

    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "langsmith"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if level > logging.DEBUG:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


logger = logging.getLogger("app")