"""
DocForge AI — redis_service.py
Redis caching layer with graceful fallback.
If Redis is unavailable, all cache ops are no-ops and the app works normally.

Cache keys & TTLs:
  departments              → 1 hour  (static DB data)
  sections:{doc_type}      → 1 hour  (static DB data)
  questions:{sec_id}       → 24 hours (expensive LLM call)
  section:{sec_id}         → 24 hours (most expensive LLM call)
  notion_library           → 5 minutes (Notion API call)

Usage:
  from backend.services.redis_service import cache
  value = await cache.get("mykey")
  await cache.set("mykey", data, ttl=3600)
  await cache.delete("mykey")
  await cache.flush_pattern("sections:*")
"""
import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis  # async Redis client

from backend.core.logger import logger

# ─── TTL constants (seconds) ──────────────────────────────────────────────────
TTL_DEPARTMENTS   = 3600          # 1 hour  — department list rarely changes
TTL_SECTIONS      = 3600          # 1 hour  — section list rarely changes
TTL_QUESTIONS     = 86400         # 24 hours — LLM-generated questions
TTL_SECTION_CONTENT = 86400       # 24 hours — LLM-generated section content
TTL_NOTION_LIBRARY  = 300         # 5 minutes — Notion API results

# ─── Cache key builders ───────────────────────────────────────────────────────
KEY_DEPARTMENTS       = "docforge:departments"
KEY_SECTIONS          = lambda doc_type: f"docforge:sections:{doc_type}"
KEY_QUESTIONS         = lambda sec_id:   f"docforge:questions:{sec_id}"
KEY_SECTION_CONTENT   = lambda sec_id:   f"docforge:section:{sec_id}"
KEY_NOTION_LIBRARY    = "docforge:notion_library"


class RedisCache:
    """
    Async Redis cache with graceful fallback and auto-reconnect.
    Call await cache.connect() on app startup.
    On timeout/connection errors, retries every 30s automatically.
    """

    _RECONNECT_COOLDOWN = 30  # seconds between reconnect attempts

    def __init__(self):
        self._redis = None
        self._available = False
        import os
        self._url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._last_fail_time = 0.0

    async def connect(self, url: Optional[str] = None) -> bool:
        """
        Try to connect to Redis. Returns True if successful.
        App continues normally if this returns False.
        """
        if url:
            self._url = url
        try:
            client = aioredis.from_url(url, encoding="utf-8",
                                       decode_responses=True,
                                       socket_connect_timeout=5,
                                       socket_timeout=5,
                                       max_connections=20,
                                       retry_on_timeout=True)
            await client.ping()
            self._redis = client
            self._available = True
            self._last_fail_time = 0.0
            logger.info("✅ Redis connected at %s", url)
            return True
        except Exception as e:
            self._available = False
            import time; self._last_fail_time = time.time()
            logger.warning("⚠️  Redis unavailable (%s) — running without cache", e)
            return False

    async def _try_reconnect(self) -> bool:
        """Auto-reconnect after cooldown period. Returns True if reconnected."""
        import time
        if time.time() - self._last_fail_time < self._RECONNECT_COOLDOWN:
            return False
        logger.info("🔄 Redis auto-reconnect attempt...")
        return await self.connect(self._url)

    async def disconnect(self):
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Core ops ───────────────────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[Any]:
        """Return deserialized value or None on miss / error."""
        if not self._available:
            if not await self._try_reconnect():
                return None
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning("Redis GET error for %s: %s", key, e)
            if "Timeout" in str(e) or "Connection" in str(e):
                import time; self._last_fail_time = time.time()
                self._available = False
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Serialize and store value with TTL. Returns False on error."""
        if not self._available:
            if not await self._try_reconnect():
                return False
        try:
            await self._redis.setex(key, ttl, json.dumps(value, default=str))
            return True
        except Exception as e:
            logger.warning("Redis SET error for %s: %s", key, e)
            if "Timeout" in str(e) or "Connection" in str(e):
                import time; self._last_fail_time = time.time()
                self._available = False
            return False

    async def delete(self, key: str) -> bool:
        """Delete a single key. Returns False if Redis is unavailable or on error."""
        if not self._available:
            if not await self._try_reconnect():
                return False
        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.warning("Redis DEL error for %s: %s", key, e)
            if "Timeout" in str(e) or "Connection" in str(e):
                import time; self._last_fail_time = time.time()
                self._available = False
            return False

    async def flush_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern using SCAN cursor (non-blocking).
        Unlike KEYS, SCAN is safe in production with large keyspaces.
        Returns count of deleted keys.
        """
        if not self._available:
            if not await self._try_reconnect():
                return 0
        try:
            deleted = 0
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            return deleted
        except Exception as e:
            logger.warning("Redis FLUSH error for %s: %s", pattern, e)
            return 0

    async def ttl(self, key: str) -> int:
        """Return remaining TTL in seconds, -1 if no TTL, -2 if not found."""
        if not self._available:
            return -2
        try:
            return await self._redis.ttl(key)
        except Exception:
            return -2

    async def exists(self, key: str) -> bool:
        """Return True if the key exists in Redis."""
        if not self._available:
            return False
        try:
            return bool(await self._redis.exists(key))
        except Exception:
            return False

    # ── DocForge-specific helpers ──────────────────────────────────────────────

    async def get_departments(self) -> Optional[list]:
        return await self.get(KEY_DEPARTMENTS)

    async def set_departments(self, data: list) -> bool:
        return await self.set(KEY_DEPARTMENTS, data, TTL_DEPARTMENTS)

    async def get_sections(self, doc_type: str) -> Optional[dict]:
        return await self.get(KEY_SECTIONS(doc_type))

    async def set_sections(self, doc_type: str, data: dict) -> bool:
        return await self.set(KEY_SECTIONS(doc_type), data, TTL_SECTIONS)

    async def get_questions(self, sec_id: int) -> Optional[dict]:
        return await self.get(KEY_QUESTIONS(sec_id))

    async def set_questions(self, sec_id: int, data: dict) -> bool:
        return await self.set(KEY_QUESTIONS(sec_id), data, TTL_QUESTIONS)

    async def get_section_content(self, sec_id: int) -> Optional[dict]:
        return await self.get(KEY_SECTION_CONTENT(sec_id))

    async def set_section_content(self, sec_id: int, data: dict) -> bool:
        return await self.set(KEY_SECTION_CONTENT(sec_id), data, TTL_SECTION_CONTENT)

    async def invalidate_section_content(self, sec_id: int) -> bool:
        """Call after manual edit so stale content is not returned."""
        return await self.delete(KEY_SECTION_CONTENT(sec_id))

    async def get_notion_library(self) -> Optional[list]:
        return await self.get(KEY_NOTION_LIBRARY)

    async def set_notion_library(self, data: list) -> bool:
        return await self.set(KEY_NOTION_LIBRARY, data, TTL_NOTION_LIBRARY)

    async def invalidate_notion_library(self) -> bool:
        """Call after publishing a new doc so library refreshes."""
        return await self.delete(KEY_NOTION_LIBRARY)

    async def cache_stats(self) -> dict:
        """Return cache health + key counts for each namespace."""
        if not self._available:
            return {"available": False}
        try:
            info = await self._redis.info("server")
            dept_keys      = len([k async for k in self._redis.scan_iter("docforge:departments*")])
            section_keys   = len([k async for k in self._redis.scan_iter("docforge:sections:*")])
            question_keys  = len([k async for k in self._redis.scan_iter("docforge:questions:*")])
            content_keys   = len([k async for k in self._redis.scan_iter("docforge:section:*")])
            notion_keys    = len([k async for k in self._redis.scan_iter("docforge:notion*")])
            return {
                "available":       True,
                "version":         info.get("redis_version", "?"),
                "departments":     dept_keys,
                "section_types":   section_keys,
                "question_sets":   question_keys,
                "section_content": content_keys,
                "notion_cache":    notion_keys,
                "total_docforge":  dept_keys + section_keys + question_keys + content_keys + notion_keys,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}


# ── Singleton instance ─────────────────────────────────────────────────────────
cache = RedisCache()    