from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.logger import logger
from backend.api.routes import router
from backend.services.db_service import get_pool, close_pool
from backend.services.redis_service import cache
from backend.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── PostgreSQL ──────────────────────────────────────────────────────────
    try:
        await get_pool()
        logger.info("✅ PostgreSQL connection pool ready")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise

    # ── Redis (optional — app works without it) ─────────────────────────────
    redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379")
    await cache.connect(redis_url)

    yield

    # ── Cleanup ─────────────────────────────────────────────────────────────
    await close_pool()
    await cache.disconnect()


app = FastAPI(title="DocForge AI", version="3.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(router, prefix="/api")


@app.get("/health")
async def health():
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = str(e)

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "redis": "connected" if cache.is_available else "unavailable (cache disabled)",
    }


@app.get("/api/cache/stats")
async def cache_stats_endpoint():
    """Show Redis cache stats — useful for debugging."""
    return await cache.cache_stats()


@app.delete("/api/cache/flush")
async def cache_flush_endpoint():
    """Flush all DocForge cache keys."""
    count = await cache.flush_pattern("docforge:*")
    return {"flushed": count, "message": f"Deleted {count} cache keys"}