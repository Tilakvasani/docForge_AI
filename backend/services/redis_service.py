import redis
import json
from backend.core.config import settings
from backend.core.logger import logger

# Connect to Redis
try:
    r = redis.from_url(settings.REDIS_URL, decode_responses=True)
    r.ping()
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Running without cache.")
    r = None

DOC_PREFIX = "doc:"
ALL_DOCS_KEY = "all_docs"
CACHE_TTL = 60 * 60 * 24  # 24 hours

def cache_doc(key: str, doc) -> None:
    """Cache a generated document in Redis"""
    if not r:
        return
    try:
        doc_dict = doc.dict() if hasattr(doc, 'dict') else doc
        # Convert datetime to string for JSON serialization
        if 'created_at' in doc_dict and not isinstance(doc_dict['created_at'], str):
            doc_dict['created_at'] = doc_dict['created_at'].isoformat()

        r.setex(f"{DOC_PREFIX}{key}", CACHE_TTL, json.dumps(doc_dict))

        # Also add to list of all docs
        r.lpush(ALL_DOCS_KEY, json.dumps(doc_dict))
        logger.info(f"Cached doc: {key}")
    except Exception as e:
        logger.error(f"Redis cache error: {e}")

def get_cached_doc(key: str):
    """Get a cached document from Redis"""
    if not r:
        return None
    try:
        data = r.get(f"{DOC_PREFIX}{key}")
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.error(f"Redis get error: {e}")
        return None

def get_all_docs() -> list:
    """Get all generated documents from Redis"""
    if not r:
        return []
    try:
        docs = r.lrange(ALL_DOCS_KEY, 0, -1)
        return [json.loads(d) for d in docs]
    except Exception as e:
        logger.error(f"Redis list error: {e}")
        return []

def delete_doc(key: str) -> None:
    """Delete a cached document"""
    if not r:
        return
    try:
        r.delete(f"{DOC_PREFIX}{key}")
    except Exception as e:
        logger.error(f"Redis delete error: {e}")
