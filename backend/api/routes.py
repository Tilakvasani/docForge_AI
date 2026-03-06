from fastapi import APIRouter, HTTPException
from backend.schemas.document_schema import DocumentRequest, DocumentResponse
from backend.schemas.notion_schema import NotionPublishRequest
from backend.services.generator import generate_document
from backend.services.notion_service import publish_to_notion
from backend.services.redis_service import get_cached_doc, cache_doc, get_all_docs
from backend.core.logger import logger

router = APIRouter()

@router.post("/generate", response_model=DocumentResponse)
async def generate(request: DocumentRequest):
    """Generate an industry-ready document"""
    try:
        logger.info(f"Generating {request.doc_type} for {request.industry}")

        # Check Redis cache first (deduplication)
        cache_key = f"{request.industry}:{request.doc_type}:{request.title}"
        cached = get_cached_doc(cache_key)
        if cached:
            logger.info("Returning cached document")
            return cached

        # Generate document
        result = await generate_document(request)

        # Cache result
        cache_doc(cache_key, result)

        return result

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish")
async def publish(request: NotionPublishRequest):
    """Publish a document to Notion"""
    try:
        logger.info(f"Publishing doc {request.doc_id} to Notion")
        result = await publish_to_notion(request)
        return {"success": True, "notion_url": result}
    except Exception as e:
        logger.error(f"Publish failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/library")
async def library():
    """Get all generated documents from Redis"""
    try:
        docs = get_all_docs()
        return {"total": len(docs), "documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
