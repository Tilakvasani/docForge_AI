import time
import httpx
from backend.core.config import settings
from backend.core.logger import logger
from backend.schemas.notion_schema import NotionPublishRequest

NOTION_API_URL = "https://api.notion.com/v1"

def get_headers():
    return {
        "Authorization": f"Bearer {settings.NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

def chunk_content(content: str, chunk_size: int = 1900) -> list:
    """Split content into Notion-safe chunks (max 2000 chars per block)"""
    return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

def with_backoff(fn, retries: int = 5):
    """Exponential backoff for Notion rate limits"""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if "rate_limited" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt
                logger.warning(f"Rate limited. Waiting {wait}s before retry {attempt+1}")
                time.sleep(wait)
            else:
                raise e
    raise Exception("Max retries exceeded for Notion API")

async def publish_to_notion(request: NotionPublishRequest) -> str:
    """Publish document to Notion database"""
    logger.info(f"Publishing to Notion: {request.title}")

    chunks = chunk_content(request.content)

    # Build Notion blocks from content chunks
    blocks = []
    for chunk in chunks:
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": chunk}}]
            }
        })

    payload = {
        "parent": {"database_id": settings.NOTION_DATABASE_ID},
        "properties": {
            "Title": {
                "title": [{"text": {"content": request.title}}]
            },
            "Industry": {
                "select": {"name": request.industry}
            },
            "Doc Type": {
                "select": {"name": request.doc_type}
            },
            "Version": {
                "rich_text": [{"text": {"content": request.version}}]
            },
            "Created By": {
                "rich_text": [{"text": {"content": request.created_by}}]
            },
            "Tags": {
                "multi_select": [{"name": tag} for tag in request.tags]
            }
        },
        "children": blocks
    }

    async with httpx.AsyncClient() as client:
        def create_page():
            return client.post(
                f"{NOTION_API_URL}/pages",
                headers=get_headers(),
                json=payload,
                timeout=30
            )

        response = with_backoff(create_page)

        if hasattr(response, 'status_code') and response.status_code == 200:
            data = response.json()
            notion_url = data.get("url", "")
            logger.info(f"Published to Notion: {notion_url}")
            return notion_url
        else:
            raise Exception(f"Notion API error: {response.text if hasattr(response, 'text') else response}")
