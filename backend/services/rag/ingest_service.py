"""
ingest_service.py  — Production RAG Ingestion
==============================================
Flow: Notion Docs → Clean → Smart Chunk → Embed → ChromaDB

Chunking Strategy:
  - Split at headings (section boundaries)
  - Short sections  (<200 tokens) → 1 chunk
  - Medium sections (200-600 tokens) → 2-3 chunks  (400 tokens, 50 overlap)
  - Long sections   (>600 tokens)   → 3-5 chunks   (400 tokens, 50 overlap)
  
Each chunk stores full metadata:
  {doc_title, section, doc_type, department, version, citation}
"""

import hashlib
import httpx
from backend.core.config import settings
from backend.core.logger import logger
from backend.services.redis_service import cache

NOTION_API_URL  = "https://api.notion.com/v1"
COLLECTION_NAME = "rag_chunks"
CHUNK_SIZE      = 400    # tokens (approx 4 chars per token)
CHUNK_OVERLAP   = 50
TTL_INGEST_LOCK = 86400  # 24 hours
KEY_INGEST_LOCK = "docforge:rag:ingest_lock"
KEY_INGEST_META = "docforge:rag:ingest_meta"


# ── Notion helpers ────────────────────────────────────────────────────────────

def _notion_headers():
    return {
        "Authorization":  f"Bearer {settings.NOTION_API_KEY}",
        "Content-Type":   "application/json",
        "Notion-Version": "2022-06-28",
    }


async def _fetch_all_pages():
    pages, cursor = [], None
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            body = {"page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            resp = await client.post(
                f"{NOTION_API_URL}/databases/{settings.NOTION_DATABASE_ID}/query",
                headers=_notion_headers(), json=body)
            if resp.status_code != 200:
                logger.error("Notion DB error: %s %s", resp.status_code, resp.text[:200])
                break
            data = resp.json()
            pages.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            # Redis rate limiting: respect Notion 3 req/sec limit
            await cache.set("docforge:rag:notion_rate", 1, ttl=1)
    logger.info("Fetched %d pages from Notion", len(pages))
    return pages


async def _fetch_page_blocks(page_id: str):
    blocks, cursor = [], None
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            resp = await client.get(
                f"{NOTION_API_URL}/blocks/{page_id}/children",
                headers=_notion_headers(), params=params)
            if resp.status_code != 200:
                break
            data = resp.json()
            blocks.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
    return blocks


# ── Block → text ──────────────────────────────────────────────────────────────

def _block_to_text(block: dict) -> str:
    btype = block.get("type", "")
    bdata = block.get(btype, {})
    if btype == "table_row":
        return " | ".join(
            "".join(rt.get("plain_text", "") for rt in cell)
            for cell in bdata.get("cells", []))
    return "".join(rt.get("plain_text", "") for rt in bdata.get("rich_text", []))


def _is_heading(block: dict):
    btype = block.get("type", "")
    if btype in ("heading_1", "heading_2", "heading_3"):
        return True, _block_to_text(block)
    return False, ""


# ── Page metadata ─────────────────────────────────────────────────────────────

def _page_meta(page: dict) -> dict:
    props = page.get("properties", {})

    def get_text(k):
        p     = props.get(k, {})
        items = (p.get("title", []) if p.get("type") == "title"
                 else p.get("rich_text", []))
        return "".join(i.get("text", {}).get("content", "") for i in items)

    def get_select(k):
        sel = props.get(k, {}).get("select")
        return sel.get("name", "") if sel else ""

    return {
        "notion_page_id": page["id"].replace("-", ""),
        "doc_title":      get_text("Title"),
        "doc_type":       get_text("Doc Type"),
        "department":     get_select("Department"),
        "version":        get_text("Version") or "v1",
    }


# ── Smart chunker ─────────────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars)."""
    return len(text) // 4


def _split_text(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by words.
    Short text (<= chunk_size tokens) → single chunk.
    """
    if _approx_tokens(text) <= chunk_size:
        return [text.strip()]

    words    = text.split()
    char_limit  = chunk_size * 4
    overlap_chars = overlap * 4
    chunks   = []
    start    = 0

    while start < len(words):
        # Build chunk up to char_limit
        chunk_words = []
        char_count  = 0
        i = start
        while i < len(words) and char_count + len(words[i]) < char_limit:
            chunk_words.append(words[i])
            char_count += len(words[i]) + 1
            i += 1

        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)

        if i >= len(words):
            break

        # Move start forward accounting for overlap
        overlap_words = 0
        overlap_count = 0
        for w in reversed(chunk_words):
            if overlap_count >= overlap_chars:
                break
            overlap_words += 1
            overlap_count += len(w) + 1

        start = i - overlap_words

    return chunks if chunks else [text.strip()]


def _chunk_blocks(blocks: list) -> list[dict]:
    """
    1. Split at heading boundaries → sections
    2. Apply token-based chunking within each section
    Returns list of {heading, content, chunk_index}
    """
    # Step 1: collect sections
    sections          = []
    current_heading   = "Introduction"
    current_lines     = []

    for block in blocks:
        is_h, heading_text = _is_heading(block)
        if is_h:
            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_lines).strip(),
                })
            current_heading = heading_text
            current_lines   = []
        else:
            text = _block_to_text(block)
            if text.strip():
                current_lines.append(text.strip())

    if current_lines:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_lines).strip(),
        })

    # Step 2: split long sections into chunks
    chunks = []
    for sec in sections:
        if not sec["content"] or len(sec["content"]) < 20:
            continue
        sub_chunks = _split_text(sec["content"])
        for i, sub in enumerate(sub_chunks):
            chunks.append({
                "heading":     sec["heading"],
                "content":     sub,
                "chunk_index": i,
            })

    return chunks


# ── Embedder + ChromaDB ───────────────────────────────────────────────────────

def _get_embedder():
    from langchain_openai import AzureOpenAIEmbeddings
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.AZURE_EMB_ENDPOINT,
        api_key=settings.AZURE_OPENAI_EMB_KEY,
        azure_deployment=settings.AZURE_EMB_DEPLOYMENT,
        api_version=settings.AZURE_EMB_API_VERSION,
    )


def _get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Main ingest ───────────────────────────────────────────────────────────────

async def ingest_from_notion(force: bool = False) -> dict:
    """
    Full pipeline: Notion → Chunks → Embeddings → ChromaDB
    Redis lock prevents re-ingest within 5 min.
    """
    if not force and await cache.exists(KEY_INGEST_LOCK):
        meta = await cache.get(KEY_INGEST_META) or {}
        logger.info("Ingest skipped — lock active")
        return {**meta, "skipped": True}

    logger.info("Starting ingest (force=%s)", force)
    collection = _get_collection()
    embedder   = _get_embedder()

    pages        = await _fetch_all_pages()
    total_chunks = 0
    skipped      = 0

    for page in pages:
        meta = _page_meta(page)
        if not meta["doc_title"]:
            skipped += 1
            continue

        blocks = await _fetch_page_blocks(page["id"])
        chunks = _chunk_blocks(blocks)
        if not chunks:
            skipped += 1
            continue

        page_id = meta["notion_page_id"]

        # Delete stale chunks for this page
        try:
            existing = collection.get(where={"notion_page_id": page_id})
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

        # Embed in batches of 20
        texts  = [c["content"] for c in chunks]
        embeds = []
        for i in range(0, len(texts), 20):
            batch = embedder.embed_documents(texts[i:i+20])
            embeds.extend(batch)

        ids, docs, metas, emb_list = [], [], [], []
        for i, (chunk, emb) in enumerate(zip(chunks, embeds)):
            chunk_id = hashlib.md5(f"{page_id}_{i}".encode()).hexdigest()
            citation = f"{meta['doc_title']} → {chunk['heading']}"
            ids.append(chunk_id)
            docs.append(chunk["content"])
            emb_list.append(emb)
            metas.append({
                "notion_page_id": page_id,
                "doc_title":      meta["doc_title"],
                "doc_type":       meta["doc_type"],
                "department":     meta["department"],
                "version":        meta["version"],
                "heading":        chunk["heading"],
                "chunk_index":    chunk["chunk_index"],
                "citation":       citation,
            })

        collection.upsert(
            ids=ids, documents=docs,
            embeddings=emb_list, metadatas=metas,
        )
        total_chunks += len(ids)
        logger.info("Ingested '%s' → %d chunks", meta["doc_title"], len(ids))

    result = {
        "total_docs":   len(pages) - skipped,
        "total_chunks": total_chunks,
        "skipped":      skipped,
    }
    await cache.set(KEY_INGEST_LOCK, 1,      ttl=TTL_INGEST_LOCK)
    await cache.set(KEY_INGEST_META, result,  ttl=86400)
    logger.info("Ingest complete: %d docs, %d chunks", result["total_docs"], total_chunks)
    return result