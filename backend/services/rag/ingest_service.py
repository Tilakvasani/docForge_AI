"""
ingest_service.py — Notion → Chunks → Embeddings → ChromaDB
=============================================================

POST /api/rag/ingest triggers this module.

Pipeline:
  1. Fetch all pages from the configured Notion database
  2. Extract + clean text from each page (blocks → plain text)
  3. Chunk text with overlap (respects paragraph boundaries)
  4. Embed chunks using Azure OpenAI text-embedding-3-large
  5. Upsert into ChromaDB with rich metadata
  6. Store ingest metadata in Redis for /status endpoint

Idempotent: chunks are upserted by deterministic ID (md5 of page_id+heading+chunk_index),
so re-running ingest updates changed content without duplicating.

force=True skips the "already ingested" Redis lock check.
"""

import asyncio
import hashlib
import json
import re
import time
from typing import Optional

from backend.core.config import settings
from backend.core.logger import logger
from backend.services.redis_service import cache

COLLECTION_NAME   = "rag_chunks"
CHUNK_SIZE        = 800     # target chars per chunk
CHUNK_OVERLAP     = 150     # overlap between consecutive chunks
MIN_CHUNK_LEN     = 80      # discard very short fragments
BATCH_EMBED_SIZE  = 64      # embed N chunks per API call
INGEST_LOCK_KEY   = "docforge:rag:ingest_lock"
INGEST_META_KEY   = "docforge:rag:ingest_meta"
INGEST_LOCK_TTL   = 1800    # 30 min — prevent concurrent ingests


# ── Singleton clients (reused across calls) ───────────────────────────────────

_embedder_instance   = None
_collection_instance = None


def _get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        from langchain_openai import AzureOpenAIEmbeddings
        _embedder_instance = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_EMB_ENDPOINT,
            api_key=settings.AZURE_OPENAI_EMB_KEY,
            azure_deployment=settings.AZURE_EMB_DEPLOYMENT,
            api_version=settings.AZURE_EMB_API_VERSION,
        )
    return _embedder_instance


def _get_collection():
    global _collection_instance
    if _collection_instance is None:
        import chromadb
        client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        _collection_instance = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection_instance


# ── Notion fetcher ─────────────────────────────────────────────────────────────

def _get_notion():
    from notion_client import Client
    return Client(auth=settings.NOTION_TOKEN)


def _fetch_all_notion_pages() -> list[dict]:
    """
    Query all pages from the configured Notion source database.
    Handles Notion pagination automatically.
    """
    notion  = _get_notion()
    db_id   = settings.NOTION_DATABASE_ID
    results = []
    cursor  = None

    while True:
        kwargs: dict = {"database_id": db_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        resp    = notion.databases.query(**kwargs)
        results.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    logger.info("Fetched %d pages from Notion DB %s", len(results), db_id)
    return results


def _fetch_page_blocks(page_id: str) -> list[dict]:
    """Recursively fetch all blocks (including children) for a Notion page."""
    notion = _get_notion()
    blocks = []
    cursor = None

    while True:
        kwargs: dict = {"block_id": page_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        resp   = notion.blocks.children.list(**kwargs)
        blocks.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    # Recurse into child blocks (toggles, callouts, etc.)
    expanded = []
    for block in blocks:
        expanded.append(block)
        if block.get("has_children"):
            try:
                child_blocks = _fetch_page_blocks(block["id"])
                expanded.extend(child_blocks)
            except Exception as e:
                logger.warning("Failed to fetch children of block %s: %s", block["id"], e)

    return expanded


# ── Text extraction ────────────────────────────────────────────────────────────

def _rich_text_to_str(rich_text_items: list) -> str:
    return "".join(t.get("plain_text", "") for t in rich_text_items)


def _block_to_text(block: dict) -> tuple[str, str]:
    """
    Extract (heading, text) from a Notion block.
    Returns ("", "") for unsupported block types.
    """
    btype = block.get("type", "")
    bdata = block.get(btype, {})

    heading = ""
    text    = ""

    if btype in ("heading_1", "heading_2", "heading_3"):
        heading = _rich_text_to_str(bdata.get("rich_text", []))
        text    = heading

    elif btype == "paragraph":
        text = _rich_text_to_str(bdata.get("rich_text", []))

    elif btype in ("bulleted_list_item", "numbered_list_item", "to_do"):
        text = "• " + _rich_text_to_str(bdata.get("rich_text", []))

    elif btype == "toggle":
        text = _rich_text_to_str(bdata.get("rich_text", []))

    elif btype == "quote":
        text = "> " + _rich_text_to_str(bdata.get("rich_text", []))

    elif btype == "callout":
        text = _rich_text_to_str(bdata.get("rich_text", []))

    elif btype == "code":
        text = _rich_text_to_str(bdata.get("rich_text", []))

    elif btype == "table_row":
        cells = bdata.get("cells", [])
        row   = " | ".join(_rich_text_to_str(cell) for cell in cells)
        text  = row

    elif btype == "divider":
        text = "---"

    return heading, text


def _extract_page_content(page: dict, blocks: list[dict]) -> dict:
    """
    Extract metadata and full text sections from a Notion page + its blocks.
    Returns a dict with title, doc_type, department, text sections with headings.
    """
    props = page.get("properties", {})

    def _prop_text(prop_name: str) -> str:
        prop  = props.get(prop_name, {})
        ptype = prop.get("type", "")
        if ptype == "title":
            return _rich_text_to_str(prop.get("title", []))
        if ptype == "rich_text":
            return _rich_text_to_str(prop.get("rich_text", []))
        if ptype == "select":
            sel = prop.get("select") or {}
            return sel.get("name", "")
        if ptype == "multi_select":
            return ", ".join(s.get("name", "") for s in prop.get("multi_select", []))
        return ""

    # Try common property name variants for title
    title = (
        _prop_text("Name") or _prop_text("Title") or _prop_text("Document Name")
        or page.get("url", "").split("/")[-1].replace("-", " ")
    )
    doc_type   = _prop_text("Doc Type") or _prop_text("Type") or _prop_text("Document Type") or ""
    department = _prop_text("Department") or _prop_text("Team") or ""
    version    = _prop_text("Version") or "v1"
    status     = _prop_text("Status") or ""

    # Walk blocks and build sections
    sections: list[dict] = []
    current_heading  = title or "General"
    current_texts: list[str] = []

    for block in blocks:
        heading, text = _block_to_text(block)
        if not text.strip():
            continue

        if heading:
            # Save previous section if it has content
            if current_texts:
                sections.append({
                    "heading": current_heading,
                    "text":    "\n".join(current_texts).strip(),
                })
                current_texts = []
            current_heading = heading
        else:
            current_texts.append(text)

    # Save last section
    if current_texts:
        sections.append({
            "heading": current_heading,
            "text":    "\n".join(current_texts).strip(),
        })

    return {
        "page_id":    page["id"],
        "title":      title,
        "doc_type":   doc_type,
        "department": department,
        "version":    version,
        "status":     status,
        "url":        page.get("url", ""),
        "sections":   sections,
    }


# ── Chunker ────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks, preferring paragraph/sentence boundaries.
    """
    if not text or len(text) < MIN_CHUNK_LEN:
        return [text] if text.strip() else []

    # Split on double newlines first (paragraphs)
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # If single paragraph exceeds size, split by sentence
            if len(para) > size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sent_buf  = ""
                for sent in sentences:
                    if len(sent_buf) + len(sent) + 1 <= size:
                        sent_buf = (sent_buf + " " + sent).strip()
                    else:
                        if sent_buf:
                            chunks.append(sent_buf)
                        sent_buf = sent
                if sent_buf:
                    current = sent_buf
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # Apply overlap: prepend tail of previous chunk to next
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail    = chunks[i - 1][-overlap:]
            merged  = (tail + " " + chunks[i]).strip()
            overlapped.append(merged)
        chunks = overlapped

    return [c for c in chunks if len(c) >= MIN_CHUNK_LEN]


# ── Chunk ID generator ─────────────────────────────────────────────────────────

def _chunk_id(page_id: str, heading: str, chunk_index: int) -> str:
    raw = f"{page_id}::{heading}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Citation formatter ─────────────────────────────────────────────────────────

def _format_citation(title: str, heading: str, doc_type: str) -> str:
    parts = [title]
    if doc_type and doc_type.lower() != title.lower():
        parts.append(doc_type)
    if heading and heading.lower() != title.lower():
        parts.append(f"§ {heading}")
    return " › ".join(parts)


# ── Embedding + upsert ─────────────────────────────────────────────────────────

def _embed_and_upsert(chunks_batch: list[dict], collection) -> int:
    """
    Embed a batch of chunk dicts and upsert into ChromaDB.
    Each chunk dict has: id, text, metadata.
    Returns number of chunks upserted.
    """
    if not chunks_batch:
        return 0

    embedder = _get_embedder()
    texts    = [c["text"] for c in chunks_batch]

    try:
        embeddings = embedder.embed_documents(texts)
    except Exception as e:
        logger.error("Embedding failed for batch of %d: %s", len(texts), e)
        return 0

    ids        = [c["id"]       for c in chunks_batch]
    metadatas  = [c["metadata"] for c in chunks_batch]

    try:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks_batch)
    except Exception as e:
        logger.error("ChromaDB upsert failed: %s", e)
        return 0


# ── Main ingest function ───────────────────────────────────────────────────────

async def ingest_from_notion(force: bool = False) -> dict:
    """
    Full ingest pipeline:
      Notion pages → extract → chunk → embed → ChromaDB upsert

    Args:
        force: if True, skip the Redis lock check (allow re-ingest)

    Returns:
        dict with total_docs, total_chunks, elapsed_s, skipped
    """
    # ── Lock check ──────────────────────────────────────────────────────────────
    if not force:
        locked = await cache.exists(INGEST_LOCK_KEY)
        if locked:
            logger.warning("Ingest already in progress — skipping (use force=True to override)")
            return {
                "status":       "skipped",
                "reason":       "ingest_locked",
                "total_docs":   0,
                "total_chunks": 0,
            }

    # Set lock
    await cache.set(INGEST_LOCK_KEY, "1", ttl=INGEST_LOCK_TTL)
    t_start = time.time()

    try:
        collection = _get_collection()

        # ── Step 1: Fetch pages ────────────────────────────────────────────────
        logger.info("Fetching pages from Notion...")
        loop  = asyncio.get_event_loop()
        pages = await loop.run_in_executor(None, _fetch_all_notion_pages)

        if not pages:
            logger.warning("No pages returned from Notion")
            return {
                "status":       "done",
                "total_docs":   0,
                "total_chunks": 0,
                "elapsed_s":    round(time.time() - t_start, 1),
            }

        total_docs   = len(pages)
        total_chunks = 0
        all_chunks:  list[dict] = []

        # ── Step 2: Extract + chunk ────────────────────────────────────────────
        for page in pages:
            page_id = page["id"]
            try:
                blocks  = await loop.run_in_executor(None, _fetch_page_blocks, page_id)
                content = _extract_page_content(page, blocks)
            except Exception as e:
                logger.error("Failed to extract page %s: %s", page_id, e)
                continue

            title      = content["title"]
            doc_type   = content["doc_type"]
            department = content["department"]
            version    = content["version"]
            url        = content["url"]

            for section in content["sections"]:
                heading = section["heading"]
                text    = section["text"]
                chunks  = _chunk_text(text)
                citation = _format_citation(title, heading, doc_type)

                for idx, chunk_text in enumerate(chunks):
                    cid = _chunk_id(page_id, heading, idx)
                    all_chunks.append({
                        "id":   cid,
                        "text": chunk_text,
                        "metadata": {
                            "notion_page_id": page_id,
                            "doc_title":      title,
                            "doc_type":       doc_type,
                            "department":     department,
                            "version":        version,
                            "heading":        heading,
                            "chunk_index":    idx,
                            "citation":       citation,
                            "source_url":     url,
                        },
                    })

        logger.info("Prepared %d chunks from %d pages", len(all_chunks), total_docs)

        # ── Step 3: Embed + upsert in batches ──────────────────────────────────
        for i in range(0, len(all_chunks), BATCH_EMBED_SIZE):
            batch        = all_chunks[i : i + BATCH_EMBED_SIZE]
            upserted     = await loop.run_in_executor(
                None, _embed_and_upsert, batch, collection
            )
            total_chunks += upserted
            logger.info(
                "Embedded batch %d/%d — %d/%d chunks upserted",
                i // BATCH_EMBED_SIZE + 1,
                (len(all_chunks) + BATCH_EMBED_SIZE - 1) // BATCH_EMBED_SIZE,
                total_chunks,
                len(all_chunks),
            )

        elapsed = round(time.time() - t_start, 1)

        # ── Step 4: Save ingest metadata to Redis ──────────────────────────────
        meta = {
            "total_docs":   total_docs,
            "total_chunks": total_chunks,
            "elapsed_s":    elapsed,
            "ingested_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        await cache.set(INGEST_META_KEY, meta)

        logger.info(
            "Ingest complete: %d docs, %d chunks, %.1fs",
            total_docs, total_chunks, elapsed,
        )
        return {"status": "done", **meta}

    except Exception as e:
        logger.error("Ingest pipeline failed: %s", e, exc_info=True)
        raise

    finally:
        # Always release the lock
        await cache.delete(INGEST_LOCK_KEY)